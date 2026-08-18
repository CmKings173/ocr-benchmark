from ocr_benchmark.core.registry import create_adapter
from ocr_benchmark.models.dots_mocr import DOTS_IMAGE_PREFIX, DOTS_MOCR_JSON_PROMPT, DotsMOCRAdapter
from ocr_benchmark.models.hunyuan_ocr import HunyuanOCRAdapter
from ocr_benchmark.models.unlimited_ocr import UnlimitedOCRAdapter


def test_new_adapters_are_registered_with_documented_defaults():
    assert isinstance(
        create_adapter("hunyuan_ocr_1_5", {"endpoint": "http://127.0.0.1:8107/v1", "verify_model": False}),
        HunyuanOCRAdapter,
    )
    assert isinstance(
        create_adapter("unlimited_ocr", {"endpoint": "http://127.0.0.1:8106/v1", "verify_model": False}),
        UnlimitedOCRAdapter,
    )
    assert isinstance(
        create_adapter("dots_mocr", {"endpoint": "http://127.0.0.1:8108/v1", "verify_model": False}),
        DotsMOCRAdapter,
    )


def test_official_prompt_profiles_are_not_generic_json_prompts():
    hunyuan = HunyuanOCRAdapter(endpoint="http://127.0.0.1:8107/v1", verify_model=False)
    unlimited = UnlimitedOCRAdapter(endpoint="http://127.0.0.1:8106/v1", verify_model=False)
    dots = DotsMOCRAdapter(endpoint="http://127.0.0.1:8108/v1", verify_model=False)
    assert hunyuan.task_type == "spotting_json"
    assert "JSON 数组" in hunyuan.prompt
    assert unlimited.prompt == "<image>document parsing."
    assert "Final Output" in dots.prompt and "single JSON object" in dots.prompt
    assert dots.prompt == DOTS_MOCR_JSON_PROMPT
    assert hunyuan.metadata()["task_type"] == "spotting_json"
    assert unlimited.metadata()["image_mode"] == "gundam"
    assert dots.metadata()["prompt_mode"] == "prompt_layout_all_en"
    assert DotsMOCRAdapter(
        endpoint="http://127.0.0.1:8108/v1",
        prompt_mode="prompt_ocr",
        verify_model=False,
    ).prompt == "Extract the text content from this image."


def test_hunyuan_spotting_json_parser_extracts_label_fields():
    adapter = HunyuanOCRAdapter(endpoint="http://127.0.0.1:8107/v1", verify_model=False)
    raw_text, fields = adapter._parse_content('[{"box":[1,2,3,4],"text":"SKU: ABC-123"},{"text":"QTY: 10 PCS"}]')
    assert "SKU: ABC-123" in raw_text
    assert fields["sku"] == "ABC-123"
    assert fields["quantity"] == "10"
    assert fields["unit"] == "PCS"


def test_unlimited_parser_removes_official_grounding_markers():
    adapter = UnlimitedOCRAdapter(endpoint="http://127.0.0.1:8106/v1", verify_model=False)
    raw_text, fields = adapter._parse_content("<|det|>text [0,0,10,10]<|/det|>SKU: ABC-123\n<|ref|>LOT<|/ref|>: L-1")
    assert "<|det|>" not in raw_text
    assert fields["sku"] == "ABC-123"
    assert fields["lot"] == "L-1"


def test_dots_layout_json_parser_reads_layout_text():
    adapter = DotsMOCRAdapter(endpoint="http://127.0.0.1:8108/v1", verify_model=False)
    raw_text, fields = adapter._parse_content({"layout": [
        {"category": "Text", "bbox": [0, 0, 10, 10], "text": "SKU: ABC-123"},
        {"category": "Text", "bbox": [0, 20, 10, 30], "text": "PO: PO-9"},
    ]})
    assert "SKU: ABC-123" in raw_text
    assert fields == {"sku": "ABC-123", "po_number": "PO-9"}


def test_dots_parser_accepts_official_list_and_wrapped_cell_shapes():
    adapter = DotsMOCRAdapter(endpoint="http://127.0.0.1:8108/v1", verify_model=False)
    official_cells = [
        {"category": "Text", "bbox": [0, 0, 10, 10], "text": "SKU: ABC-123"},
        {"category": "Text", "bbox": [0, 20, 10, 30], "text": "PO: PO-9"},
    ]
    raw_text, fields = adapter._parse_content(official_cells)
    assert raw_text == "SKU: ABC-123\nPO: PO-9"
    assert fields == {"sku": "ABC-123", "po_number": "PO-9"}

    _, wrapped_fields = adapter._parse_content({"cells": official_cells})
    assert wrapped_fields == {"sku": "ABC-123", "po_number": "PO-9"}


def test_new_adapters_build_provider_specific_request_bodies():
    encoded = "dGVzdA=="

    hunyuan = HunyuanOCRAdapter(endpoint="http://127.0.0.1:8107/v1", verify_model=False)
    hbody = hunyuan._build_request_body(mime="image/jpeg", encoded=encoded)
    assert hbody["model"] == "tencent/HunyuanOCR"
    assert hbody["messages"][0] == {"role": "system", "content": ""}
    assert hbody["messages"][1]["content"][0]["type"] == "image_url"
    assert hbody["top_k"] == -1

    unlimited = UnlimitedOCRAdapter(endpoint="http://127.0.0.1:8106/v1", verify_model=False)
    ubody = unlimited._build_request_body(mime="image/jpeg", encoded=encoded)
    assert ubody["messages"][0]["content"][0]["text"].startswith("<image>")
    assert ubody["skip_special_tokens"] is False
    assert ubody["vllm_xargs"] == {"ngram_size": 35, "window_size": 128}

    dots = DotsMOCRAdapter(endpoint="http://127.0.0.1:8108/v1", verify_model=False)
    dbody = dots._build_request_body(mime="image/jpeg", encoded=encoded)
    assert dbody["model"] == "rednote-hilab/dots.mocr"
    assert dbody["max_completion_tokens"] == 32768
    assert dbody["temperature"] == 0.1
    assert dbody["top_p"] == 0.9
    assert dbody["messages"][0]["content"][1]["text"].startswith(DOTS_IMAGE_PREFIX)
    assert "single JSON object" in dbody["messages"][0]["content"][1]["text"]
    assert dots.metadata()["prompt_sha256"]
