from pathlib import Path

from PIL import Image

from ocr_benchmark.core.registry import create_adapter
from ocr_benchmark.models.monkey_ocr_native import MonkeyOCRv2NativeAdapter, _generated_text


def test_native_monkey_adapter_is_registered_without_importing_transformers():
    adapter = create_adapter(
        "monkey_ocr_v2_b_parsing_native",
        {"model_id": "local/monkey", "device_map": "cpu"},
    )
    assert isinstance(adapter, MonkeyOCRv2NativeAdapter)
    assert adapter.metadata()["runtime"] == "transformers-native"


def test_generated_text_handles_chat_pipeline_shape():
    output = [
        {
            "generated_text": [
                {"role": "user", "content": "ignored"},
                {"role": "assistant", "content": '{"raw_text":"SKU: ABC","fields":{"SKU":"ABC"}}'},
            ]
        }
    ]
    assert _generated_text(output) == '{"raw_text":"SKU: ABC","fields":{"SKU":"ABC"}}'


def test_native_monkey_predict_uses_local_image_and_parses_fields(tmp_path: Path):
    image_path = tmp_path / "label.jpg"
    Image.new("RGB", (8, 8), color="white").save(image_path)
    calls = []

    class FakePipeline:
        def __call__(self, *, text, max_new_tokens):
            calls.append((text, max_new_tokens))
            image_path_value = text[0]["content"][0]["image"]
            assert image_path_value == str(image_path.resolve())
            assert text[0]["content"][0]["max_pixels"] == 1003520
            return [{"generated_text": "```json\n{\"raw_text\":\"SKU: ABC\",\"fields\":{\"SKU\":\"ABC\"}}\n```"}]

    adapter = MonkeyOCRv2NativeAdapter(max_new_tokens=17)
    adapter._pipeline = FakePipeline()
    prediction = adapter.predict(image_path)

    assert prediction.raw_text == "SKU: ABC"
    assert prediction.fields == {"SKU": "ABC"}
    assert prediction.timing.inference_ms is not None
    assert calls and calls[0][1] == 17
