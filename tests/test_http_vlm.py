import pytest

from ocr_benchmark.models.glm_ocr import GLMOCRAdapter
from ocr_benchmark.models.http_vlm import _parse_structured_content


def test_remote_vlm_endpoint_requires_explicit_opt_in():
    adapter = GLMOCRAdapter(endpoint="https://example.invalid/v1")
    with pytest.raises(RuntimeError, match="remote endpoint"):
        adapter.load()


def test_vlm_parser_accepts_fenced_json_and_list_fields():
    content = """```json
{"raw_text":"SKU: ABC-123","fields":[{"name":"SKU","value":"ABC-123"}]}
```"""
    raw_text, fields = _parse_structured_content(content)
    assert raw_text == "SKU: ABC-123"
    assert fields == {"SKU": "ABC-123"}


def test_vlm_parser_accepts_content_blocks():
    raw_text, fields = _parse_structured_content(
        [{"type": "text", "text": '{"raw_text":"SKU: ABC","fields":{"SKU":"ABC"}}'}]
    )
    assert raw_text == "SKU: ABC"
    assert fields == {"SKU": "ABC"}


def test_local_vlm_load_rejects_endpoint_serving_a_different_model(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, _limit):
            return b'{"data":[{"id":"Qwen/Qwen2-VL-7B-Instruct"}]}'

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response())
    adapter = GLMOCRAdapter(endpoint="http://127.0.0.1:8104/v1", model="MonkeyOCRv2-B-Parsing")
    with pytest.raises(RuntimeError, match="MODEL_MISMATCH"):
        adapter.load()


def test_local_vlm_load_accepts_server_model_basename(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, _limit):
            return b'{"models":[{"name":"zenosai/MonkeyOCRv2-B-Parsing"}]}'

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response())
    adapter = GLMOCRAdapter(endpoint="http://127.0.0.1:8104/v1", model="MonkeyOCRv2-B-Parsing")
    adapter.load()
    assert adapter.server_model_id == "zenosai/MonkeyOCRv2-B-Parsing"
