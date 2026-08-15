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
