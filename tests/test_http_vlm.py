import json

import pytest
from PIL import Image

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


def test_vlm_load_and_predict_round_trip(monkeypatch, tmp_path):
    image_path = tmp_path / "label.jpg"
    Image.new("RGB", (8, 8), color="white").save(image_path)
    responses = [
        b'{"data":[{"id":"glm-ocr"}]}',
        b'{"choices":[{"message":{"content":"```json\\n{\\"raw_text\\":\\"SKU: ABC\\",\\"fields\\":{\\"SKU\\":\\"ABC\\"}}\\n```"}}]}',
    ]
    requests = []

    class Response:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, _limit):
            return self.body

    def fake_urlopen(request, **kwargs):
        requests.append((request, kwargs))
        return Response(responses.pop(0))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    adapter = GLMOCRAdapter(endpoint="http://127.0.0.1:8080/v1", model="glm-ocr")
    adapter.load()
    prediction = adapter.predict(image_path)

    assert prediction.raw_text == "SKU: ABC"
    assert prediction.fields == {"sku": "ABC"}
    assert prediction.timing.inference_ms is not None
    assert len(requests) == 2
    body = json.loads(requests[1][0].data.decode("utf-8"))
    assert body["model"] == "glm-ocr"
    assert body["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert body["messages"][0]["content"][0]["text"].startswith("Text Recognition:")
    assert prediction.metadata["prompt_profile"] == "glm_text_recognition_json_v1"


def test_vlm_predict_rejects_empty_message_content(monkeypatch, tmp_path):
    image_path = tmp_path / "label.png"
    Image.new("RGB", (8, 8), color="white").save(image_path)
    responses = [
        b'{"data":[{"id":"glm-ocr"}]}',
        b'{"choices":[{"message":{"content":""}}]}',
    ]

    class Response:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, _limit):
            return self.body

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response(responses.pop(0)))
    adapter = GLMOCRAdapter(endpoint="http://127.0.0.1:8080/v1", model="glm-ocr")
    adapter.load()
    with pytest.raises(RuntimeError, match="empty message content"):
        adapter.predict(image_path)
