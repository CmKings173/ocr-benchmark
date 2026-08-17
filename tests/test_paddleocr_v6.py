from pathlib import Path

import pytest

from ocr_benchmark.models.paddleocr_v6 import PPOCRv6Adapter


class _FakePipeline:
    def __init__(self, output):
        self.output = output

    def predict(self, input: str):
        return self.output


def test_ppocr_v6_predict_normalizes_text_scores_and_boxes(tmp_path: Path):
    image = tmp_path / "label.jpg"
    image.write_bytes(b"not decoded by the fake pipeline")
    adapter = PPOCRv6Adapter(device="cpu")
    adapter.pipeline = _FakePipeline(
        [{
            "rec_texts": ["SKU: ABC-123"],
            "rec_scores": [0.98],
            "rec_boxes": [[[1, 2], [11, 2], [11, 12], [1, 12]]],
        }]
    )

    prediction = adapter.predict(image)

    assert prediction.raw_text == "SKU: ABC-123"
    assert len(prediction.detections) == 1
    assert prediction.detections[0].confidence == pytest.approx(0.98)
    assert prediction.detections[0].bbox == [1.0, 2.0, 11.0, 12.0]


def test_ppocr_v6_accepts_single_mapping_output(tmp_path: Path):
    image = tmp_path / "label.jpg"
    image.write_bytes(b"not decoded by the fake pipeline")
    adapter = PPOCRv6Adapter(device="cpu")
    adapter.pipeline = _FakePipeline({"rec_texts": ["HELLO"]})

    prediction = adapter.predict(image)

    assert prediction.raw_text == "HELLO"


@pytest.mark.parametrize("output", [[], [{}], [{"markdown": ""}]])
def test_ppocr_v6_rejects_empty_or_schema_invalid_output(tmp_path: Path, output):
    image = tmp_path / "label.jpg"
    image.write_bytes(b"not decoded by the fake pipeline")
    adapter = PPOCRv6Adapter(device="cpu")
    adapter.pipeline = _FakePipeline(output)

    with pytest.raises(RuntimeError, match="INVALID_OUTPUT"):
        adapter.predict(image)
