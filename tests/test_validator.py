import json

from PIL import Image

from ocr_benchmark.data.validator import load_and_validate_dataset
from ocr_benchmark.data.validator import DatasetValidationError


def test_dataset_validator_accepts_single_label(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (10, 10), "white").save(image_dir / "one.png")
    gt = tmp_path / "ground_truth.json"
    gt.write_text(json.dumps({"samples": [{"image": "one.png", "fields": {"sku": "ABC-123"}}]}))
    dataset = load_and_validate_dataset(image_dir, gt)
    assert len(dataset.samples) == 1


def test_dataset_validator_accepts_top_level_list_and_mixed_language_tag(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (10, 10), "white").save(image_dir / "one.png")
    gt = tmp_path / "ground_truth.json"
    gt.write_text(json.dumps([{"image": "one.png", "tags": ["vietnamese_mixed"], "required_fields": ["sku"], "fields": {"sku": "x"}, "notes": "synthetic"}]))
    dataset = load_and_validate_dataset(image_dir, gt)
    assert len(dataset.samples) == 1


def test_dataset_validator_rejects_multi_label_sample(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (10, 10), "white").save(image_dir / "many.png")
    gt = tmp_path / "ground_truth.json"
    gt.write_text(json.dumps({"samples": [{"image": "many.png", "label_count": 2, "fields": {"sku": "x"}}]}))
    try:
        load_and_validate_dataset(image_dir, gt)
    except DatasetValidationError as exc:
        assert "exactly one physical label" in str(exc)
    else:
        raise AssertionError("multi-label sample was accepted")


def test_dataset_validator_rejects_unknown_tag(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (10, 10), "white").save(image_dir / "one.png")
    gt = tmp_path / "ground_truth.json"
    gt.write_text(json.dumps({"samples": [{"image": "one.png", "tags": ["unknown"], "fields": {"sku": "x"}}]}))
    try:
        load_and_validate_dataset(image_dir, gt)
    except DatasetValidationError as exc:
        assert "invalid tags" in str(exc)
    else:
        raise AssertionError("unknown tag was accepted")


def test_dataset_validator_rejects_absolute_path(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    outside = tmp_path / "outside.png"
    Image.new("RGB", (10, 10), "white").save(outside)
    gt = tmp_path / "ground_truth.json"
    gt.write_text(json.dumps({"samples": [{"image": str(outside), "fields": {"sku": "x"}}]}))
    try:
        load_and_validate_dataset(image_dir, gt)
    except DatasetValidationError as exc:
        assert "relative" in str(exc)
    else:
        raise AssertionError("absolute image path was accepted")
