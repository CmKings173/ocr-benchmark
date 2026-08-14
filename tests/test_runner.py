import json

from PIL import Image

from ocr_benchmark.benchmark.runner import run_accuracy_pass, run_performance_pass
from ocr_benchmark.data.validator import load_and_validate_dataset
from ocr_benchmark.reporting.export import export_results


def test_mock_two_pass_and_export(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (8, 8), "white").save(image_root / "one.png")
    gt_path = tmp_path / "ground_truth.json"
    gt_path.write_text(json.dumps({"samples": [{"image": "one.png", "fields": {"sku": "ABC"}, "full_text": "ABC"}]}))
    dataset = load_and_validate_dataset(image_root, gt_path)
    records = run_accuracy_pass(dataset, image_root, "mock")
    performance = run_performance_pass(dataset, image_root, "mock", repetitions=2, batch_sizes=[1, 4], concurrency_levels=[1, 2])
    out = tmp_path / "results"
    export_results(records, performance, out)
    assert records[0]["prediction"]["status"] == "SUCCESS"
    assert performance["samples"] == 2
    assert performance["batch_results"][1]["status"] == "SUCCESS"
    assert performance["concurrency_results"][1]["status"] == "SUCCESS"
    assert (out / "report.html").is_file()


def test_worker_failure_is_returned_as_status(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (8, 8), "white").save(image_root / "one.png")
    gt_path = tmp_path / "ground_truth.json"
    gt_path.write_text(json.dumps({"samples": [{"image": "one.png", "fields": {"sku": "ABC"}}]}))
    dataset = load_and_validate_dataset(image_root, gt_path)
    performance = run_performance_pass(dataset, image_root, "unknown_model", repetitions=1)
    assert performance["failure_rate"] == 1.0
    assert performance["status"] == "EXCEPTION"


def test_accuracy_checkpoint_resumes_without_reinference(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (8, 8), "white").save(image_root / "one.png")
    gt_path = tmp_path / "ground_truth.json"
    gt_path.write_text(json.dumps({"samples": [{"image": "one.png", "fields": {"sku": "ABC"}}]}))
    dataset = load_and_validate_dataset(image_root, gt_path)
    checkpoint = tmp_path / "accuracy.checkpoint.jsonl"
    first = run_accuracy_pass(dataset, image_root, "mock", checkpoint_path=checkpoint)
    second = run_accuracy_pass(dataset, image_root, "mock", checkpoint_path=checkpoint)
    assert first == second
    assert len(checkpoint.read_text().splitlines()) == 1
