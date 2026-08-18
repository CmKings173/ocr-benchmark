import json

from PIL import Image

from ocr_benchmark.barcode.decoder import BarcodeDecoder, BarcodeResult
from ocr_benchmark.barcode.runner import aggregate_barcode, build_system_records, run_barcode_pass
from ocr_benchmark.data.validator import load_and_validate_dataset


class FakeDecoder(BarcodeDecoder):
    name = "fake"

    def decode(self, image_path):
        return BarcodeResult(engine=self.name, image=str(image_path), status="SUCCESS", value="123")


def test_barcode_pass_and_exact_aggregation(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (8, 8), "white").save(image_root / "one.png")
    gt_path = tmp_path / "ground_truth.json"
    gt_path.write_text(json.dumps({"samples": [{"image": "one.png", "fields": {"barcode": "123"}}]}))
    dataset = load_and_validate_dataset(image_root, gt_path)
    records = run_barcode_pass(dataset, image_root, [FakeDecoder()])
    summary = aggregate_barcode(records)
    assert summary["engines"]["fake"]["exact_match_accuracy"] == 1.0


def test_system_pipeline_prefers_zxing_without_ground_truth_lookup():
    dataset = type("Dataset", (), {"samples": [type("Sample", (), {"image": "one.png", "fields": {"barcode": "123"}, "critical_fields": [], "label_type": "default"})()]})()
    model_records = [{"image": "one.png", "prediction": {"status": "SUCCESS", "fields": {"sku": "ABC"}}}]
    barcode_records = [{
        "image": "one.png",
        "engines": [
            {"result": {"engine": "opencv", "status": "SUCCESS", "value": "wrong"}},
            {"result": {"engine": "zxing_cpp", "status": "SUCCESS", "value": "123"}},
        ],
    }]

    records = build_system_records(model_records, barcode_records, dataset)

    assert records[0]["barcode_engine"] == "zxing_cpp"
    assert records[0]["prediction"]["fields"]["barcode"] == "123"
