from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

from ocr_benchmark.barcode.decoder import BarcodeDecoder, BarcodeResult, OpenCVBarcodeDecoder, ZXingCppDecoder, decode_exact_match
from ocr_benchmark.core.schemas import Dataset


def _expected_barcode(sample: Any) -> Any:
    return sample.fields.get("barcode") if isinstance(sample.fields, dict) else None


def run_barcode_pass(dataset: Dataset, dataset_root: Path, decoders: Optional[Iterable[BarcodeDecoder]] = None) -> list[dict[str, Any]]:
    engines = list(decoders or (OpenCVBarcodeDecoder(), ZXingCppDecoder()))
    records: list[dict[str, Any]] = []
    for sample in dataset.samples:
        image_path = (dataset_root / sample.image).resolve()
        expected = _expected_barcode(sample)
        results: list[dict[str, Any]] = []
        for decoder in engines:
            result: BarcodeResult = decoder.decode(image_path)
            results.append({"result": result.model_dump(mode="json"), "exact_match": decode_exact_match(result, str(expected) if expected is not None else None)})
        records.append({"image": sample.image, "expected": expected, "barcode_type": sample.barcode_type, "engines": results})
    return records


def aggregate_barcode(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_engine: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        for item in record.get("engines", []):
            result = item.get("result", {})
            by_engine.setdefault(result.get("engine", "unknown"), []).append(item)
    summary: dict[str, Any] = {"images": len(records), "engines": {}}
    for engine, items in by_engine.items():
        comparable = [item for item in items if item.get("exact_match") is not None]
        exact = [item for item in comparable if item.get("exact_match") is True]
        summary["engines"][engine] = {
            "images": len(items),
            "successes": sum(item.get("result", {}).get("status") == "SUCCESS" for item in items),
            "exact_matches": len(exact),
            "exact_match_accuracy": len(exact) / len(comparable) if comparable else None,
            "mean_latency_ms": sum(item.get("result", {}).get("latency_ms", 0) or 0 for item in items) / len(items) if items else None,
        }
    return summary


def build_system_records(
    model_records: list[dict[str, Any]],
    barcode_records: list[dict[str, Any]],
    dataset: Dataset,
    field_schema: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    barcode_by_image = {record["image"]: record for record in barcode_records}
    sample_by_image = {sample.image: sample for sample in dataset.samples}
    output: list[dict[str, Any]] = []
    for record in model_records:
        image = record["image"]
        prediction = dict(record.get("prediction", {}))
        fields = dict(prediction.get("fields", {}))
        barcode = barcode_by_image.get(image, {})
        decoded = None
        selected_engine = None
        # Decoder choice is a deployment policy, never a ground-truth lookup.
        # ZXing is the preferred production decoder on this dataset; OpenCV is
        # retained as a deterministic fallback.  The full engine-level results
        # remain available in barcode_results.json for comparison.
        engines = barcode.get("engines", [])
        preferred = {"zxing_cpp": 0, "opencv": 1}
        ordered = sorted(enumerate(engines), key=lambda pair: (preferred.get(pair[1].get("result", {}).get("engine"), 99), pair[0]))
        for _, item in ordered:
            result = item.get("result", {})
            if result.get("status") == "SUCCESS" and result.get("value"):
                decoded = result["value"]
                selected_engine = result.get("engine")
                break
        if decoded is not None:
            fields["barcode"] = decoded
        prediction["fields"] = fields
        sample = sample_by_image.get(image)
        system = {
            "image": image,
            "prediction": prediction,
            "barcode_source": "decoder" if decoded is not None else "ocr",
            "barcode_engine": selected_engine,
        }
        if sample is not None:
            required = list(sample.fields.keys())
            from ocr_benchmark.metrics.fields import field_exact_metrics

            critical = list(getattr(sample, "critical_fields", []))
            if not critical:
                label_schema = (field_schema or {}).get("label_types", {}).get(sample.label_type, {})
                critical = [field for field, spec in label_schema.get("fields", {}).items() if isinstance(spec, dict) and spec.get("critical") and field in sample.fields]
            system["field_metrics"] = field_exact_metrics(sample.fields, fields, required, critical)
        output.append(system)
    return output
