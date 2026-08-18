from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any, Optional


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export_results(
    records: list[dict[str, Any]],
    performance: dict[str, Any],
    output_dir: Path,
    environment: Optional[dict[str, Any]] = None,
    accuracy: Optional[dict[str, Any]] = None,
    gates: Optional[dict[str, Any]] = None,
    barcode_records: Optional[list[dict[str, Any]]] = None,
    barcode_summary: Optional[dict[str, Any]] = None,
    system_records: Optional[list[dict[str, Any]]] = None,
    system_accuracy: Optional[dict[str, Any]] = None,
    category_accuracy: Optional[dict[str, Any]] = None,
    score: Optional[float] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "benchmark_raw.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    _write_json(output_dir / "performance.json", performance)
    _write_json(output_dir / "environment.json", environment or {})
    _write_json(output_dir / "barcode_results.json", barcode_records or [])
    _write_json(output_dir / "barcode_summary.json", barcode_summary or {})
    _write_json(output_dir / "system_results.json", system_records or [])
    _write_json(output_dir / "summary.json", {"accuracy": accuracy or {}, "gates": gates or {}, "score": score, "performance": performance, "barcode": barcode_summary or {}, "system_accuracy": system_accuracy or {}, "category_accuracy": category_accuracy or {}})

    detailed_rows = []
    for record in records:
        prediction = record.get("prediction", {})
        fields = record.get("field_metrics", {})
        detailed_rows.append({"image": record.get("image"), "status": prediction.get("status"), "field_exact_match_accuracy": fields.get("field_exact_match_accuracy"), "critical_field_exact_match_accuracy": fields.get("critical_field_exact_match_accuracy"), "full_label_exact_match": fields.get("full_label_exact_match"), "cer": record.get("cer"), "wer": record.get("wer")})
    _write_csv(output_dir / "detailed_results.csv", detailed_rows, ["image", "status", "field_exact_match_accuracy", "critical_field_exact_match_accuracy", "full_label_exact_match", "cer", "wer"])

    field_rows = []
    for record in records:
        metrics = record.get("field_metrics", {})
        for field, correct in metrics.get("per_field", {}).items():
            field_rows.append({"image": record.get("image"), "field": field, "correct": correct, "critical": field in metrics.get("critical_per_field", {})})
    _write_csv(output_dir / "field_accuracy.csv", field_rows, ["image", "field", "correct", "critical"])

    category_rows = [{"category": category, **summary} for category, summary in (category_accuracy or {}).items()]
    _write_csv(output_dir / "category_accuracy.csv", category_rows, ["category", "images", "successful_images", "failure_rate", "field_exact_accuracy", "critical_field_accuracy", "full_label_accuracy", "mean_cer", "mean_wer"])

    resource = performance.get("resource_usage", {})
    _write_csv(output_dir / "resource_usage.csv", [resource], list(resource.keys()) or ["sample_count"])
    _write_csv(output_dir / "concurrency.csv", performance.get("concurrency_results", []), ["concurrency", "status", "images", "failures", "failure_rate", "p50_ms", "p95_ms", "p99_ms", "wall_time_ms", "throughput_images_per_second"])
    _write_csv(output_dir / "batch.csv", performance.get("batch_results", []), ["batch_size", "status", "images", "failures", "failure_rate", "p50_ms", "p95_ms", "p99_ms", "throughput_images_per_second", "execution_mode"])

    rows = "".join(
        f"<tr><td>{html.escape(str(record.get('image')))}</td><td>{html.escape(str(record.get('prediction', {}).get('status')))}</td><td>{html.escape(str(record.get('field_metrics', {}).get('field_exact_match_accuracy', 'N/A')))}</td></tr>"
        for record in records
    )
    report = "<!doctype html><html><head><meta charset='utf-8'><title>OCR Benchmark</title></head><body><h1>OCR Benchmark Report</h1><pre>" + html.escape(json.dumps({"accuracy": accuracy or {}, "gates": gates or {}, "score": score, "performance": performance, "barcode": barcode_summary or {}, "system_accuracy": system_accuracy or {}}, indent=2)) + "</pre><table><tr><th>Image</th><th>Status</th><th>Field exact</th></tr>" + rows + "</table></body></html>"
    (output_dir / "report.html").write_text(report, encoding="utf-8")
