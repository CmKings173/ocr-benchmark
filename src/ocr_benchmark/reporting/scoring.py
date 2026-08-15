from __future__ import annotations

import math
from typing import Any, Optional


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    if trials <= 0:
        return (0.0, 0.0)
    z = 1.959963984540054 if confidence >= 0.95 else 1.6448536269514722
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * trials)) / trials) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


def gate_result(summary: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, Optional[bool]] = {}
    critical_value = summary.get("critical_field_accuracy")
    for metric, measured, minimum in (("full_label_accuracy", summary.get("full_label_accuracy"), gates.get("full_label_accuracy_min")), ("critical_field_accuracy", critical_value, gates.get("critical_field_accuracy_min"))):
        checks[metric] = None if minimum is None or measured is None else measured >= minimum
    for metric, maximum in (("failure_rate", gates.get("failure_rate_max")), ("p95_ms", gates.get("p95_latency_max_ms")), ("p99_ms", gates.get("p99_latency_max_ms")), ("peak_unified_memory_bytes", gates.get("peak_memory_max_bytes"))):
        checks[metric] = None if maximum is None or summary.get(metric) is None else summary[metric] <= maximum
    applicable = [value for value in checks.values() if value is not None]
    has_successful_predictions = summary.get("successful_images", 0) > 0
    gate_eligible = all(applicable) if applicable else True
    return {
        "eligible": bool(gate_eligible and has_successful_predictions),
        "valid": has_successful_predictions,
        "checks": checks,
    }


def aggregate_accuracy(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [record for record in records if record.get("prediction", {}).get("status") == "SUCCESS"]
    field_scores = [record["field_metrics"]["field_exact_match_accuracy"] for record in valid if "field_metrics" in record]
    labels = [record["field_metrics"]["full_label_exact_match"] for record in valid if "field_metrics" in record]
    critical_scores = [record["field_metrics"]["critical_field_exact_match_accuracy"] for record in valid if record.get("field_metrics", {}).get("critical_field_exact_match_accuracy") is not None]
    critical_labels = [record["field_metrics"]["critical_fields_exact"] for record in valid if record.get("field_metrics", {}).get("critical_fields")]
    full_label_successes = sum(labels)
    critical_successes = sum(critical_labels)
    return {
        "images": len(records),
        "successful_images": len(valid),
        "failure_rate": (len(records) - len(valid)) / max(len(records), 1),
        "field_exact_accuracy": sum(field_scores) / len(field_scores) if field_scores else None,
        "critical_field_accuracy": sum(critical_scores) / len(critical_scores) if critical_scores else None,
        "critical_full_label_accuracy": sum(critical_labels) / len(critical_labels) if critical_labels else None,
        "full_label_accuracy": sum(labels) / len(labels) if labels else None,
        "full_label_accuracy_wilson_95": wilson_interval(full_label_successes, len(labels)) if labels else None,
        "critical_full_label_accuracy_wilson_95": wilson_interval(critical_successes, len(critical_labels)) if critical_labels else None,
        "mean_cer": sum(record.get("cer", 0.0) for record in valid) / len(valid) if valid else None,
        "mean_wer": sum(record.get("wer", 0.0) for record in valid) / len(valid) if valid else None,
    }


def aggregate_by_tag(records: list[dict[str, Any]], dataset: Any) -> dict[str, Any]:
    samples = {sample.image: sample for sample in dataset.samples}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        sample = samples.get(record.get("image"))
        for tag in (getattr(sample, "tags", []) if sample is not None else []):
            grouped.setdefault(tag, []).append(record)
    return {tag: aggregate_accuracy(items) for tag, items in grouped.items()}


def composite_score(summary: dict[str, Any], weights: Optional[dict[str, float]] = None) -> Optional[float]:
    if summary.get("eligible") is False:
        return None
    weights = weights or {"full_label_accuracy": 0.5, "critical_field_accuracy": 0.3, "latency_score": 0.1, "reliability_score": 0.1}
    values = {
        "full_label_accuracy": summary.get("full_label_accuracy"),
        "critical_field_accuracy": summary.get("critical_field_accuracy"),
        "latency_score": 1.0 / (1.0 + max(float(summary.get("p95_ms", 0.0) or 0.0), 0.0) / 1000.0),
        "reliability_score": 1.0 - min(max(float(summary.get("failure_rate", 1.0) or 1.0), 0.0), 1.0),
    }
    available = [(weight, values[name]) for name, weight in weights.items() if values.get(name) is not None]
    denominator = sum(weight for weight, _ in available)
    return sum(weight * float(value) for weight, value in available) / denominator if denominator else None
