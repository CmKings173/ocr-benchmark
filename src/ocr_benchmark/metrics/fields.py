from typing import Any, Optional


def structural_normalize_fields(fields: dict[str, Any], aliases: Optional[dict[str, str]] = None) -> dict[str, Any]:
    aliases = aliases or {}
    return {aliases.get(str(key), str(key)): value for key, value in fields.items()}


def strict_equal(expected: Any, actual: Any) -> bool:
    # Deliberately no case-folding, punctuation repair, digit/letter repair or value coercion.
    return type(expected) is type(actual) and expected == actual


def field_exact_metrics(
    expected: dict[str, Any],
    actual: dict[str, Any],
    required: list[str],
    critical: Optional[list[str]] = None,
) -> dict[str, Any]:
    per_field = {}
    for field in required:
        per_field[field] = strict_equal(expected.get(field), actual.get(field))
    correct = sum(per_field.values())
    critical_fields = [field for field in (critical or []) if field in required]
    critical_per_field = {field: per_field[field] for field in critical_fields}
    critical_correct = sum(critical_per_field.values())
    return {
        "per_field": per_field,
        "correct_fields": correct,
        "required_fields": len(required),
        "field_exact_match_accuracy": correct / len(required) if required else 0.0,
        "full_label_exact_match": bool(required) and correct == len(required),
        "critical_per_field": critical_per_field,
        "critical_correct_fields": critical_correct,
        "critical_fields": len(critical_fields),
        "critical_field_exact_match_accuracy": critical_correct / len(critical_fields) if critical_fields else None,
        "critical_fields_exact": bool(critical_fields) and critical_correct == len(critical_fields),
    }
