from __future__ import annotations

import re
from typing import Any


def _coerce(value: str, expected: Any) -> Any:
    if isinstance(expected, bool):
        return value.lower() == "true" if value.lower() in ("true", "false") else value
    if isinstance(expected, int) and not isinstance(expected, bool):
        try:
            return int(value)
        except ValueError:
            return value
    if isinstance(expected, float):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def extract_fields(raw_text: str, model_fields: dict[str, Any], expected_fields: dict[str, Any]) -> dict[str, Any]:
    aliases = {key.lower(): key for key in expected_fields}
    for alias, canonical in {"qty": "quantity", "po": "po_number", "po no": "po_number", "serial_no": "serial", "barcode_value": "barcode"}.items():
        if canonical in expected_fields:
            aliases[alias] = canonical
    values: dict[str, Any] = {}
    for key, value in model_fields.items():
        canonical = aliases.get(str(key).strip().lower(), str(key).strip())
        if canonical in expected_fields:
            values[canonical] = _coerce(str(value), expected_fields[canonical])
    if raw_text:
        for line in raw_text.splitlines():
            match = re.match(r"^\s*([A-Za-z][A-Za-z0-9 _-]*)\s*[:=]\s*(.*?)\s*$", line)
            if not match:
                continue
            canonical = aliases.get(match.group(1).strip().lower())
            if canonical and canonical not in values:
                values[canonical] = _coerce(match.group(2), expected_fields[canonical])
    return values
