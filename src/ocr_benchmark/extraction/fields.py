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


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def extract_fields(raw_text: str, model_fields: dict[str, Any], expected_fields: dict[str, Any]) -> dict[str, Any]:
    aliases = {key.lower(): key for key in expected_fields}
    for alias, canonical in {
        "qty": "quantity",
        "po": "po_number",
        "po no": "po_number",
        "serial_no": "serial",
        "serial no": "serial",
        "s/n": "serial",
        "sn": "serial",
        "barcode_value": "barcode",
        "barcode no": "barcode",
        "code": "barcode",
        "qr": "barcode",
        "qr code": "barcode",
    }.items():
        if canonical in expected_fields:
            aliases[alias] = canonical

    def set_value(canonical: str, value: str) -> None:
        if canonical in expected_fields and (canonical not in values or _is_blank(values[canonical])):
            values[canonical] = _coerce(value.strip(), expected_fields[canonical])

    values: dict[str, Any] = {}
    for key, value in model_fields.items():
        if _is_blank(value):
            continue
        canonical = aliases.get(str(key).strip().lower(), str(key).strip())
        if canonical in expected_fields:
            value_text = str(value).strip()
            if canonical == "quantity" and "unit" in expected_fields:
                quantity_match = re.match(r"^\s*([0-9]+(?:[.,][0-9]+)?)\s+(.+?)\s*$", value_text)
                if quantity_match:
                    values[canonical] = _coerce(quantity_match.group(1).replace(",", "."), expected_fields[canonical])
                    if _is_blank(values.get("unit")):
                        values["unit"] = _coerce(quantity_match.group(2), expected_fields["unit"])
                    continue
            values[canonical] = _coerce(value_text, expected_fields[canonical])
    if raw_text:
        lines = [line.strip() for line in raw_text.splitlines()]
        label_pattern = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 _/.-]*)\s*[:=]\s*(.*?)\s*$")
        for index, line in enumerate(lines):
            match = label_pattern.match(line)
            if not match:
                continue
            canonical = aliases.get(re.sub(r"\s+", " ", match.group(1).strip().lower()))
            if not canonical or canonical not in expected_fields:
                continue
            value = match.group(2).strip()
            if not value:
                # OCR commonly emits `SKU:` and the value on the next line.
                for next_line in lines[index + 1 :]:
                    if not next_line:
                        continue
                    if label_pattern.match(next_line):
                        break
                    value = next_line
                    break
            if not value:
                continue
            if canonical == "quantity" and "unit" in expected_fields:
                quantity_match = re.match(r"^\s*([0-9]+(?:[.,][0-9]+)?)\s+(.+?)\s*$", value)
                if quantity_match:
                    set_value("quantity", quantity_match.group(1).replace(",", "."))
                    set_value("unit", quantity_match.group(2))
                    continue
            set_value(canonical, value)
    return values
