"""GLM-OCR output normalization.

GLM-OCR can return strict JSON, fenced JSON, a ``text`` object, or a nested
object keyed by a document title.  This module turns those model-specific
shapes into the benchmark's canonical field contract without changing field
values.
"""

from __future__ import annotations

import re
from typing import Any

from ocr_benchmark.extraction.fields import extract_fields
from ocr_benchmark.models.structured_output import decode_structured_content


EXPECTED_FIELDS: dict[str, str] = {
    "sku": "",
    "lot": "",
    "quantity": "",
    "unit": "",
    "serial": "",
    "po_number": "",
    "barcode": "",
}

_ALIASES = {
    "sku": "sku",
    "stock keeping unit": "sku",
    "lot": "lot",
    "lot no": "lot",
    "lot number": "lot",
    "qty": "quantity",
    "quantity": "quantity",
    "serial": "serial",
    "serial no": "serial",
    "serial number": "serial",
    "s/n": "serial",
    "sn": "serial",
    "unit": "unit",
    "po": "po_number",
    "po no": "po_number",
    "po number": "po_number",
    "purchase order": "po_number",
    "code": "barcode",
    "barcode": "barcode",
    "barcode value": "barcode",
    "qr": "barcode",
    "qr code": "barcode",
}

_LABEL_RE = re.compile(
    r"^\s*[\"']?(?P<key>[A-Za-z][A-Za-z0-9 _/.-]*)[\"']?\s*[:=]\s*"
    r"[\"']?(?P<value>.*?)[\"']?\s*(?:[,}]\s*)?$"
)


def _canonical_key(key: Any) -> str | None:
    normalized = re.sub(r"\s+", " ", str(key).strip().lower().replace("_", " "))
    return _ALIASES.get(normalized)


def _clean_scalar(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"\"", "'"}:
        value = value[1:-1]
    return value.strip()


def _label_pairs(text: str) -> dict[str, Any]:
    pairs: dict[str, Any] = {}
    for line in text.splitlines():
        match = _LABEL_RE.match(line.strip())
        if not match:
            continue
        canonical = _canonical_key(match.group("key"))
        if canonical is not None:
            pairs[canonical] = _clean_scalar(match.group("value"))
    return pairs


def _collect_fields(value: Any, output: dict[str, Any]) -> None:
    if isinstance(value, dict):
        item_name = value.get("name", value.get("key", value.get("field")))
        if item_name is not None and "value" in value:
            canonical = _canonical_key(item_name)
            if canonical is not None:
                output[canonical] = _clean_scalar(value["value"])
        for key, item in value.items():
            canonical = _canonical_key(key)
            if canonical is not None and not isinstance(item, (dict, list)):
                output[canonical] = _clean_scalar(item)
            if isinstance(item, (dict, list)):
                _collect_fields(item, output)
            elif isinstance(item, str):
                output.update(_label_pairs(item))
        return
    if isinstance(value, list):
        for item in value:
            _collect_fields(item, output)
        return
    if isinstance(value, str):
        output.update(_label_pairs(value))


def parse_glm_content(content: Any) -> tuple[str, dict[str, Any]]:
    """Parse the output shapes observed from GLM-OCR into canonical fields."""
    parsed, fallback = decode_structured_content(content)
    raw_text = fallback
    field_candidates: dict[str, Any] = {}

    if isinstance(parsed, dict):
        explicit_text = parsed.get("raw_text", parsed.get("text"))
        if isinstance(explicit_text, str) and explicit_text.strip():
            raw_text = explicit_text
        _collect_fields(parsed, field_candidates)
        if isinstance(explicit_text, str):
            field_candidates.update(_label_pairs(explicit_text))
    elif isinstance(parsed, str):
        raw_text = parsed
        field_candidates.update(_label_pairs(parsed))
    else:
        field_candidates.update(_label_pairs(fallback))

    # Also handle JSON-like output that is too malformed for json.loads.
    field_candidates.update(_label_pairs(fallback))
    fields = extract_fields(raw_text, field_candidates, EXPECTED_FIELDS)
    return raw_text, fields
