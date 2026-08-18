"""Provider-neutral parsing for document/OCR model responses.

The three providers added in this module do not share an output schema:
Hunyuan can emit a JSON spotting array or Markdown, Unlimited emits grounded
Markdown markers, and dots.mocr emits a layout JSON object.  This helper keeps
the benchmark contract stable without pretending that their native outputs
are interchangeable.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

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
_UNLIMITED_MARKER_RE = re.compile(r"<\|/?(?:det|ref|grounding)\|>")


def _canonical_key(value: Any) -> str | None:
    normalized = re.sub(r"\s+", " ", str(value).strip().lower().replace("_", " "))
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


def _ordered_text(value: Any) -> Iterable[str]:
    """Yield textual content in document/reading order for common layouts."""
    if isinstance(value, str):
        if value.strip():
            yield value.strip()
        return
    if isinstance(value, list):
        for item in value:
            yield from _ordered_text(item)
        return
    if not isinstance(value, dict):
        return

    # Prefer explicit document text and layout containers.  Coordinate and
    # confidence values are deliberately not rendered as OCR text.
    preferred = ("raw_text", "markdown", "text", "content", "value")
    yielded = False
    for key in preferred:
        item = value.get(key)
        if isinstance(item, str) and item.strip():
            yielded = True
            yield item.strip()
    for key in ("layout", "cells", "elements", "blocks", "regions", "items", "results", "detections", "data"):
        item = value.get(key)
        if isinstance(item, (dict, list)):
            yielded = True
            yield from _ordered_text(item)
    if not yielded:
        for key, item in value.items():
            if key in {"fields", "bbox", "box", "confidence", "score", "category", "name", "key", "field"}:
                continue
            if isinstance(item, (dict, list, str)):
                yield from _ordered_text(item)


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


def parse_document_content(content: Any, *, strip_unlimited_markers: bool = False) -> tuple[str, dict[str, Any]]:
    """Normalize Hunyuan, Unlimited-OCR, or dots.mocr content."""
    parsed, fallback = decode_structured_content(content)
    source: Any = parsed if parsed is not None else fallback
    text_parts = list(_ordered_text(source))
    raw_text = "\n".join(dict.fromkeys(text_parts)).strip()
    if not raw_text:
        raw_text = fallback.strip()
    if strip_unlimited_markers:
        raw_text = _UNLIMITED_MARKER_RE.sub("", raw_text)
        # Official Unlimited-OCR grounding headers carry category/bbox before
        # the actual block content.  Keep the content and remove only markers.
        raw_text = re.sub(r"(?m)^\s*\w+\s*\[[^\]]*\]\s*", "", raw_text)
        raw_text = re.sub(r"\n{3,}", "\n\n", raw_text).strip()

    candidates: dict[str, Any] = {}
    _collect_fields(source, candidates)
    candidates.update(_label_pairs(raw_text))
    candidates.update(_label_pairs(fallback))
    return raw_text, extract_fields(raw_text, candidates, EXPECTED_FIELDS)
