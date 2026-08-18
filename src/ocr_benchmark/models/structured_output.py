"""Helpers for parsing structured output from OCR/VLM runtimes.

These helpers deliberately live outside the HTTP adapter.  Native
Transformers adapters and OpenAI-compatible adapters should produce the same
benchmark contract without coupling their runtime implementations.
"""

from __future__ import annotations

import json
from typing import Any


def fields_from_payload(value: Any) -> dict[str, Any]:
    """Convert common VLM field shapes into the benchmark mapping."""
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, list):
        fields: dict[str, Any] = {}
        for item in value:
            if not isinstance(item, dict):
                continue
            name = item.get("name", item.get("key", item.get("field")))
            if name is not None and "value" in item:
                fields[str(name)] = item["value"]
        return fields
    return {}


def decode_structured_content(content: Any) -> tuple[Any, str]:
    """Decode common provider content while preserving the original fallback.

    The decoded value is intentionally exposed so provider-specific adapters
    can handle their own output schemas without duplicating fence/JSON logic.
    """
    if isinstance(content, list):
        # Some OpenAI-compatible servers return content blocks instead of one
        # string.  Preserve text blocks and feed the combined value through
        # the same JSON/fenced-JSON parser.
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                value = item.get("text", item.get("content"))
                if isinstance(value, str):
                    parts.append(value)
        if parts:
            return decode_structured_content("\n".join(parts))
        parsed = None
        fallback = json.dumps(content, ensure_ascii=False)
    elif isinstance(content, dict):
        parsed: Any = content
        fallback = json.dumps(content, ensure_ascii=False)
    elif isinstance(content, str):
        fallback = content
        candidate = content.strip()
        if candidate.startswith("```"):
            lines = candidate.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            candidate = "\n".join(lines).strip()
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            # Some providers prepend a sentence before the JSON object.
            parsed = None
            decoder = json.JSONDecoder()
            for index, char in enumerate(candidate):
                if char != "{":
                    continue
                try:
                    parsed, _ = decoder.raw_decode(candidate[index:])
                    break
                except json.JSONDecodeError:
                    continue
    else:
        fallback = json.dumps(content, ensure_ascii=False)
        parsed = None

    return parsed, fallback


def parse_structured_content(content: Any) -> tuple[str, dict[str, Any]]:
    """Parse plain text or JSON/fenced JSON returned by an OCR/VLM model."""
    parsed, fallback = decode_structured_content(content)

    if isinstance(parsed, dict):
        raw_text = parsed.get("raw_text", parsed.get("text", fallback))
        return str(raw_text), fields_from_payload(parsed.get("fields", {}))
    return fallback, {}
