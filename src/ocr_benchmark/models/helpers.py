from __future__ import annotations

import json
from typing import Any, Optional


def to_plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [to_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_plain(item) for key, item in value.items()}
    for method_name in ("model_dump", "to_dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                return to_plain(method())
            except TypeError:
                return to_plain(method(mode="json"))
    json_method = getattr(value, "json", None)
    if callable(json_method):
        try:
            return to_plain(json.loads(json_method()))
        except Exception:
            pass
    return {str(key): to_plain(item) for key, item in vars(value).items() if not key.startswith("_")}


def first_mapping(value: Any) -> dict[str, Any]:
    plain = to_plain(value)
    if isinstance(plain, list):
        return first_mapping(plain[0]) if plain else {}
    return plain if isinstance(plain, dict) else {}


def find_first(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    for value in mapping.values():
        if isinstance(value, dict):
            found = find_first(value, keys)
            if found is not None:
                return found
    return None


def text_from_payload(payload: dict[str, Any]) -> str:
    value = find_first(payload, ("rec_texts", "texts", "text", "markdown", "content"))
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value or "")


def normalize_bbox(value: Any) -> Optional[list[float]]:
    """Convert flat or polygon boxes to canonical xmin,ymin,xmax,ymax."""
    if not isinstance(value, (list, tuple)):
        return None
    if len(value) == 4 and all(isinstance(item, (int, float)) for item in value):
        return [float(item) for item in value]
    points = [point for point in value if isinstance(point, (list, tuple)) and len(point) >= 2]
    if not points:
        return None
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return [min(xs), min(ys), max(xs), max(ys)]
