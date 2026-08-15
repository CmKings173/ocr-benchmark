from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel


def _as_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    try:
        return list(value)
    except TypeError:
        return [value]


def _decode_text_lists(payload: Any) -> tuple[list[Any], list[Any]]:
    """Extract decoded values/types across OpenCV 4.x return signatures."""
    parts = list(payload) if isinstance(payload, (list, tuple)) else [payload]
    text_lists: list[list[Any]] = []
    for part in parts:
        items = _as_sequence(part)
        if all(item is None or isinstance(item, (str, bytes)) for item in items):
            text_lists.append(items)
    values = text_lists[0] if text_lists else []
    types = text_lists[1] if len(text_lists) > 1 else []
    return values, types


def _first_text(payload: Any) -> Optional[str]:
    parts = list(payload) if isinstance(payload, (list, tuple)) else [payload]
    for part in parts:
        if isinstance(part, str) and part:
            return part
    return None


class BarcodeResult(BaseModel):
    engine: str
    image: str
    status: str
    value: Optional[str] = None
    symbology: Optional[str] = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None


class BarcodeDecoder(ABC):
    name: str

    @abstractmethod
    def decode(self, image_path: Path) -> BarcodeResult: ...


class OpenCVBarcodeDecoder(BarcodeDecoder):
    name = "opencv"

    def decode(self, image_path: Path) -> BarcodeResult:
        started = time.perf_counter()
        try:
            import cv2
        except ImportError as exc:
            return BarcodeResult(engine=self.name, image=str(image_path), status="DEPENDENCY_ERROR", error=str(exc))
        try:
            image = cv2.imread(str(image_path))
            detector = getattr(getattr(cv2, "barcode", None), "BarcodeDetector", None)
            if detector is not None:
                values, types = _decode_text_lists(detector().detectAndDecodeWithType(image))
                for value, symbology in zip(values or [], types or []):
                    if value:
                        return BarcodeResult(engine=self.name, image=str(image_path), status="SUCCESS", value=str(value), symbology=str(symbology), latency_ms=(time.perf_counter() - started) * 1000)
            qr_value = _first_text(cv2.QRCodeDetector().detectAndDecode(image))
            if qr_value:
                return BarcodeResult(engine=self.name, image=str(image_path), status="SUCCESS", value=str(qr_value), symbology="QR", latency_ms=(time.perf_counter() - started) * 1000)
            return BarcodeResult(engine=self.name, image=str(image_path), status="NO_DECODE", latency_ms=(time.perf_counter() - started) * 1000)
        except Exception as exc:
            return BarcodeResult(engine=self.name, image=str(image_path), status="EXCEPTION", latency_ms=(time.perf_counter() - started) * 1000, error=str(exc))


class ZXingCppDecoder(BarcodeDecoder):
    name = "zxing_cpp"

    def decode(self, image_path: Path) -> BarcodeResult:
        started = time.perf_counter()
        try:
            import cv2
            import zxingcpp
        except ImportError as exc:
            return BarcodeResult(engine=self.name, image=str(image_path), status="DEPENDENCY_ERROR", error=str(exc))
        try:
            image = cv2.imread(str(image_path))
            results = zxingcpp.read_barcodes(image)
            if results:
                result = results[0]
                return BarcodeResult(engine=self.name, image=str(image_path), status="SUCCESS", value=str(result.text), symbology=str(result.format), latency_ms=(time.perf_counter() - started) * 1000)
            return BarcodeResult(engine=self.name, image=str(image_path), status="NO_DECODE", latency_ms=(time.perf_counter() - started) * 1000)
        except Exception as exc:
            return BarcodeResult(engine=self.name, image=str(image_path), status="EXCEPTION", latency_ms=(time.perf_counter() - started) * 1000, error=str(exc))


def decode_exact_match(result: BarcodeResult, expected: Optional[str]) -> Optional[bool]:
    if expected is None:
        return None
    return result.status == "SUCCESS" and result.value == expected
