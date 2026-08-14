from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel


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
                values, types, _ = detector().detectAndDecodeWithType(image)
                for value, symbology in zip(values or [], types or []):
                    if value:
                        return BarcodeResult(engine=self.name, image=str(image_path), status="SUCCESS", value=str(value), symbology=str(symbology), latency_ms=(time.perf_counter() - started) * 1000)
            qr_value, _, _ = cv2.QRCodeDetector().detectAndDecode(image)
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
