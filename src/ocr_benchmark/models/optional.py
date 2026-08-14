from __future__ import annotations

from pathlib import Path
from typing import Any

from ocr_benchmark.core.adapter import OCRAdapter
from ocr_benchmark.core.schemas import Prediction


class UnavailableAdapter(OCRAdapter):
    reason = "NOT_CONFIGURED"

    def __init__(self, reason: str = "NOT_CONFIGURED: adapter requires an explicitly configured official runtime"):
        self.reason = reason

    def load(self) -> None:
        raise RuntimeError(self.reason)

    def predict(self, image_path: Path) -> Prediction:
        raise RuntimeError(self.reason)

    def metadata(self) -> dict[str, Any]:
        return {"name": self.name, "status": self.reason}
