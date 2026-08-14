from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from ocr_benchmark.core.schemas import Prediction


class OCRAdapter(ABC):
    name: str

    @abstractmethod
    def load(self) -> None: ...

    def warmup(self, iterations: int = 1, image_path: Optional[Path] = None) -> None:
        if image_path is None:
            raise RuntimeError("warmup image is required")
        for _ in range(iterations):
            self.predict(image_path)

    def predict_batch(self, image_paths: list[Path]) -> list[Prediction]:
        """Default batch contract; adapters may override with native batching."""
        return [self.predict(image_path) for image_path in image_paths]

    @abstractmethod
    def predict(self, image_path: Path) -> Prediction: ...

    def metadata(self) -> dict[str, Any]:
        return {"name": self.name}

    def unload(self) -> None:
        return None


class MockOCRAdapter(OCRAdapter):
    name = "mock"

    def load(self) -> None:
        return None

    def predict(self, image_path: Path) -> Prediction:
        return Prediction(model=self.name, image=str(image_path), raw_text="")

    def warmup(self, iterations: int = 1, image_path: Optional[Path] = None) -> None:
        # Mock inference does not touch the filesystem and is safe without a fixture.
        return None
