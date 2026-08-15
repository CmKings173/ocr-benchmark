from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

from ocr_benchmark.core.adapter import OCRAdapter
from ocr_benchmark.core.schemas import Prediction, Timing
from ocr_benchmark.models.helpers import first_mapping, text_from_payload, to_plain


class PaddleOCRVLAdapter(OCRAdapter):
    name = "paddleocr_vl_1_6"

    def __init__(self, device: str = "gpu:0", engine: Optional[str] = None, revision: Optional[str] = None, license_status: str = "VERIFY_REQUIRED"):
        self.device = device
        self.engine = engine
        self.revision = revision
        self.license_status = license_status
        self.pipeline: Any = None

    def load(self) -> None:
        try:
            import paddle
            from paddleocr import PaddleOCRVL
        except ImportError as exc:
            raise RuntimeError("DEPENDENCY_ERROR: install the official paddleocr package") from exc
        requested_gpu = str(self.device).lower().startswith(("gpu", "cuda"))
        if requested_gpu and (
            not paddle.is_compiled_with_cuda()
            or str(paddle.device.get_device()).lower() == "cpu"
        ):
            raise RuntimeError(
                "GPU_REQUESTED_BUT_PADDLE_CPU_BUILD: this PaddlePaddle installation "
                "has no CUDA support; install a CUDA-enabled aarch64 build or set "
                "device=cpu explicitly"
            )
        kwargs = {"pipeline_version": "v1.6", "device": self.device}
        if self.engine:
            kwargs["engine"] = self.engine
        self.pipeline = PaddleOCRVL(**kwargs)

    def predict(self, image_path: Path) -> Prediction:
        if self.pipeline is None:
            raise RuntimeError("adapter is not loaded")
        preprocess_started = time.perf_counter()
        input_path = str(image_path)
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000
        inference_started = time.perf_counter()
        output = self.pipeline.predict(input=input_path)
        inference_ms = (time.perf_counter() - inference_started) * 1000
        postprocess_started = time.perf_counter()
        results = list(output)
        payload = first_mapping(results)
        raw = text_from_payload(payload)
        if not raw:
            raw = str(to_plain(payload.get("markdown", "")))
        postprocess_ms = (time.perf_counter() - postprocess_started) * 1000
        return Prediction(
            model=self.name,
            image=str(image_path),
            raw_text=raw,
            timing=Timing(preprocess_ms=preprocess_ms, inference_ms=inference_ms, postprocess_ms=postprocess_ms),
            metadata={"pipeline_version": "v1.6", "device": self.device, "engine": self.engine or "default", "revision": self.revision, "license_status": self.license_status},
        )

    def metadata(self) -> dict[str, Any]:
        return {"name": self.name, "pipeline_version": "v1.6", "device": self.device, "revision": self.revision, "license_status": self.license_status, "official_source": "https://github.com/PaddlePaddle/PaddleOCR"}
