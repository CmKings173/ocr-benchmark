from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

from ocr_benchmark.core.adapter import OCRAdapter
from ocr_benchmark.core.schemas import Detection, Prediction, RunStatus, Timing
from ocr_benchmark.models.helpers import first_mapping, find_first, normalize_bbox, text_from_payload


class PPOCRv6Adapter(OCRAdapter):
    name = "ppocr_v6"

    def __init__(self, det_model: str = "PP-OCRv6_medium_det", rec_model: str = "PP-OCRv6_medium_rec", device: str = "gpu:0", det_dir: Optional[str] = None, rec_dir: Optional[str] = None, revision: Optional[str] = None, license_status: str = "VERIFY_REQUIRED"):
        self.det_model = det_model
        self.rec_model = rec_model
        self.device = device
        self.det_dir = det_dir
        self.rec_dir = rec_dir
        self.revision = revision
        self.license_status = license_status
        self.pipeline: Any = None

    def load(self) -> None:
        try:
            import paddle
            from paddleocr import PaddleOCR
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
        kwargs = {
            "ocr_version": "PP-OCRv6",
            "text_detection_model_name": self.det_model,
            "text_recognition_model_name": self.rec_model,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "device": self.device,
        }
        if self.det_dir:
            kwargs["text_detection_model_dir"] = self.det_dir
        if self.rec_dir:
            kwargs["text_recognition_model_dir"] = self.rec_dir
        self.pipeline = PaddleOCR(**kwargs)

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
        payload = first_mapping(list(output))
        texts = find_first(payload, ("rec_texts", "texts"))
        scores = find_first(payload, ("rec_scores", "scores"))
        boxes = find_first(payload, ("rec_boxes", "dt_polys", "boxes"))
        detections = []
        for index, text in enumerate(texts or []):
            score = scores[index] if isinstance(scores, list) and index < len(scores) else None
            box = boxes[index] if isinstance(boxes, list) and index < len(boxes) else None
            detections.append(Detection(text=str(text), confidence=float(score) if score is not None else None, bbox=normalize_bbox(box)))
        postprocess_ms = (time.perf_counter() - postprocess_started) * 1000
        return Prediction(
            model=self.name,
            image=str(image_path),
            raw_text=text_from_payload(payload),
            detections=detections,
            timing=Timing(preprocess_ms=preprocess_ms, inference_ms=inference_ms, postprocess_ms=postprocess_ms),
            metadata={"det_model": self.det_model, "rec_model": self.rec_model, "device": self.device, "revision": self.revision, "license_status": self.license_status},
        )

    def metadata(self) -> dict[str, Any]:
        return {"name": self.name, "det_model": self.det_model, "rec_model": self.rec_model, "device": self.device, "revision": self.revision, "license_status": self.license_status, "official_source": "https://github.com/PaddlePaddle/PaddleOCR"}
