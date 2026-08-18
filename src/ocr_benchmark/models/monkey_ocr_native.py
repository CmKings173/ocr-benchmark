from __future__ import annotations

import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional, Union

from ocr_benchmark.core.adapter import OCRAdapter
from ocr_benchmark.core.schemas import Prediction, Timing
from ocr_benchmark.models.structured_output import parse_structured_content
from ocr_benchmark.models.prompts import MONKEY_OCR_PROMPT


def _generated_text(value: Any) -> str:
    """Extract assistant text from common Transformers pipeline outputs."""
    if isinstance(value, list):
        if not value:
            return ""
        # A text-generation pipeline returns a list of result objects.
        if len(value) == 1 and isinstance(value[0], dict):
            return _generated_text(value[0])
        # A multimodal chat result may itself be a list of messages.
        for item in reversed(value):
            if isinstance(item, dict) and item.get("role") == "assistant":
                return _generated_text(item.get("content", ""))
        return "\n".join(_generated_text(item) for item in value if item is not None)
    if isinstance(value, dict):
        if "generated_text" in value:
            return _generated_text(value["generated_text"])
        if "text" in value:
            return _generated_text(value["text"])
        if "content" in value:
            return _generated_text(value["content"])
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return str(value)


class MonkeyOCRv2NativeAdapter(OCRAdapter):
    """Run MonkeyOCRv2 through Transformers in the benchmark worker.

    This is the model benchmark path.  The existing
    :class:`MonkeyOCRv2Adapter` remains the HTTP/vLLM system benchmark path so
    deployment and IPC overhead can still be measured separately.
    """

    name = "monkey_ocr_v2_b_parsing_native"
    model_id = "zenosai/MonkeyOCRv2-B-Parsing"
    official_source = "https://huggingface.co/zenosai/MonkeyOCRv2-B-Parsing"

    def __init__(
        self,
        model: Optional[str] = None,
        model_id: Optional[str] = None,
        model_dir: Optional[str] = None,
        device_map: Union[str, dict[str, Any]] = "auto",
        dtype: Optional[str] = "auto",
        trust_remote_code: bool = True,
        prompt: Optional[str] = None,
        max_new_tokens: int = 4096,
        max_pixels: int = 1003520,
        revision: Optional[str] = None,
        license_status: str = "VERIFY_REQUIRED",
    ) -> None:
        self.model_id = model_id or model or self.model_id
        self.model_dir = model_dir
        self.device_map = device_map
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.prompt = prompt if prompt is not None else MONKEY_OCR_PROMPT
        self.max_new_tokens = max_new_tokens
        self.max_pixels = max_pixels
        self.revision = revision
        self.license_status = license_status
        self._pipeline: Any = None

    def load(self) -> None:
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "DEPENDENCY_ERROR: native MonkeyOCR requires transformers, torch, and accelerate"
            ) from exc
        if not self.model_id and not self.model_dir:
            raise RuntimeError("NOT_CONFIGURED: model_id or model_dir is required")

        kwargs: dict[str, Any] = {
            "task": "image-text-to-text",
            "model": self.model_dir or self.model_id,
            "trust_remote_code": self.trust_remote_code,
            "device_map": self.device_map,
        }
        # Let Transformers choose the installed BF16/FP16 dtype by default.
        # Passing dtype="auto" is not supported consistently across versions.
        if self.dtype and self.dtype != "auto":
            kwargs["dtype"] = self.dtype
        try:
            self._pipeline = pipeline(**kwargs)
        except Exception as exc:
            raise RuntimeError(f"MODEL_LOAD_ERROR: native MonkeyOCR failed to load: {exc}") from exc

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "runtime": "transformers-native",
            "model_id": self.model_id,
            "model_dir": self.model_dir,
            "device_map": self.device_map,
            "dtype": self.dtype,
            "max_pixels": self.max_pixels,
            "revision": self.revision,
            "license_status": self.license_status,
            "official_source": self.official_source,
            "prompt_profile": "monkey_document_parsing_json_v1",
            "prompt_sha256": hashlib.sha256(self.prompt.encode("utf-8")).hexdigest(),
        }

    def predict(self, image_path: Path) -> Prediction:
        if self._pipeline is None:
            raise RuntimeError("adapter is not loaded")

        preprocess_started = time.perf_counter()
        if not image_path.is_file():
            raise RuntimeError(f"INVALID_INPUT: image does not exist: {image_path}")
        # Use the local-path form from MonkeyOCRv2's native inference example;
        # this never creates an HTTP request for the image.
        messages = [
            {
                "role": "user",
                "content": [
                    # MonkeyOCRv2's native inference example uses the
                    # ``image`` field for a local path.  ``url`` is reserved
                    # for remote image URLs in the generic Transformers
                    # examples and can accidentally trigger URL handling.
                    {"type": "image", "image": str(image_path.resolve()), "max_pixels": self.max_pixels},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000

        inference_started = time.perf_counter()
        try:
            generated = self._pipeline(text=messages, max_new_tokens=self.max_new_tokens)
        except Exception as exc:
            raise RuntimeError(f"INFERENCE_ERROR: native MonkeyOCR failed: {exc}") from exc
        inference_ms = (time.perf_counter() - inference_started) * 1000

        postprocess_started = time.perf_counter()
        content = _generated_text(generated)
        if not content.strip():
            raise RuntimeError("INVALID_OUTPUT: native MonkeyOCR returned empty text")
        raw_text, fields = parse_structured_content(content)
        postprocess_ms = (time.perf_counter() - postprocess_started) * 1000
        return Prediction(
            model=self.name,
            image=str(image_path),
            raw_text=raw_text,
            fields=fields,
            timing=Timing(
                preprocess_ms=preprocess_ms,
                inference_ms=inference_ms,
                postprocess_ms=postprocess_ms,
            ),
            metadata=self.metadata(),
        )

    def unload(self) -> None:
        self._pipeline = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
