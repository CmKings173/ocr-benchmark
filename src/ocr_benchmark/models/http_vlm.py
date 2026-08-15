from __future__ import annotations

import base64
import json
import mimetypes
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from ocr_benchmark.core.adapter import OCRAdapter
from ocr_benchmark.core.schemas import Prediction, Timing


class OpenAICompatibleVLMAdapter(OCRAdapter):
    model_id = ""
    official_source = ""

    def __init__(self, endpoint: Optional[str] = None, model: Optional[str] = None, api_key: Optional[str] = None, prompt: str = "Extract the document text. Return JSON with raw_text and fields.", revision: Optional[str] = None, license_status: str = "VERIFY_REQUIRED", timeout_seconds: float = 60.0, max_tokens: int = 2048, allow_remote_endpoint: bool = False):
        self.endpoint = (endpoint or "").rstrip("/")
        self.model_id = model or self.model_id
        self.api_key = api_key
        self.prompt = prompt
        self.revision = revision
        self.license_status = license_status
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.allow_remote_endpoint = allow_remote_endpoint

    def load(self) -> None:
        if not self.endpoint:
            raise RuntimeError("NOT_CONFIGURED: an OpenAI-compatible VLM endpoint is required")
        if not self.model_id:
            raise RuntimeError("NOT_CONFIGURED: model id is required")
        parsed = urlparse(self.endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise RuntimeError("NOT_CONFIGURED: endpoint must be an HTTP(S) URL")
        if not self.allow_remote_endpoint and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise RuntimeError("NOT_CONFIGURED: remote endpoint requires allow_remote_endpoint=true")

    def predict(self, image_path: Path) -> Prediction:
        preprocess_started = time.perf_counter()
        mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        body = {
            "model": self.model_id,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": [{"type": "text", "text": self.prompt}, {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}]}],
        }
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000
        request = urllib.request.Request(self.endpoint + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        if self.api_key:
            request.add_header("Authorization", f"Bearer {self.api_key}")
        inference_started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            raw_response = response.read(10 * 1024 * 1024 + 1)
        if len(raw_response) > 10 * 1024 * 1024:
            raise RuntimeError("INVALID_OUTPUT: VLM response exceeds 10 MiB")
        try:
            payload = json.loads(raw_response.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("INVALID_OUTPUT: VLM response is not valid JSON") from exc
        inference_ms = (time.perf_counter() - inference_started) * 1000
        postprocess_started = time.perf_counter()
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("INVALID_OUTPUT: response missing choices[0].message.content") from exc
        fields: dict[str, Any] = {}
        raw_text = content if isinstance(content, str) else json.dumps(content)
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    fields = dict(parsed.get("fields", {}))
                    raw_text = str(parsed.get("raw_text", content))
            except json.JSONDecodeError:
                pass
        postprocess_ms = (time.perf_counter() - postprocess_started) * 1000
        return Prediction(model=self.name, image=str(image_path), raw_text=raw_text, fields=fields, timing=Timing(preprocess_ms=preprocess_ms, inference_ms=inference_ms, postprocess_ms=postprocess_ms), metadata={"endpoint": self.endpoint, "model_id": self.model_id, "revision": self.revision, "license_status": self.license_status})
