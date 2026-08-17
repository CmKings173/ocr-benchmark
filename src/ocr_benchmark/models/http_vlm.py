from __future__ import annotations

import base64
import json
import mimetypes
import time
import urllib.request
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from ocr_benchmark.core.adapter import OCRAdapter
from ocr_benchmark.core.schemas import Prediction, Timing
from ocr_benchmark.models.structured_output import parse_structured_content

# Backwards-compatible private name used by existing tests and callers.
_parse_structured_content = parse_structured_content


class OpenAICompatibleVLMAdapter(OCRAdapter):
    model_id = ""
    official_source = ""

    def __init__(self, endpoint: Optional[str] = None, model: Optional[str] = None, api_key: Optional[str] = None, prompt: str = "Extract the document text. Return JSON with raw_text and fields.", revision: Optional[str] = None, license_status: str = "VERIFY_REQUIRED", timeout_seconds: float = 60.0, max_tokens: int = 2048, allow_remote_endpoint: bool = False, verify_model: bool = True):
        self.endpoint = (endpoint or "").rstrip("/")
        self.model_id = model or self.model_id
        self.api_key = api_key
        self.prompt = prompt
        self.revision = revision
        self.license_status = license_status
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.allow_remote_endpoint = allow_remote_endpoint
        self.verify_model = verify_model
        self.server_model_id: Optional[str] = None

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
        if self.verify_model:
            self.server_model_id = self._verify_server_model()

    def _verify_server_model(self) -> str:
        """Ensure the local endpoint is serving the model being benchmarked."""
        request = urllib.request.Request(self.endpoint + "/models", headers={"Accept": "application/json"})
        if self.api_key:
            request.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw_response = response.read(1024 * 1024 + 1)
        except Exception as exc:
            raise RuntimeError(f"SERVER_CHECK_ERROR: cannot query {self.endpoint}/models: {exc}") from exc
        if len(raw_response) > 1024 * 1024:
            raise RuntimeError("SERVER_CHECK_ERROR: /models response exceeds 1 MiB")
        try:
            payload = json.loads(raw_response.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("SERVER_CHECK_ERROR: /models response is not valid JSON") from exc

        advertised: list[str] = []
        for collection_key in ("data", "models"):
            collection = payload.get(collection_key, []) if isinstance(payload, dict) else []
            if isinstance(collection, dict):
                collection = [collection]
            if not isinstance(collection, list):
                continue
            for item in collection:
                if isinstance(item, str):
                    advertised.append(item)
                elif isinstance(item, dict):
                    for key in ("id", "name", "model"):
                        value = item.get(key)
                        if value:
                            advertised.append(str(value))
        if not advertised:
            raise RuntimeError(f"MODEL_MISMATCH: {self.endpoint}/models did not advertise any model")

        requested = {self.model_id, self.model_id.rsplit("/", 1)[-1]}
        for candidate in advertised:
            if candidate in requested or candidate.rsplit("/", 1)[-1] in requested:
                return candidate
        raise RuntimeError(
            f"MODEL_MISMATCH: requested {self.model_id!r}; endpoint advertises {sorted(set(advertised))!r}"
        )

    def predict(self, image_path: Path) -> Prediction:
        preprocess_started = time.perf_counter()
        mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
        try:
            image_bytes = image_path.read_bytes()
        except OSError as exc:
            raise RuntimeError(f"INVALID_INPUT: cannot read image {image_path}: {exc}") from exc
        encoded = base64.b64encode(image_bytes).decode("ascii")
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
        if content is None or content == [] or (isinstance(content, str) and not content.strip()):
            raise RuntimeError("INVALID_OUTPUT: response returned empty message content")
        raw_text, fields = _parse_structured_content(content)
        postprocess_ms = (time.perf_counter() - postprocess_started) * 1000
        return Prediction(model=self.name, image=str(image_path), raw_text=raw_text, fields=fields, timing=Timing(preprocess_ms=preprocess_ms, inference_ms=inference_ms, postprocess_ms=postprocess_ms), metadata={"endpoint": self.endpoint, "model_id": self.model_id, "server_model_id": self.server_model_id, "revision": self.revision, "license_status": self.license_status})
