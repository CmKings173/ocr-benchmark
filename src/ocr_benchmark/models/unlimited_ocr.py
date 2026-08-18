from __future__ import annotations

from typing import Any

from ocr_benchmark.models.document_output import parse_document_content
from ocr_benchmark.models.http_vlm import OpenAICompatibleVLMAdapter


class UnlimitedOCRAdapter(OpenAICompatibleVLMAdapter):
    name = "unlimited_ocr"
    model_id = "baidu/Unlimited-OCR"
    official_source = "https://github.com/baidu/Unlimited-OCR"
    default_prompt = "<image>document parsing."
    prompt_profile = "unlimited_ocr_official_document_parsing_v1"

    def __init__(
        self,
        image_mode: str = "gundam",
        ngram_size: int = 35,
        ngram_window: int = 128,
        **kwargs: Any,
    ):
        if image_mode not in {"gundam", "base"}:
            raise ValueError("Unlimited-OCR image_mode must be 'gundam' or 'base'")
        self.image_mode = image_mode
        self.ngram_size = ngram_size
        self.ngram_window = ngram_window
        kwargs.setdefault("max_tokens", 32768)
        kwargs.setdefault("prompt", self.default_prompt)
        super().__init__(**kwargs)

    def _build_request_body(self, *, mime: str, encoded: str) -> dict[str, object]:
        # These are the documented vLLM recipe fields.  The logits processor
        # itself is installed when the server is launched; vllm_xargs carries
        # its per-request parameters.
        return {
            "model": self.model_id,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": self._request_prompt()},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
            ]}],
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
            "skip_special_tokens": False,
            "vllm_xargs": {"ngram_size": self.ngram_size, "window_size": self.ngram_window},
            "stream": False,
        }

    def _parse_content(self, content: object) -> tuple[str, dict[str, object]]:
        return parse_document_content(content, strip_unlimited_markers=True)

    def metadata(self) -> dict[str, object]:
        return {
            **super().metadata(),
            "image_mode": self.image_mode,
            "ngram_size": self.ngram_size,
            "ngram_window": self.ngram_window,
        }
