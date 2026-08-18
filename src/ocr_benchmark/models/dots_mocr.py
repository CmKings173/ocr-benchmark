from __future__ import annotations

from typing import Any

from ocr_benchmark.models.document_output import parse_document_content
from ocr_benchmark.models.http_vlm import OpenAICompatibleVLMAdapter


DOTS_MOCR_JSON_PROMPT = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]
2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].
3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object."""
DOTS_IMAGE_PREFIX = "<|img|><|imgpad|><|endofimg|>"
DOTS_PROMPTS = {
    "prompt_layout_all_en": DOTS_MOCR_JSON_PROMPT,
    "prompt_layout_only_en": "Please output the layout information from this PDF image, including each layout's bbox and its category. The bbox should be in the format [x1, y1, x2, y2]. The layout categories for the PDF document include ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']. Do not output the corresponding text. The layout result should be in JSON format.",
    "prompt_ocr": "Extract the text content from this image.",
    "prompt_grounding_ocr": "Extract text from the given bounding box on the image (format: [x1, y1, x2, y2]).\nBounding Box:\n",
    "prompt_web_parsing": "Parsing the layout info of this webpage image with format json:\n",
    "prompt_scene_spotting": "Detect and recognize the text in the image.",
    "prompt_image_to_svg": "Please generate the SVG code based on the image.",
    "prompt_general": " ",
}


class DotsMOCRAdapter(OpenAICompatibleVLMAdapter):
    name = "dots_mocr"
    model_id = "rednote-hilab/dots.mocr"
    official_source = "https://github.com/studio-dots-ai/dots.mocr"
    default_prompt = DOTS_MOCR_JSON_PROMPT
    prompt_profile = "dots_mocr_official_layout_json_v1"

    def __init__(self, prompt_mode: str = "prompt_layout_all_en", **kwargs: Any):
        if prompt_mode not in DOTS_PROMPTS:
            raise ValueError(f"unknown dots.mocr prompt_mode: {prompt_mode}")
        self.prompt_mode = prompt_mode
        self.prompt_profile = f"dots_mocr_{prompt_mode}_v1"
        # The official vLLM client uses 32768 completion tokens.  Keep the
        # provider default aligned with that client; callers may still lower
        # it explicitly for a bounded benchmark or production workload.
        kwargs.setdefault("max_tokens", 32768)
        kwargs.setdefault("prompt", DOTS_PROMPTS[prompt_mode])
        super().__init__(**kwargs)

    def _build_request_body(self, *, mime: str, encoded: str) -> dict[str, object]:
        # The official dots.mocr vLLM client prepends this image sentinel to
        # the text block; omitting it changes the model's chat-template path.
        return {
            "model": self.model_id,
            "temperature": 0.1,
            "top_p": 0.9,
            "max_completion_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                {"type": "text", "text": self._request_prompt()},
            ]}],
        }

    def _request_prompt(self) -> str:
        return f"{DOTS_IMAGE_PREFIX}{self.prompt}"

    def _parse_content(self, content: object) -> tuple[str, dict[str, object]]:
        return parse_document_content(content)

    def metadata(self) -> dict[str, object]:
        return {**super().metadata(), "prompt_mode": self.prompt_mode}
