from __future__ import annotations

from typing import Any

from ocr_benchmark.models.document_output import parse_document_content
from ocr_benchmark.models.http_vlm import OpenAICompatibleVLMAdapter


# Copied from HunyuanOCR-1.5's official inference/utils/tasks.py.  The model
# intentionally exposes task types rather than arbitrary user prompts.
TASK_PROMPTS = {
    "doc_parse": "提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略，表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。",
    "structured_parse": "提取图中的文字。",
    "spotting_json": "检测并识别图中所有的文字行，请按从上到下、从左到右的阅读顺序进行识别。输出格式为 JSON 数组，每个元素必须包含：\"box\": [xmin, ymin, xmax, ymax]（坐标需归一化到 [0, 1000] 范围内）；\"text\": \"识别出的文字内容\"。注意：请直接输出 JSON 数组，不要包含任何多余的描述性文字。",
    "spotting_hunyuan": "检测并识别图片中的文字，将文本坐标格式化输出。",
    "layout": "按照阅读顺序解析图中的版式信息。",
    "layout_parse": "提取文档图片中所有内容用markdown格式表示，表格用html格式表达，文档中公式用latex格式表示，请按照阅读顺序组织进行全文解析，并输出版式分析信息。",
    "chart_parse": "解析图中的图表，对于流程图使用Mermaid格式表示，其他图表使用Markdown格式表示。",
    "formula": "识别图片中的公式，用LaTeX格式表示。",
    "table": "把图中的表格解析为HTML。",
    "doc_trans_en2zh": "先解析文档，再将文档内容翻译为中文，其中页眉、页脚忽略，公式用latex格式表示，表格用html格式表示。",
    "trans_other2en": "按照阅读顺序，提取图中文字，公式用latex格式表示，表格用markdown格式表示，再将文字内容翻译为英文。",
    "trans_other2zh": "按照阅读顺序，提取图中文字，公式用latex格式表示，表格用markdown格式表示，再将文字内容翻译为中文。",
}


class HunyuanOCRAdapter(OpenAICompatibleVLMAdapter):
    name = "hunyuan_ocr_1_5"
    model_id = "tencent/HunyuanOCR"
    official_source = "https://github.com/Tencent-Hunyuan/HunyuanOCR"
    prompt_profile = "hunyuan_ocr_official_task_v1"

    def __init__(
        self,
        task_type: str = "spotting_json",
        top_p: float = 1.0,
        top_k: int = -1,
        repetition_penalty: float = 1.08,
        **kwargs: Any,
    ):
        if task_type not in TASK_PROMPTS:
            raise ValueError(f"unknown HunyuanOCR task_type: {task_type}")
        self.task_type = task_type
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        kwargs.setdefault("prompt", TASK_PROMPTS[task_type])
        kwargs.setdefault("max_tokens", 32768)
        super().__init__(**kwargs)

    def _build_request_body(self, *, mime: str, encoded: str) -> dict[str, object]:
        # Match the official OpenAI client: image first, then the locked task
        # prompt, with Hunyuan's documented sampling values.
        return {
            "model": self.model_id,
            "temperature": 0.0,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                    {"type": "text", "text": self._request_prompt()},
                ]},
            ],
            # ``extra_body`` is an OpenAI-Python client convenience; this
            # adapter posts raw JSON, so vLLM extension fields must be at the
            # top level of the HTTP body.
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "skip_special_tokens": True,
        }

    def _parse_content(self, content: object) -> tuple[str, dict[str, object]]:
        return parse_document_content(content)

    def metadata(self) -> dict[str, object]:
        return {
            **super().metadata(),
            "task_type": self.task_type,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
        }
