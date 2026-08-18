from ocr_benchmark.models.http_vlm import OpenAICompatibleVLMAdapter
from ocr_benchmark.models.glm_output import parse_glm_content
from ocr_benchmark.models.prompts import GLM_OCR_PROMPT


class GLMOCRAdapter(OpenAICompatibleVLMAdapter):
    name = "glm_ocr"
    model_id = "glm-ocr"
    official_source = "https://github.com/zai-org/GLM-OCR"
    default_prompt = GLM_OCR_PROMPT
    prompt_profile = "glm_text_recognition_json_v1"

    def _parse_content(self, content: object) -> tuple[str, dict[str, object]]:
        return parse_glm_content(content)
