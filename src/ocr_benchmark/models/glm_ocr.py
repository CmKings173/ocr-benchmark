from ocr_benchmark.models.http_vlm import OpenAICompatibleVLMAdapter


class GLMOCRAdapter(OpenAICompatibleVLMAdapter):
    name = "glm_ocr"
    model_id = "glm-ocr"
    official_source = "https://github.com/zai-org/GLM-OCR"
