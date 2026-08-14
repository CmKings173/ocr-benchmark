from ocr_benchmark.models.http_vlm import OpenAICompatibleVLMAdapter


class MistralOCRAdapter(OpenAICompatibleVLMAdapter):
    name = "mistral_ocr4"
    model_id = "mistral-ocr"
    official_source = "VERIFY_REQUIRED: private or enterprise deployment"
