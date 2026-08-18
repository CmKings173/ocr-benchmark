from ocr_benchmark.models.http_vlm import OpenAICompatibleVLMAdapter
from ocr_benchmark.models.prompts import MISTRAL_OCR_PROMPT


class MistralOCRAdapter(OpenAICompatibleVLMAdapter):
    name = "mistral_ocr4"
    model_id = "mistral-ocr"
    official_source = "VERIFY_REQUIRED: private or enterprise deployment"
    default_prompt = MISTRAL_OCR_PROMPT
    prompt_profile = "mistral_document_ocr_json_v1"
