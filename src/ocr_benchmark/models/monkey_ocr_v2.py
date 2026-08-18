from ocr_benchmark.models.http_vlm import OpenAICompatibleVLMAdapter
from ocr_benchmark.models.prompts import MONKEY_OCR_PROMPT


class MonkeyOCRv2Adapter(OpenAICompatibleVLMAdapter):
    name = "monkey_ocr_v2_b_parsing"
    model_id = "MonkeyOCRv2-B-Parsing"
    official_source = "https://github.com/Yuliang-Liu/MonkeyOCRv2"
    default_prompt = MONKEY_OCR_PROMPT
    prompt_profile = "monkey_document_parsing_json_v1"
