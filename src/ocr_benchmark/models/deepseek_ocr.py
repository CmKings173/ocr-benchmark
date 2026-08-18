from ocr_benchmark.models.http_vlm import OpenAICompatibleVLMAdapter
from ocr_benchmark.models.prompts import DEEPSEEK_OCR_PROMPT


class DeepSeekOCRAdapter(OpenAICompatibleVLMAdapter):
    name = "deepseek_ocr"
    model_id = "deepseek-ai/DeepSeek-OCR"
    official_source = "https://huggingface.co/deepseek-ai/DeepSeek-OCR"
    default_prompt = DEEPSEEK_OCR_PROMPT
    prompt_profile = "deepseek_free_ocr_json_v1"
