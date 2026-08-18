from ocr_benchmark.models.http_vlm import OpenAICompatibleVLMAdapter
from ocr_benchmark.models.prompts import QWEN_OCR_PROMPT


class Qwen2VLAdapter(OpenAICompatibleVLMAdapter):
    name = "qwen2_vl_7b"
    model_id = "Qwen/Qwen2-VL-7B-Instruct"
    official_source = "https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct"
    default_prompt = QWEN_OCR_PROMPT
    prompt_profile = "qwen_label_ocr_json_v1"
