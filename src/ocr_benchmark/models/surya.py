from ocr_benchmark.models.http_vlm import OpenAICompatibleVLMAdapter
from ocr_benchmark.models.prompts import SURYA_OCR_PROMPT


class SuryaOCRAdapter(OpenAICompatibleVLMAdapter):
    name = "surya_ocr_2"
    model_id = "datalab-to/surya-ocr-2"
    official_source = "https://huggingface.co/datalab-to/surya-ocr-2"
    default_prompt = SURYA_OCR_PROMPT
    prompt_profile = "surya_label_ocr_json_v1"
