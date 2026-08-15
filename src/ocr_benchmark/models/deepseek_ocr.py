from ocr_benchmark.models.http_vlm import OpenAICompatibleVLMAdapter


class DeepSeekOCRAdapter(OpenAICompatibleVLMAdapter):
    name = "deepseek_ocr"
    model_id = "deepseek-ai/DeepSeek-OCR"
    official_source = "https://huggingface.co/deepseek-ai/DeepSeek-OCR"
