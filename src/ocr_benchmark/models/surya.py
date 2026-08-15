from ocr_benchmark.models.http_vlm import OpenAICompatibleVLMAdapter


class SuryaOCRAdapter(OpenAICompatibleVLMAdapter):
    name = "surya_ocr_2"
    model_id = "datalab-to/surya-ocr-2"
    official_source = "https://huggingface.co/datalab-to/surya-ocr-2"
