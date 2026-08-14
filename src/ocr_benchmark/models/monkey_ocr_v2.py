from ocr_benchmark.models.http_vlm import OpenAICompatibleVLMAdapter


class MonkeyOCRv2Adapter(OpenAICompatibleVLMAdapter):
    name = "monkey_ocr_v2_b_parsing"
    model_id = "MonkeyOCRv2-B-Parsing"
    official_source = "https://github.com/Yuliang-Liu/MonkeyOCRv2"
