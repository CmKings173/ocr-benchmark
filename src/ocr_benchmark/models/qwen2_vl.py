from ocr_benchmark.models.http_vlm import OpenAICompatibleVLMAdapter


class Qwen2VLAdapter(OpenAICompatibleVLMAdapter):
    name = "qwen2_vl_7b"
    model_id = "Qwen/Qwen2-VL-7B-Instruct"
    official_source = "https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct"
