from __future__ import annotations

from typing import Any, Optional

from ocr_benchmark.core.adapter import MockOCRAdapter, OCRAdapter
from ocr_benchmark.models.glm_ocr import GLMOCRAdapter
from ocr_benchmark.models.monkey_ocr_v2 import MonkeyOCRv2Adapter
from ocr_benchmark.models.monkey_ocr_native import MonkeyOCRv2NativeAdapter
from ocr_benchmark.models.paddleocr_v6 import PPOCRv6Adapter
from ocr_benchmark.models.paddleocr_vl import PaddleOCRVLAdapter
from ocr_benchmark.models.deepseek_ocr import DeepSeekOCRAdapter
from ocr_benchmark.models.donut import DonutAdapter
from ocr_benchmark.models.layoutlmv3 import LayoutLMv3Adapter
from ocr_benchmark.models.mistral_ocr4 import MistralOCRAdapter
from ocr_benchmark.models.qwen2_vl import Qwen2VLAdapter
from ocr_benchmark.models.surya import SuryaOCRAdapter
from ocr_benchmark.models.hunyuan_ocr import HunyuanOCRAdapter
from ocr_benchmark.models.unlimited_ocr import UnlimitedOCRAdapter
from ocr_benchmark.models.dots_mocr import DotsMOCRAdapter


def create_adapter(name: str, config: Optional[dict[str, Any]] = None) -> OCRAdapter:
    config = config or {}
    adapters = {
        "mock": MockOCRAdapter,
        "ppocr_v6": PPOCRv6Adapter,
        "glm_ocr": GLMOCRAdapter,
        "paddleocr_vl_1_6": PaddleOCRVLAdapter,
        "monkey_ocr_v2_b_parsing": MonkeyOCRv2Adapter,
        "monkey_ocr_v2_b_parsing_native": MonkeyOCRv2NativeAdapter,
        "surya_ocr_2": SuryaOCRAdapter,
        "deepseek_ocr": DeepSeekOCRAdapter,
        "qwen2_vl_7b": Qwen2VLAdapter,
        "donut": DonutAdapter,
        "layoutlmv3_kie": LayoutLMv3Adapter,
        "mistral_ocr4": MistralOCRAdapter,
        "hunyuan_ocr_1_5": HunyuanOCRAdapter,
        "unlimited_ocr": UnlimitedOCRAdapter,
        "dots_mocr": DotsMOCRAdapter,
    }
    if name not in adapters:
        raise KeyError(f"unknown model adapter: {name}")
    return adapters[name](**config)
