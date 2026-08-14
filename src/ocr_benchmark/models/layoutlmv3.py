from ocr_benchmark.models.optional import UnavailableAdapter


class LayoutLMv3Adapter(UnavailableAdapter):
    name = "layoutlmv3_kie"

    def __init__(self, reason: str = "NOT_COMPARABLE: a fine-tuned KIE checkpoint is required"):
        super().__init__(reason)
