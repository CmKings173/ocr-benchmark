# Model Notes — Tier A

These notes record the official sources checked before adapter implementation. Exact model revisions and installed package versions must be captured in each benchmark run.

## PP-OCRv6

- Official repository: https://github.com/PaddlePaddle/PaddleOCR
- Official OCR pipeline documentation: https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/pipeline_usage/OCR.en.md
- Official docs expose PP-OCRv6 detection and recognition as separate configurable modules, including `PP-OCRv6_medium_det`, `PP-OCRv6_small_det`, `PP-OCRv6_medium_rec` and `PP-OCRv6_small_rec`.
- Adapter uses the official PaddleOCR pipeline with orientation/unwarping disabled by default for the raw benchmark.

## GLM-OCR

- Official repository: https://github.com/zai-org/GLM-OCR
- Official self-host path supports vLLM and SGLang. The adapter uses an OpenAI-compatible self-host endpoint and does not use the cloud/MaaS path.
- If endpoint/model configuration is absent, result status must be `NOT_CONFIGURED`.

## PaddleOCR-VL 1.6

- Official repository: https://github.com/PaddlePaddle/PaddleOCR
- Official usage: https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/pipeline_usage/PaddleOCR-VL.en.md
- The official docs distinguish the complete layout-analysis + VLM pipeline from the standalone VLM component. The adapter targets `PaddleOCRVL(pipeline_version="v1.6")`.

## MonkeyOCRv2

- Official repository: https://github.com/Yuliang-Liu/MonkeyOCRv2
- Official parsing path uses vLLM; DFlash is optional and version-dependent.
- The adapter targets an OpenAI-compatible local vLLM endpoint. DFlash is metadata/configuration, not an assumed capability.

## License and reproducibility

The benchmark must capture repository URL, model identifier, exact revision, package versions, license status and verification date. Vendor benchmark numbers are informational only; customer holdout metrics are authoritative.
