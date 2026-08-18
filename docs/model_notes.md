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
- `monkey_ocr_v2_b_parsing` targets an OpenAI-compatible local vLLM endpoint. DFlash is metadata/configuration, not an assumed capability.
- `monkey_ocr_v2_b_parsing_native` uses the Transformers `image-text-to-text` pipeline inside the worker for model-only measurements. Keep both scopes: native for model latency, HTTP for production system/IPC latency.
- Native Transformers usage is documented by the [MonkeyOCRv2-B-Parsing model card](https://huggingface.co/zenosai/MonkeyOCRv2-B-Parsing); run a one-image preflight before a full benchmark because custom remote code and runtime versions must match the GX10 environment.

## HunyuanOCR-1.5

- Official repository and inference guide: https://github.com/Tencent-Hunyuan/HunyuanOCR
- Official vLLM server is OpenAI-compatible and uses the model id `tencent/HunyuanOCR`.
- The adapter locks prompts to the official task vocabulary. The default benchmark task is `spotting_json` (JSON array of normalized boxes and text); `doc_parse` remains available for Markdown parsing.
- License: Tencent Hunyuan Community License Agreement. CUDA 13 is required by the unified 1.5 environment; the official guide documents lighter per-configuration recipes for other CUDA versions.

## Unlimited-OCR

- Official repository: https://github.com/baidu/Unlimited-OCR
- Official vLLM recipe: https://recipes.vllm.ai/baidu/Unlimited-OCR
- The model requires the dedicated `vllm/vllm-openai:unlimited-ocr` image, `NGramPerReqLogitsProcessor`, the literal `<image>` prompt prefix, `skip_special_tokens=false`, and per-request n-gram arguments. The adapter sends these fields and parses the native Markdown/grounding output.
- Native output is not a strict JSON contract. License: MIT.

## dots.mocr

- Official repository: https://github.com/studio-dots-ai/dots.mocr
- Official vLLM integration requires `--chat-template-content-format string --trust-remote-code`; the repository documents `prompt_layout_all_en` for document parsing.
- The adapter uses the official layout prompt that requires one JSON object with bbox/category/text, then normalizes its layout elements into the benchmark fields. License: MIT.

## License and reproducibility

The benchmark must capture repository URL, model identifier, exact revision, package versions, license status and verification date. Vendor benchmark numbers are informational only; customer holdout metrics are authoritative.
