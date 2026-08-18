# HunyuanOCR-1.5, Unlimited-OCR và dots.mocr

Ba model này được đăng ký như các adapter OpenAI-compatible riêng. Benchmark
không tự tải weights và không tự khởi động server; hãy khởi động đúng server
vLLM trước, sau đó chạy `scripts/run_all.py` với
`configs/models.huggingface.yaml`.

## HunyuanOCR-1.5

Model: `tencent/HunyuanOCR`  
Port mặc định: `8107`  
Adapter: `hunyuan_ocr_1_5`

Hunyuan có 12 task prompt cố định trong tài liệu chính thức. Config benchmark
dùng `task_type: spotting_json`, vì task này yêu cầu JSON array gồm `box` và
`text`, phù hợp với việc đọc tem. Nếu cần Markdown document parsing, đổi thành
`task_type: doc_parse`; không tự thay prompt bằng prompt JSON chung.

Server chính thức (CUDA 13, unified environment):

```bash
MODEL_PATH=./HunyuanOCR GPU=0 PORT=8107 bash inference/vLLM/serve.sh
curl -sf http://127.0.0.1:8107/v1/models
```

Weights chính thức:

```bash
hf download tencent/HunyuanOCR --local-dir ./HunyuanOCR --exclude 'v1.0/*'
```

## Unlimited-OCR

Model: `baidu/Unlimited-OCR`  
Port mặc định: `8106`  
Adapter: `unlimited_ocr`

Model này bắt buộc recipe vLLM riêng. Dùng custom image của tác giả và đăng ký
NGram processor; thiếu processor, `<image>` prefix hoặc
`skip_special_tokens=false` có thể trả output rỗng/lặp:

```bash
docker run --rm --gpus all --network host --ipc host \
  -v "$HF_HOME:/root/.cache/huggingface" \
  vllm/vllm-openai:unlimited-ocr \
  baidu/Unlimited-OCR \
  --trust-remote-code \
  --logits_processors vllm.model_executor.models.unlimited_ocr:NGramPerReqLogitsProcessor \
  --no-enable-prefix-caching \
  --mm-processor-cache-gb 0 \
  --port 8106
```

Adapter gửi prompt chính thức `<image>document parsing.`,
`skip_special_tokens: false` và `vllm_xargs: {ngram_size: 35,
window_size: 128}`. Parser loại grounding markers rồi trích fields từ Markdown.

## dots.mocr

Model mặc định theo repo vLLM: `rednote-hilab/dots.mocr`  
Port mặc định: `8108`  
Adapter: `dots_mocr`

```bash
vllm serve rednote-hilab/dots.mocr \
  --host 127.0.0.1 --port 8108 \
  --served-model-name rednote-hilab/dots.mocr \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --chat-template-content-format string \
  --trust-remote-code
```

Adapter dùng `prompt_layout_all_en` và prompt layout chính thức của model,
đồng thời thêm prefix ảnh `<|img|><|imgpad|><|endofimg|>` mà client vLLM
chính thức yêu cầu. Prompt yêu cầu output là **một JSON object duy nhất**.
Parser đọc các phần tử layout (`category`, `bbox`, `text`) theo reading order
rồi ánh xạ sang schema benchmark. Nếu server quảng cáo alias `dots-studio/dots.mocr`, chỉ cần đổi
`model:` trong YAML; adapter không hard-code alias ngoài giá trị mặc định.

## JSON contract

| Model | JSON do model yêu cầu chính thức | Output native thường gặp | Adapter benchmark |
|---|---|---|---|
| HunyuanOCR-1.5 | Có, với task `spotting_json` | JSON array `box` + `text` hoặc Markdown với `doc_parse` | Parse cả JSON/Markdown |
| Unlimited-OCR | Không phải contract JSON | Markdown + grounding markers | Strip markers, parse text/fields |
| dots.mocr | Có, prompt layout yêu cầu single JSON object | Layout JSON (`bbox`, category, text) | Parse layout JSON và text |

“JSON hỗ trợ” ở đây là output task/prompt chính thức; nó không thay thế việc
benchmark ghi `raw_text`, `fields` và artifacts riêng. `fields` vẫn được chuẩn
hóa vào schema benchmark để so sánh công bằng.

## Chạy benchmark

```bash
.venv-vllm/bin/python scripts/run_all.py \
  --config configs/benchmark.gx10-smoke.yaml \
  --models-config configs/models.huggingface.yaml \
  --models hunyuan_ocr_1_5 \
  --dataset data/images/ocr_label_dataset_v1/images \
  --ground-truth data/images/ocr_label_dataset_v1/ground_truth.json \
  --max-samples 3 --skip-performance \
  --output results/hunyuan-smoke
```

Sau khi parser/transport smoke pass, bỏ `--max-samples` và
`--skip-performance` để chạy full. Chạy mỗi model trên server riêng; không chạy
ba server nặng đồng thời khi GPU đang có model khác.
