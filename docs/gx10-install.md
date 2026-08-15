# GX10 installation and benchmark runbook

ASUS Ascent GX10 is an ARM64 Ubuntu system built around NVIDIA GB10/Blackwell with unified memory. Treat it as an ARM64/GB10 target, not as an x86 workstation. Verify the installed DGX OS/driver release before choosing framework wheels or containers.

## 1. Host preflight

Run these commands on the GX10:

```bash
uname -m
cat /etc/os-release
nvidia-smi
docker --version
nvidia-ctk --version
```

Expected architecture is `aarch64`. If `nvidia-smi` fails, stop here and fix the host driver/firmware first. Do not collect benchmark numbers while the GPU runtime is unhealthy.

Validate Docker GPU access with an NVIDIA image compatible with the installed driver:

```bash
docker run --rm --gpus=all \
  nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04 nvidia-smi
```

The DGX Spark/GX10 software stack normally includes Docker and the NVIDIA Container Runtime. If the command reports a missing runtime, check `nvidia-ctk --version` and the Docker daemon configuration before installing anything manually.

## 2. Clone and install the benchmark core

```bash
git clone https://github.com/CmKings173/ocr-benchmark.git
cd ocr-benchmark
git checkout main

sudo apt update
sudo apt install -y python3-dev build-essential \
  libglib2.0-0 libgl1

curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv python install 3.11
uv sync --dev
source .venv/bin/activate
```

Keep model downloads outside Git:

```bash
mkdir -p data/images model_cache results
export HF_HOME="$PWD/model_cache/huggingface"
```

## 3. Add and validate data

```bash
cp data/ground_truth.example.json data/ground_truth.json
# Copy the corresponding images into data/images/
uv run python scripts/inspect_environment.py
uv run python scripts/validate_dataset.py \
  --dataset data/images \
  --ground-truth data/ground_truth.json
```

If you are using the included starter pack already located at `data/images/ocr_label_dataset_v1/`, validate it with its nested paths instead of copying files:

```bash
uv run python scripts/validate_dataset.py \
  --dataset data/images/ocr_label_dataset_v1/images \
  --ground-truth data/images/ocr_label_dataset_v1/ground_truth.json
```

V1 accepts exactly one physical label per image. The validator rejects `label_count != 1` and unsafe paths. Keep the image set and ground-truth file versioned together; never resume a checkpoint against a changed dataset.

## 4. Phase 1: PP-OCRv6 + barcode (real model)

Start with the framework smoke test:

```bash
uv run python scripts/run_all.py --config configs/benchmark.gx10-smoke.yaml --models mock \
  --dataset data/images/ocr_label_dataset_v1/images \
  --ground-truth data/images/ocr_label_dataset_v1/ground_truth.json \
  --output results/mock-smoke
```

The smoke config keeps the 100-image accuracy check but reduces performance to one repetition, batch size 1 and concurrency 1. For the full benchmark, use the default `configs/benchmark.yaml`.

```bash
uv run python scripts/run_all.py --models mock \
  --dataset data/images/ocr_label_dataset_v1/images \
  --ground-truth data/images/ocr_label_dataset_v1/ground_truth.json \
  --output results/mock
```

Install the PP-OCR runtime only after confirming that the vendor provides an ARM64/GB10-compatible build. The generic requirements file is intentionally not a promise that every upstream wheel supports this platform:

```bash
# PP-OCR runs in the benchmark worker. Run this inside the dedicated env.
uv venv .venv-ppocr --python 3.12
uv pip install --python .venv-ppocr/bin/python -r requirements/paddle.txt

# Verify the actual engine before downloading/running a 100-image benchmark.
.venv-ppocr/bin/python - <<'PY'
import cv2
import paddle
import paddleocr
print("paddleocr", getattr(paddleocr, "__version__", "unknown"))
print("paddle", paddle.__version__)
print("compiled_with_cuda", paddle.is_compiled_with_cuda())
print("device", paddle.device.get_device())
print("opencv", cv2.__version__)
try:
    import zxingcpp
    print("zxingcpp", "available")
except Exception as exc:
    print("zxingcpp", "ERROR", exc)
PY

# This is the first real model download/load. It uses one clear image only.
.venv-ppocr/bin/python scripts/preflight_model.py \
  --model ppocr_v6 \
  --dataset data/images/ocr_label_dataset_v1/images \
  --ground-truth data/images/ocr_label_dataset_v1/ground_truth.json \
  --timeout 300

# Continue only when the preflight JSON ends with: "valid": true.
.venv-ppocr/bin/python scripts/run_all.py \
  --config configs/benchmark.gx10-smoke.yaml \
  --models ppocr_v6 \
  --dataset data/images/ocr_label_dataset_v1/images \
  --ground-truth data/images/ocr_label_dataset_v1/ground_truth.json \
  --output results/ppocr-v6-smoke-v2

# Run the full performance pass only after the smoke run exits 0.
.venv-ppocr/bin/python scripts/run_all.py \
  --config configs/benchmark.yaml \
  --models ppocr_v6 \
  --dataset data/images/ocr_label_dataset_v1/images \
  --ground-truth data/images/ocr_label_dataset_v1/ground_truth.json \
  --output results/ppocr-v6-full-v2

## 5. Optional model environments

Install optional model dependencies only after confirming that the vendor provides an ARM64/GB10-compatible build. The generic requirements files are intentionally not a promise that every upstream wheel supports this platform:

```bash
# GLM-OCR and MonkeyOCR are HTTP servers. Keep their conflicting vLLM
# requirements in separate environments.
uv venv .venv-glm --python 3.11
uv pip install --python .venv-glm/bin/python -r requirements/glm.txt
uv venv .venv-monkey --python 3.11
uv pip install --python .venv-monkey/bin/python -r requirements/monkey.txt
```

For HTTP VLMs, start the server using its official GB10/ARM64 instructions and keep the endpoint local (`127.0.0.1`). The configured endpoints are:

- GLM-OCR: `http://127.0.0.1:8080/v1`
- MonkeyOCR: `http://127.0.0.1:8000/v1`

For HTTP VLMs, run their official server first, then use the same
`scripts/preflight_model.py` command with the correct model configuration.
The benchmark worker must never be pointed at a server hosting a different
model. On a 128 GB unified-memory system, start one large VLM worker at a
time; raise concurrency only after observing peak memory and OOM status.

### Hugging Face-backed VLM smoke tests

The benchmark client does not download VLM weights. The server downloads the
Hugging Face checkpoint on first start (or uses the cache in `HF_HOME`). Use
the dedicated config so the requested model id matches the server id:

```bash
export HF_HOME="$PWD/model_cache/huggingface"
mkdir -p "$HF_HOME"
git pull --ff-only origin main

# Install the HF CLI once if it is not already available.
uv pip install --python .venv-ppocr/bin/python 'huggingface-hub>=0.30'
```

Start only one server at a time on the GX10. Each command downloads from
Hugging Face automatically and exposes an OpenAI-compatible endpoint:

```bash
# Qwen2-VL (port 8101)
HF_HOME="$HF_HOME" vllm serve Qwen/Qwen2-VL-7B-Instruct \
  --host 127.0.0.1 --port 8101 \
  --served-model-name Qwen/Qwen2-VL-7B-Instruct \
  --max-model-len 4096 --gpu-memory-utilization 0.80

# Surya OCR 2 (port 8102)
HF_HOME="$HF_HOME" vllm serve datalab-to/surya-ocr-2 \
  --host 127.0.0.1 --port 8102 \
  --served-model-name datalab-to/surya-ocr-2 \
  --trust-remote-code --max-model-len 8192 --gpu-memory-utilization 0.70

# DeepSeek-OCR (port 8103)
HF_HOME="$HF_HOME" vllm serve deepseek-ai/DeepSeek-OCR \
  --host 127.0.0.1 --port 8103 \
  --served-model-name deepseek-ai/DeepSeek-OCR \
  --trust-remote-code --max-model-len 8192 --gpu-memory-utilization 0.75

# MonkeyOCRv2-B-Parsing (port 8104)
HF_HOME="$HF_HOME" vllm serve zenosai/MonkeyOCRv2-B-Parsing \
  --host 127.0.0.1 --port 8104 \
  --served-model-name zenosai/MonkeyOCRv2-B-Parsing \
  --trust-remote-code --max-model-len 8192 --gpu-memory-utilization 0.70
```

Before running the benchmark, verify the endpoint advertises the expected
model id, for example:

```bash
curl -s http://127.0.0.1:8101/v1/models
```

Then run a three-image smoke test. Always use a new output directory when
changing model, checkpoint, or server port:

```bash
.venv-ppocr/bin/python scripts/run_all.py \
  --config configs/benchmark.gx10-smoke.yaml \
  --models-config configs/models.huggingface.yaml \
  --models qwen2_vl_7b \
  --dataset data/images/ocr_label_dataset_v1/images \
  --ground-truth data/images/ocr_label_dataset_v1/ground_truth.json \
  --max-samples 3 --skip-performance \
  --output results/qwen2-vl-hf-accuracy-3
```

Replace only `--models` and the output directory for `surya_ocr_2`,
`deepseek_ocr`, and `monkey_ocr_v2_b_parsing`. A `valid=True eligible=False`
result is expected for this accuracy-only triage run; `eligible` requires a
separate performance pass.

The accuracy pass and performance pass are separate. Do not use repeated
performance inferences as extra accuracy samples.

## 6. Container path (optional)

The repository includes a CPU/core image and a GPU Compose profile:

```bash
docker compose up --build benchmark
docker compose --profile gpu up --build benchmark-gpu
```

The GPU profile only exposes the GPU; it does not install model-specific Paddle/vLLM runtimes. Build or select an ARM64/GB10-compatible base image for those dependencies, then mount `data/`, `results/`, and `model_cache/` as shown in `docker-compose.yml`.

## 7. Artifacts and resume

Each model output contains accuracy/performance checkpoints and separate model, barcode, and system reports. A rerun with the same output directory resumes completed samples. Use a new output directory whenever the dataset, model revision, framework, or device configuration changes.

Useful references:

- [ASUS Ascent GX10 specifications](https://www.asus.com/us/networking-iot-servers/desktop-ai-supercomputers/ultra-small-ai-supercomputers/asus-ascent-gx10/techspec/)
- [NVIDIA DGX Spark first boot](https://docs.nvidia.com/dgx/dgx-spark/first-boot.html)
- [NVIDIA DGX Spark container runtime](https://docs.nvidia.com/dgx/dgx-spark/nvidia-container-runtime-for-docker.html)
- [NVIDIA DGX Spark NGC guide](https://docs.nvidia.com/dgx/dgx-spark/ngc.html)
- [uv project sync and lock](https://docs.astral.sh/uv/concepts/projects/sync/)
