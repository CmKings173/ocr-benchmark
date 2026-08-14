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

## 4. Smoke test, then install one model at a time

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

Install optional model dependencies only after confirming that the vendor provides an ARM64/GB10-compatible build. The generic requirements files are intentionally not a promise that every upstream wheel supports this platform:

```bash
# PP-OCR runs in the benchmark worker; install only after confirming the
# PaddlePaddle ARM64 + GB10 wheel/container is available.
uv pip install -r requirements/paddle.txt

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

Run one model first, with concurrency 1 and batch 1. Then increase workload only after checking `environment.json`, `performance.json`, and `resource_usage.csv`:

```bash
uv run python scripts/run_all.py --models ppocr_v6 \
  --dataset data/images/ocr_label_dataset_v1/images \
  --ground-truth data/images/ocr_label_dataset_v1/ground_truth.json \
  --output results/ppocr_v6
```

The accuracy pass and performance pass are separate. Do not use repeated performance inferences as extra accuracy samples. On a 128 GB unified-memory system, start one large VLM worker at a time; raise concurrency only after observing peak memory and OOM status.

## 5. Container path (optional)

The repository includes a CPU/core image and a GPU Compose profile:

```bash
docker compose up --build benchmark
docker compose --profile gpu up --build benchmark-gpu
```

The GPU profile only exposes the GPU; it does not install model-specific Paddle/vLLM runtimes. Build or select an ARM64/GB10-compatible base image for those dependencies, then mount `data/`, `results/`, and `model_cache/` as shown in `docker-compose.yml`.

## 6. Artifacts and resume

Each model output contains accuracy/performance checkpoints and separate model, barcode, and system reports. A rerun with the same output directory resumes completed samples. Use a new output directory whenever the dataset, model revision, framework, or device configuration changes.

Useful references:

- [ASUS Ascent GX10 specifications](https://www.asus.com/us/networking-iot-servers/desktop-ai-supercomputers/ultra-small-ai-supercomputers/asus-ascent-gx10/techspec/)
- [NVIDIA DGX Spark first boot](https://docs.nvidia.com/dgx/dgx-spark/first-boot.html)
- [NVIDIA DGX Spark container runtime](https://docs.nvidia.com/dgx/dgx-spark/nvidia-container-runtime-for-docker.html)
- [NVIDIA DGX Spark NGC guide](https://docs.nvidia.com/dgx/dgx-spark/ngc.html)
- [uv project sync and lock](https://docs.astral.sh/uv/concepts/projects/sync/)
