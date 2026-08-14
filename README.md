# OCR Benchmark Suite

Production-oriented OCR/VLM benchmark lab for customer shipping and product labels. The suite compares strict accuracy, field accuracy, full-label accuracy, barcode decoding, latency, throughput, reliability and resource usage on a fixed reference machine.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp data/ground_truth.example.json data/ground_truth.json
mkdir -p data/images
# Put an image matching the ground-truth record (for example data/images/001.jpg) here.
python scripts/validate_dataset.py --dataset data/images --ground-truth data/ground_truth.json
python scripts/run_all.py --models mock
```

The mock adapter is only for framework smoke tests. It is never a production benchmark result.

## Dataset

Put images under `data/images/`. Each ground-truth record must refer to exactly one physical label and include `label_count: 1`, `fields`, and optional `tags`, `label_type`, `full_text` and barcode metadata. The validator rejects `label_count != 1`.

## Models

Tier A adapters are registered as:

- `ppocr_v6`
- `glm_ocr`
- `paddleocr_vl_1_6`
- `monkey_ocr_v2_b_parsing`

Optional dependencies and endpoints are configured per model. Missing dependencies are reported, not replaced with fake output. See [model notes](docs/model_notes.md).

## Reports

Each model run writes checkpointed raw JSONL predictions, performance JSON, field/category/resource CSVs, barcode/system artifacts and an offline HTML report. Model, barcode and system scopes remain separate. HTTP VLM adapters are local-only by default; set `allow_remote_endpoint: true` only after approving the data boundary.

## Architecture and implementation

- [Accepted architecture](docs/architecture/ocr-benchmark-architecture.md)
- [Implementation plan](docs/implementation-plan.md)
- [Requirements and decisions](docs/requirements-questionnaire.md)
- [Dataset sourcing and V1 policy](docs/datasets.md)
- [Benchmark runbook](docs/runbook.md)
