# Benchmark runbook

## 1. Validate the dataset

```bash
python scripts/validate_dataset.py \
  --dataset data/images \
  --ground-truth data/ground_truth.json
```

The validator requires relative image paths, readable images, a single annotated physical label and non-empty ground-truth fields.

## 2. Run the mock smoke test

```bash
python scripts/run_all.py --models mock \
  --dataset data/images \
  --ground-truth data/ground_truth.json
```

For a fast CI smoke test, use a benchmark config with one repetition, batch size 1 and concurrency 1.

## 3. Stage model adapters

```bash
python scripts/run_all.py \
  --models ppocr_v6,glm_ocr,paddleocr_vl_1_6,monkey_ocr_v2_b_parsing \
  --dataset data/images \
  --ground-truth data/ground_truth.json
```

Missing packages/endpoints are recorded as explicit statuses. They are never replaced with mock predictions.

## 4. GX10 run

Run `scripts/inspect_environment.py` first and keep its JSON with the results. Confirm the model package revision, device selection, driver/runtime and model cache path before collecting performance numbers. Run accuracy and performance as separate commands/configurations when changing repetitions or concurrency.

## 5. Resume and artifacts

Each model output contains `accuracy.checkpoint.jsonl` and `performance.checkpoint.jsonl`. A rerun with the same output directory resumes completed keys. Do not mix checkpoints from different dataset versions or model revisions.

Review at least:

- `summary.json` and `leaderboard.csv`
- `detailed_results.csv` and `field_accuracy.csv`
- `barcode_summary.json` and `system_results.json`
- `performance.json`, `resource_usage.csv` and `concurrency.csv`
- `environment.json`

Remote VLM endpoints are disabled by default. Set `allow_remote_endpoint: true` only after the image-data boundary and retention policy are approved.
