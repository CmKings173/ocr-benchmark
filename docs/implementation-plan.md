# OCR Benchmark Suite — Implementation Plan

## Delivery order

Current status: P0 safety/worker lifecycle, append-only checkpoints, barcode/system scopes, critical-field scoring, stage timing, batch/concurrency contracts and expanded reports are implemented. Remaining work is native model-batch support per adapter, GX10 telemetry validation, official Tier A dependency smoke runs and release hardening.

### Phase 1 — Core framework

- `pyproject.toml`, package layout and CLI entrypoint.
- Typed schemas for dataset, normalized prediction, status and run metadata.
- YAML configuration loading and validation.
- Dataset validator: one physical label per image, required metadata, readable files, valid field schema.
- Environment snapshot and run identity.
- Mock worker and end-to-end core smoke test.

### Phase 2 — Worker lifecycle

- Common worker contract: load, warmup, predict, metadata, unload.
- Subprocess worker manager with timeout, OOM/error capture and cleanup.
- Checkpoint/resume keyed by run, model, pass and sample.
- Dual-sided timing: worker stage timings plus orchestrator round-trip/pipeline timings.

### Phase 3 — Evaluation and measurement

- Independent accuracy and performance passes.
- Structural normalization and strict value exact-match metrics.
- Field, full-label, CER/WER and category metrics.
- OpenCV + ZXing-C++ barcode benchmark.
- CPU/RSS/unified-memory/GPU monitoring with explicit N/A handling.

### Phase 4 — Reporting and ranking

- JSONL raw predictions, JSON summary, CSV details and offline HTML report.
- Configurable gates, confidence intervals and score weights.
- Separate model, barcode and system benchmark views.
- Leaderboards by tier; invalid/skipped statuses excluded from rank.

### Phase 5 — Tier A adapters

- PP-OCRv6 detection/recognition adapter.
- GLM-OCR adapter using the official supported backend.
- PaddleOCR-VL adapter.
- MonkeyOCRv2 adapter.
- Official revision/license/dependency metadata and smoke tests per adapter.

### Phase 6 — Packaging and handoff

- Docker/compose path where ARM64 and dependency compatibility permit.
- Model cache and pre-download command.
- GX10 smoke benchmark with 100-image fixture.
- README, troubleshooting and model notes.

## Definition of Done for the first runnable milestone

The command below must run against a mock adapter and produce valid artifacts before real model integration begins:

```bash
python scripts/run_all.py --models mock --dataset ./data
```

It must validate the dataset, run both passes, persist raw predictions, calculate metrics, capture resources, handle a simulated worker failure, and emit JSON/CSV/HTML outputs.

## Guardrails

- No fake model result or fake metric.
- Official documentation/model card wins over stale assumptions.
- Accuracy and performance sample counts remain separate.
- Strict exact match is never replaced by business normalization.
- A failing model produces a status and does not stop later models.
