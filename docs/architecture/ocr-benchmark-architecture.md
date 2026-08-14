# OCR Benchmark Suite — Accepted Architecture

**Status:** Accepted  
**Scope:** V1 benchmark lab on ASUS Ascent GX10  
**Implementation status:** Core implementation in progress; this document remains the source of truth for implementation planning and acceptance.

## 1. Architectural shape

V1 uses a host-local orchestrator with one isolated worker process per model. A worker may later be wrapped by a container when its dependency stack requires it; the benchmark contract does not change.

```text
                         ORCHESTRATOR
                              |
        +---------------------+---------------------+
        |                     |                     |
   Dataset Validator   Environment Snapshot   Checkpoint/Resume
                              |
                       Worker Manager
          +----------------+---+----------------+
          |                    |                 |
      PP-OCR Worker        GLM Worker       Paddle-VL Worker
          |                    |                 |
          +--------------------+-----------------+
                              |
                    Normalized Prediction
                              |
             +----------------+----------------+
             |                                 |
          Evaluator                      Raw Artifacts
             |
     +-------+--------+---------+---------+
     |       |        |         |
 Accuracy  Field   Full Label  CER/WER  Detection*
             |
          Aggregator
             |
       +-----+-------+-------+
       |             |       |
     Gates       Leaderboard Report

 Resource Monitor remains outside workers and observes worker PID/processes.
```

The design is intentionally host-local: no scheduler, queue, web UI or distributed control plane is required in V1.

## 2. Component responsibilities

### Orchestrator

Owns configuration, model registry, dataset validation, run identity, environment capture, worker lifecycle, pass separation, checkpointing, aggregation and report generation. It never contains model-specific inference code.

### Worker

Implements the common lifecycle:

`load → warmup → predict → metadata → unload`

It accepts a normalized request and returns a normalized prediction envelope. Model-specific tensors, prompt formats and framework objects never cross the worker boundary.

### Shared domain

Contains dataset schemas, dynamic field schemas, structural normalization, strict validators, status taxonomy and serialization models. It must remain deterministic and framework-independent.

### Resource monitor

Runs outside workers and samples process RSS, CPU, unified memory, GPU utilization/counters, power and temperature when available. Missing counters are recorded as N/A.

### Barcode pipeline

Runs as an independent sibling pipeline using OpenCV and ZXing-C++. It produces barcode-specific results and is not silently folded into OCR model accuracy.

## 3. Timing contract

Latency is measured at two levels.

| Owner | Measurements |
|---|---|
| Worker | preprocess_ms, inference_ms, postprocess_ms |
| Orchestrator | worker_startup_ms, model_load_ms, request_round_trip_ms, pipeline_e2e_ms |

Worker timing uses a monotonic high-resolution clock around the actual stages. Orchestrator timing includes IPC, serialization and scheduling overhead. Reports expose both; neither replaces the other.

## 4. Execution passes

Accuracy and performance are independent executions.

### Accuracy pass

Each holdout image is processed once. Predictions are persisted before scoring. Metrics include strict field exact match, full-label exact match, CER/WER, barcode exactness and detection metrics when annotations exist.

### Performance pass

The model is warmed first, then executed under configured repetitions, batch sizes and concurrency levels. It measures P50/P90/P95/P99, throughput, failures and resource behavior. These repetitions never increase the accuracy sample count.

## 5. Normalization contract

Structural normalization is allowed:

`model output → canonical schema → evaluator`

Examples include mapping `SKU`, `sku_value` and equivalent model keys to the canonical field `sku`.

Value normalization is not applied before strict exact match. The evaluator must not turn `ABC-I23` into `ABC-123`, map `O` to `0`, remove hyphens or uppercase values for the strict metric.

If business normalization is needed, it is a separate metric named `business_normalized_field_match`. It never replaces `strict_field_exact_match`.

## 6. Evaluation scopes

### Model benchmark

`image → OCR model → model OCR result`

Measures the model's own text, detection and structured extraction behavior.

### Barcode benchmark

`image → barcode decoder → barcode result`

Measures decode success, exact value, symbology and decoder latency independently.

### System benchmark

`image → OCR model + barcode decoder → validation → ERP-shaped result`

Measures combined field correctness, full-label correctness, fallback eligibility and end-to-end latency. A barcode decoded by ZXing is credited to the barcode subsystem, not to the OCR model.

## 7. Failure and isolation behavior

Each worker has explicit startup, load, warmup, inference, timeout, invalid-output, OOM and unload states. A failed worker produces a result status and diagnostic metadata, is terminated and cleaned up, and does not stop later models.

The orchestrator persists raw predictions and run metadata before aggregation. Checkpoints are keyed by run, model, pass and sample identity, enabling resume without silently mixing accuracy and performance data.

## 8. Trade-offs accepted

- Subprocess lifecycle and IPC add measurable overhead; that overhead is reported separately and is part of system latency.
- A container is not mandatory for every worker; this reduces V1 complexity while preserving an upgrade path for dependency conflicts.
- Unified-memory counters may be incomplete on GB10; unsupported measurements remain N/A.
- V1 does not evaluate multi-label images, production fallback execution or other hardware without a separate benchmark run.

## 9. Revisit triggers

Reconsider this architecture when at least one of these becomes true:

- multiple concurrent users or remote jobs are required;
- worker startup dominates total run time at production scale;
- a model cannot be isolated reliably with subprocesses;
- multi-GX10 scheduling is required;
- multi-label annotations and region matching are available;
- benchmark results must be served as a long-running API.

## 10. Decision references

This architecture implements decisions D-001 through D-018 in `docs/requirements-questionnaire.md`, especially:

- D-014: orchestrator plus subprocess workers;
- D-015: dual-sided latency measurement;
- D-016: separate accuracy and performance executions;
- D-017: structural versus value normalization;
- D-018: separate model, barcode and system benchmarks.
