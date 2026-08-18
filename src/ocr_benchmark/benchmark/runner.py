from __future__ import annotations

import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Optional

from ocr_benchmark.benchmark.checkpoint import CheckpointStore
from ocr_benchmark.benchmark.monitor import ResourceMonitor
from ocr_benchmark.benchmark.worker_manager import SubprocessWorker
from ocr_benchmark.core.schemas import Dataset, Prediction, RunStatus
from ocr_benchmark.extraction.fields import extract_fields
from ocr_benchmark.metrics.fields import field_exact_metrics
from ocr_benchmark.metrics.text import cer, wer


def _failure_prediction(model: str, image: str, status: RunStatus, error: str) -> Prediction:
    return Prediction(model=model, image=image, status=status, error=error)


def _classify_error(exc: Exception) -> RunStatus:
    message = str(exc).upper()
    if "INVALID_OUTPUT" in message or "VALIDATIONERROR" in message or "VALIDATION ERROR" in message:
        return RunStatus.INVALID_OUTPUT
    if "TIMEOUT" in message or "TIMED OUT" in message:
        return RunStatus.TIMEOUT
    if "NOT_CONFIGURED" in message:
        return RunStatus.NOT_CONFIGURED
    if "NOT_SUPPORTED" in message:
        return RunStatus.NOT_SUPPORTED
    if "DEPENDENCY" in message or isinstance(exc, ImportError):
        return RunStatus.DEPENDENCY_ERROR
    if "OUT OF MEMORY" in message or "OOM" in message:
        return RunStatus.OOM
    return RunStatus.EXCEPTION


def _image_path(dataset_root: Path, image: str) -> Path:
    root = dataset_root.resolve()
    candidate = (root / image).resolve()
    candidate.relative_to(root)
    return candidate


def _critical_fields(sample: Any, field_schema: Optional[dict[str, Any]]) -> list[str]:
    if getattr(sample, "critical_fields", None):
        return [field for field in sample.critical_fields if field in sample.fields]
    label_schema = (field_schema or {}).get("label_types", {}).get(sample.label_type, {})
    configured = label_schema.get("fields", {}) if isinstance(label_schema, dict) else {}
    return [field for field, spec in configured.items() if isinstance(spec, dict) and spec.get("critical") and field in sample.fields]


def _record(image: str, prediction: Prediction, sample: Any, field_schema: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    # Keep artifacts portable and avoid leaking the local absolute dataset path.
    prediction.image = image
    record: dict[str, Any] = {"image": image, "prediction": prediction.model_dump(mode="json")}
    if sample is None:
        return record
    prediction.fields = extract_fields(prediction.raw_text, prediction.fields, sample.fields)
    record["prediction"] = prediction.model_dump(mode="json")
    required = list(sample.fields.keys())
    record["field_metrics"] = field_exact_metrics(sample.fields, prediction.fields, required, _critical_fields(sample, field_schema))
    record["cer"] = cer(sample.full_text, prediction.raw_text)
    record["wer"] = wer(sample.full_text, prediction.raw_text)
    return record


def _load_worker(worker: SubprocessWorker, warmup_iterations: int = 0, warmup_image: Optional[Path] = None) -> tuple[float, float]:
    startup_ms = worker.start()
    load_started = time.perf_counter()
    worker.request({"operation": "load"})
    model_load_ms = (time.perf_counter() - load_started) * 1000
    if warmup_iterations:
        if warmup_image is None:
            raise RuntimeError("warmup image is required when warmup_iterations > 0")
        worker.request({"operation": "warmup", "iterations": warmup_iterations, "image": str(warmup_image)})
    return startup_ms, model_load_ms


def _recover_worker(worker: SubprocessWorker, warmup_iterations: int = 0, warmup_image: Optional[Path] = None) -> bool:
    try:
        worker.stop()
        _load_worker(worker, warmup_iterations=warmup_iterations, warmup_image=warmup_image)
        return True
    except Exception:
        worker.stop()
        return False


def run_accuracy_pass(
    dataset: Dataset,
    dataset_root: Path,
    model: str,
    model_config: Optional[dict[str, Any]] = None,
    timeout_seconds: float = 60.0,
    checkpoint_path: Optional[Path] = None,
    resume: bool = True,
    warmup_iterations: int = 0,
    warmup_image: Optional[Path] = None,
    field_schema: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    checkpoint = CheckpointStore(checkpoint_path) if resume or checkpoint_path else CheckpointStore()
    records: list[dict[str, Any]] = []
    pending = []
    for sample in dataset.samples:
        saved = checkpoint.get(sample.image) if resume else None
        if saved is not None:
            records.append(saved["record"])
        else:
            pending.append(sample)
    if not pending:
        return records

    worker = SubprocessWorker(timeout_seconds=timeout_seconds, model_name=model, model_config=model_config)
    monitor: Optional[ResourceMonitor] = None
    startup_ms: Optional[float] = None
    model_load_ms: Optional[float] = None
    try:
        try:
            startup_ms, model_load_ms = _load_worker(worker, warmup_iterations=warmup_iterations, warmup_image=warmup_image)
        except Exception as exc:
            status = _classify_error(exc)
            for sample in pending:
                record = _record(sample.image, _failure_prediction(model, sample.image, status, str(exc)), sample, field_schema)
                records.append(record)
                checkpoint.append(sample.image, record)
            return records

        for sample in pending:
            image_path = _image_path(dataset_root, sample.image)
            started = time.perf_counter()
            try:
                response = worker.request({"operation": "predict", "image": str(image_path)})
                prediction = Prediction.model_validate(response["prediction"])
                prediction.timing.worker_startup_ms = startup_ms
                prediction.timing.model_load_ms = model_load_ms
                prediction.timing.request_round_trip_ms = (time.perf_counter() - started) * 1000
                prediction.timing.pipeline_e2e_ms = prediction.timing.request_round_trip_ms
            except Exception as exc:
                status = _classify_error(exc)
                prediction = _failure_prediction(model, sample.image, status, str(exc))
                if status in {RunStatus.TIMEOUT, RunStatus.OOM, RunStatus.EXCEPTION}:
                    _recover_worker(worker)
            record = _record(sample.image, prediction, sample, field_schema)
            records.append(record)
            checkpoint.append(sample.image, record)
        return records
    finally:
        worker.stop()


def _percentiles(values: list[float]) -> dict[str, Optional[float]]:
    if not values:
        return {"p50_ms": None, "p90_ms": None, "p95_ms": None, "p99_ms": None, "mean_ms": None}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
        return ordered[index]

    return {
        "p50_ms": percentile(0.50),
        "p90_ms": percentile(0.90),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
        "mean_ms": statistics.fmean(values),
    }


def _run_batch_level(
    dataset: Dataset,
    dataset_root: Path,
    model: str,
    model_config: Optional[dict[str, Any]],
    timeout_seconds: float,
    repetitions: int,
    batch_size: int,
    warmup_image: Path,
) -> dict[str, Any]:
    worker = SubprocessWorker(timeout_seconds=timeout_seconds, model_name=model, model_config=model_config)
    monitor: Optional[ResourceMonitor] = None
    latencies: list[float] = []
    failures = 0
    try:
        _load_worker(worker, warmup_iterations=1, warmup_image=warmup_image)
        monitor = ResourceMonitor(pid=worker.process.pid if worker.process else None)
        monitor.start()
        for _ in range(repetitions):
            for offset in range(0, len(dataset.samples), batch_size):
                samples = dataset.samples[offset : offset + batch_size]
                started = time.perf_counter()
                try:
                    worker.request({"operation": "predict_batch", "images": [str(_image_path(dataset_root, sample.image)) for sample in samples]})
                    latencies.append((time.perf_counter() - started) * 1000)
                except Exception:
                    failures += len(samples)
                    _recover_worker(worker, warmup_iterations=1, warmup_image=warmup_image)
        metrics = _percentiles(latencies)
        total_images = len(dataset.samples) * repetitions
        total_ms = sum(latencies)
        return {"batch_size": batch_size, "status": "SUCCESS", "images": total_images - failures, "failures": failures, "failure_rate": failures / max(total_images, 1), **metrics, "throughput_images_per_second": (total_images - failures) / (total_ms / 1000) if total_ms else 0.0, "execution_mode": "adapter_batch_contract_sequential_fallback", "resource_usage": ResourceMonitor.summary(monitor.stop() if monitor else [])}
    except Exception as exc:
        return {"batch_size": batch_size, "status": _classify_error(exc).value, "images": 0, "failures": len(dataset.samples) * repetitions, "failure_rate": 1.0, **_percentiles([]), "throughput_images_per_second": 0.0, "resource_usage": ResourceMonitor.summary(monitor.stop() if monitor else []), "error": str(exc)}
    finally:
        if monitor is not None:
            monitor.stop()
        worker.stop()


def _run_concurrency_level(
    dataset: Dataset,
    dataset_root: Path,
    model: str,
    model_config: Optional[dict[str, Any]],
    timeout_seconds: float,
    repetitions: int,
    concurrency: int,
    warmup_image: Path,
) -> dict[str, Any]:
    workers = [SubprocessWorker(timeout_seconds=timeout_seconds, model_name=model, model_config=model_config) for _ in range(concurrency)]
    # A worker is a line-oriented subprocess protocol, not a thread-safe RPC
    # client.  Keep each worker's stdin/stdout pair serialized while allowing
    # different workers to process requests concurrently.
    worker_locks = [Lock() for _ in workers]
    monitors: list[ResourceMonitor] = []
    latencies: list[float] = []
    failures = 0
    try:
        for worker in workers:
            _load_worker(worker, warmup_iterations=1, warmup_image=warmup_image)
            monitor = ResourceMonitor(pid=worker.process.pid if worker.process else None)
            monitor.start()
            monitors.append(monitor)

        def predict(item: tuple[int, Any]) -> float:
            worker = workers[item[0] % len(workers)]
            lock = worker_locks[item[0] % len(workers)]
            with lock:
                started = time.perf_counter()
                try:
                    worker.request({"operation": "predict", "image": str(_image_path(dataset_root, item[1].image))})
                except Exception:
                    _recover_worker(worker, warmup_iterations=1, warmup_image=warmup_image)
                    raise
                return (time.perf_counter() - started) * 1000

        items = [(index % concurrency, sample) for index, sample in enumerate(dataset.samples * repetitions)]
        wall_started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(predict, item) for item in items]
            for future in as_completed(futures):
                try:
                    latencies.append(future.result())
                except Exception:
                    failures += 1
        wall_time_ms = (time.perf_counter() - wall_started) * 1000
        metrics = _percentiles(latencies)
        resource_samples = [sample for monitor in monitors for sample in monitor.stop()]
        return {"concurrency": concurrency, "status": "SUCCESS", "images": len(latencies), "failures": failures, "failure_rate": failures / max(len(items), 1), **metrics, "wall_time_ms": wall_time_ms, "throughput_images_per_second": len(latencies) / (wall_time_ms / 1000) if wall_time_ms else 0.0, "resource_usage": ResourceMonitor.summary(resource_samples)}
    except Exception as exc:
        resource_samples = [sample for monitor in monitors for sample in monitor.stop()]
        return {"concurrency": concurrency, "status": _classify_error(exc).value, "images": 0, "failures": len(dataset.samples) * repetitions, "failure_rate": 1.0, **_percentiles([]), "throughput_images_per_second": 0.0, "resource_usage": ResourceMonitor.summary(resource_samples), "error": str(exc)}
    finally:
        for monitor in monitors:
            monitor.stop()
        for worker in workers:
            worker.stop()


def run_performance_pass(
    dataset: Dataset,
    dataset_root: Path,
    model: str,
    repetitions: int = 3,
    model_config: Optional[dict[str, Any]] = None,
    timeout_seconds: float = 60.0,
    batch_sizes: Optional[list[int]] = None,
    concurrency_levels: Optional[list[int]] = None,
    checkpoint_path: Optional[Path] = None,
    warmup_iterations: int = 5,
) -> dict[str, Any]:
    worker = SubprocessWorker(timeout_seconds=timeout_seconds, model_name=model, model_config=model_config)
    monitor: Optional[ResourceMonitor] = None
    latencies: list[float] = []
    failures = 0
    warmup_image = _image_path(dataset_root, dataset.samples[0].image) if dataset.samples else None
    checkpoint = CheckpointStore(checkpoint_path) if checkpoint_path else CheckpointStore()
    try:
        try:
            worker.start()
            monitor = ResourceMonitor(pid=worker.process.pid if worker.process else None)
            monitor.start()
            worker.request({"operation": "load"})
            if warmup_iterations:
                if warmup_image is None:
                    raise RuntimeError("warmup image is required")
                worker.request({"operation": "warmup", "iterations": warmup_iterations, "image": str(warmup_image)})
        except Exception as exc:
            return {"model": model, "status": _classify_error(exc).value, "samples": 0, "failures": len(dataset.samples), "failure_rate": 1.0, **_percentiles([]), "throughput_images_per_second": 0.0, "batch_results": [], "concurrency_results": [], "resource_usage": ResourceMonitor.summary(monitor.stop() if monitor else []), "error": str(exc)}

        for repetition in range(repetitions):
            for sample in dataset.samples:
                key = f"{model}:performance:{repetition}:{sample.image}"
                saved = checkpoint.get(key)
                if saved is not None:
                    record = saved["record"]
                    if record.get("status") == "SUCCESS":
                        latencies.append(float(record["latency_ms"]))
                    else:
                        failures += 1
                    continue
                started = time.perf_counter()
                status = "SUCCESS"
                try:
                    worker.request({"operation": "predict", "image": str(_image_path(dataset_root, sample.image))})
                    latency_ms = (time.perf_counter() - started) * 1000
                    latencies.append(latency_ms)
                except Exception as exc:
                    latency_ms = (time.perf_counter() - started) * 1000
                    failures += 1
                    status = _classify_error(exc).value
                    if status in {RunStatus.TIMEOUT.value, RunStatus.OOM.value, RunStatus.EXCEPTION.value}:
                        _recover_worker(worker)
                checkpoint.append(key, {"image": sample.image, "repetition": repetition, "status": status, "latency_ms": latency_ms})

        metrics = _percentiles(latencies)
        total_ms = sum(latencies)
        result: dict[str, Any] = {"model": model, "status": "SUCCESS", "samples": len(latencies), "failures": failures, "failure_rate": failures / max(failures + len(latencies), 1), **metrics, "throughput_images_per_second": len(latencies) / (total_ms / 1000) if total_ms else 0.0}
        result["batch_results"] = [_run_batch_level(dataset, dataset_root, model, model_config, timeout_seconds, repetitions, size, warmup_image) for size in sorted(set(batch_sizes or [1])) if size > 0]
        result["concurrency_results"] = [_run_concurrency_level(dataset, dataset_root, model, model_config, timeout_seconds, repetitions, level, warmup_image) for level in sorted(set(concurrency_levels or [1])) if level > 0]
        result["resource_usage"] = ResourceMonitor.summary(monitor.stop() if monitor else [])
        return result
    finally:
        if monitor is not None:
            monitor.stop()
        worker.stop()
