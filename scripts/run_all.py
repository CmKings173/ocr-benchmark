import argparse
import math
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocr_benchmark.barcode.runner import aggregate_barcode, build_system_records, run_barcode_pass
from ocr_benchmark.benchmark.runner import run_accuracy_pass, run_performance_pass
from ocr_benchmark.core.config import load_config, load_field_schema
from ocr_benchmark.core.environment import capture_environment
from ocr_benchmark.data.validator import DatasetValidationError, load_and_validate_dataset
from ocr_benchmark.reporting.export import export_results
from ocr_benchmark.reporting.leaderboard import write_leaderboard
from ocr_benchmark.reporting.scoring import aggregate_accuracy, aggregate_by_tag, composite_score, gate_result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="mock")
    parser.add_argument("--dataset", type=Path, default=Path("data/images"))
    parser.add_argument("--ground-truth", type=Path, default=Path("data/ground_truth.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark.yaml"))
    parser.add_argument("--models-config", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--fields-config", type=Path, default=Path("configs/fields.yaml"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Evaluate only the first N validated samples (use a new output directory)",
    )
    parser.add_argument(
        "--skip-performance",
        action="store_true",
        help="Run accuracy/barcode evaluation only; do not run latency/resource passes",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    try:
        dataset = load_and_validate_dataset(args.dataset, args.ground_truth, allowed_tags=config.allowed_tags or None)
    except DatasetValidationError as exc:
        print(f"FAIL: {exc}")
        return 1

    if args.max_samples is not None and args.max_samples <= 0:
        print("FAIL: --max-samples must be greater than zero")
        return 1
    field_schema = load_field_schema(args.fields_config)
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    validated_count = len(dataset.samples)
    if args.max_samples is not None:
        dataset = dataset.model_copy(update={"samples": dataset.samples[: args.max_samples]})
    print(
        f"dataset validated: {validated_count} samples; evaluating {len(dataset.samples)} samples; "
        f"models={','.join(models)}",
        flush=True,
    )
    output_root = args.output or config.output_dir
    multi_model = len(models) > 1
    invalid_models: list[str] = []
    model_configs_payload = yaml.safe_load(args.models_config.read_text()) if args.models_config.is_file() else {}
    model_configs = (model_configs_payload or {}).get("models", {})
    print("barcode pass: starting", flush=True)
    barcode_records = run_barcode_pass(dataset, args.dataset)
    barcode_summary = aggregate_barcode(barcode_records)
    print("barcode pass: completed", flush=True)
    leaderboard_rows = []

    minimum_repetitions = max(config.performance_repetitions, math.ceil(config.performance_min_iterations / max(len(dataset.samples), 1)))
    for model in models:
        model_config = dict(model_configs.get(model, {}))
        adapter_name = model_config.pop("adapter", model)
        if adapter_name in {"glm_ocr", "monkey_ocr_v2_b_parsing", "mistral_ocr4"}:
            model_config.setdefault("timeout_seconds", config.timeout_seconds)
        output_dir = (output_root / model) if (multi_model or args.output is None) else output_root
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"{model}: accuracy pass starting ({len(dataset.samples)} samples)", flush=True)
        records = run_accuracy_pass(
            dataset,
            args.dataset,
            adapter_name,
            model_config=model_config,
            timeout_seconds=config.timeout_seconds,
            checkpoint_path=output_dir / "accuracy.checkpoint.jsonl",
            field_schema=field_schema,
        )
        print(f"{model}: accuracy pass completed", flush=True)
        if args.skip_performance:
            performance = {
                "model": model,
                "status": "SKIPPED",
                "scope": "accuracy_only",
                "samples": 0,
                "failures": 0,
                "failure_rate": None,
                "p50_ms": None,
                "p90_ms": None,
                "p95_ms": None,
                "p99_ms": None,
                "mean_ms": None,
                "throughput_images_per_second": None,
                "batch_results": [],
                "concurrency_results": [],
                "resource_usage": {},
            }
            print(f"{model}: performance pass skipped (--skip-performance)", flush=True)
        else:
            print(f"{model}: performance pass starting (repetitions={minimum_repetitions}, batch={config.batch_sizes}, concurrency={config.concurrency})", flush=True)
            performance = run_performance_pass(
                dataset,
                args.dataset,
                adapter_name,
                repetitions=minimum_repetitions,
                model_config=model_config,
                timeout_seconds=config.timeout_seconds,
                batch_sizes=config.batch_sizes,
                concurrency_levels=config.concurrency,
                checkpoint_path=output_dir / "performance.checkpoint.jsonl",
                warmup_iterations=config.warmup_iterations,
            )
            print(f"{model}: performance pass completed", flush=True)
        accuracy = aggregate_accuracy(records)
        system_records = build_system_records(records, barcode_records, dataset, field_schema=field_schema)
        system_accuracy = aggregate_accuracy(system_records)
        gates = gate_result(
            {
                **accuracy,
                "p95_ms": performance.get("p95_ms"),
                "p99_ms": performance.get("p99_ms"),
                "peak_unified_memory_bytes": performance.get("resource_usage", {}).get("peak_unified_memory_bytes"),
            },
            config.production_gates,
        )
        performance_failure_rate = performance.get("failure_rate")
        performance_valid = performance.get("status") == "SUCCESS" and performance_failure_rate is not None and float(performance_failure_rate) < 1.0
        if args.skip_performance:
            run_valid = bool(gates.get("valid"))
            gates["performance_skipped"] = True
            gates["eligible"] = False
        else:
            run_valid = bool(gates.get("valid")) and performance_valid
        gates["valid"] = run_valid
        if not args.skip_performance:
            gates["eligible"] = bool(gates.get("eligible") and run_valid)
        score = composite_score({**accuracy, "p95_ms": performance.get("p95_ms"), "eligible": gates["eligible"]}, config.score_weights)
        export_results(
            records,
            performance,
            output_dir,
            environment=capture_environment(),
            accuracy=accuracy,
            gates=gates,
            barcode_records=barcode_records,
            barcode_summary=barcode_summary,
            system_records=system_records,
            system_accuracy=system_accuracy,
            category_accuracy=aggregate_by_tag(records, dataset),
            score=score,
        )
        leaderboard_rows.append({"model": model, "eligible": gates["eligible"], "full_label_accuracy": accuracy.get("full_label_accuracy"), "critical_field_accuracy": accuracy.get("critical_field_accuracy"), "field_exact_accuracy": accuracy.get("field_exact_accuracy"), "p95_ms": performance.get("p95_ms"), "failure_rate": performance.get("failure_rate"), "score": score if score is not None else "N/A"})
        if not gates["valid"]:
            invalid_models.append(model)
        print(f"completed {model}: {output_dir} valid={gates['valid']} eligible={gates['eligible']}")

    write_leaderboard(leaderboard_rows, output_root / "leaderboard.csv")
    if invalid_models:
        print(f"invalid model runs: {', '.join(invalid_models)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
