from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocr_benchmark.barcode.decoder import OpenCVBarcodeDecoder, ZXingCppDecoder
from ocr_benchmark.benchmark.worker_manager import SubprocessWorker
from ocr_benchmark.data.validator import load_and_validate_dataset


def _model_config(path: Path, model: str) -> tuple[str, dict[str, Any]]:
    payload = yaml.safe_load(path.read_text()) if path.is_file() else {}
    config = dict((payload or {}).get("models", {}).get(model, {}))
    adapter = config.pop("adapter", model)
    return adapter, config


def _preflight_image(dataset: Any, requested: str | None) -> str:
    if requested:
        return requested
    for sample in dataset.samples:
        if "clear" in (sample.tags or []):
            return sample.image
    return dataset.samples[0].image


def main() -> int:
    parser = argparse.ArgumentParser(description="Load one model and decode one image before a 100-image run.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--models-config", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--image", help="relative image path; defaults to the first clear sample")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    dataset = load_and_validate_dataset(args.dataset, args.ground_truth)
    image = _preflight_image(dataset, args.image)
    image_path = (args.dataset / image).resolve()
    adapter, config = _model_config(args.models_config, args.model)
    worker = SubprocessWorker(timeout_seconds=args.timeout, model_name=adapter, model_config=config)
    result: dict[str, Any] = {
        "model": args.model,
        "adapter": adapter,
        "image": image,
        "model_load": None,
        "prediction": None,
        "barcode": [],
        "barcode_valid": False,
        "valid": False,
    }
    started = time.perf_counter()
    try:
        startup_ms = worker.start()
        load_started = time.perf_counter()
        loaded = worker.request({"operation": "load"})
        result["model_load"] = {
            "status": "SUCCESS",
            "startup_ms": startup_ms,
            "model_load_ms": (time.perf_counter() - load_started) * 1000,
            "metadata": loaded.get("metadata", {}),
        }
        prediction = worker.request({"operation": "predict", "image": str(image_path)})
        result["prediction"] = prediction.get("prediction", {})
        result["valid"] = result["prediction"].get("status") == "SUCCESS"
    except Exception as exc:
        result["error"] = str(exc)
    finally:
        worker.stop()

    for decoder in (OpenCVBarcodeDecoder(), ZXingCppDecoder()):
        decoded = decoder.decode(image_path)
        result["barcode"].append(decoded.model_dump(mode="json"))

    result["barcode_valid"] = all(item["status"] in {"SUCCESS", "NO_DECODE"} for item in result["barcode"])
    result["elapsed_ms"] = (time.perf_counter() - started) * 1000
    result["valid"] = bool(result["valid"] and result["barcode_valid"])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
