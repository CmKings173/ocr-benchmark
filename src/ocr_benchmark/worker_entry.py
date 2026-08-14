import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Optional

from ocr_benchmark.core.registry import create_adapter


def _error_status(exc: Exception) -> str:
    message = str(exc).upper()
    if "INVALID_OUTPUT" in message or "VALIDATION ERROR" in message:
        return "INVALID_OUTPUT"
    if "TIMEOUT" in message:
        return "TIMEOUT"
    if "OUT OF MEMORY" in message or "OOM" in message:
        return "OOM"
    if "DEPENDENCY" in message or isinstance(exc, ImportError):
        return "DEPENDENCY_ERROR"
    if "NOT_CONFIGURED" in message:
        return "NOT_CONFIGURED"
    if "NOT_SUPPORTED" in message:
        return "NOT_SUPPORTED"
    return "EXCEPTION"


def _handle(adapter: Any, loaded: bool, request: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    operation = request.get("operation")
    if operation == "load":
        with redirect_stdout(sys.stderr):
            adapter.load()
            metadata = adapter.metadata()
        return {"ok": True, "metadata": metadata}, True
    if operation == "warmup":
        image = request.get("image")
        with redirect_stdout(sys.stderr):
            adapter.warmup(int(request.get("iterations", 1)), Path(image) if image else None)
        return {"ok": True}, loaded
    if operation == "predict":
        if not loaded:
            raise RuntimeError("worker is not loaded")
        with redirect_stdout(sys.stderr):
            prediction = adapter.predict(Path(request["image"]))
        return {"ok": True, "prediction": prediction.model_dump(mode="json")}, loaded
    if operation == "predict_batch":
        if not loaded:
            raise RuntimeError("worker is not loaded")
        with redirect_stdout(sys.stderr):
            predictions = adapter.predict_batch([Path(image) for image in request.get("images", [])])
        return {"ok": True, "predictions": [item.model_dump(mode="json") for item in predictions]}, loaded
    if operation == "unload":
        with redirect_stdout(sys.stderr):
            adapter.unload()
        return {"ok": True}, loaded
    raise ValueError(f"unknown operation: {operation}")


def main() -> int:
    model_name = os.environ.get("OCR_BENCH_MODEL", "mock")
    adapter_config = json.loads(os.environ.get("OCR_BENCH_MODEL_CONFIG", "{}"))
    adapter_error: Optional[Exception] = None
    try:
        adapter = create_adapter(model_name, adapter_config)
    except Exception as exc:
        adapter = None
        adapter_error = exc
    loaded = False
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if adapter_error is not None:
                raise adapter_error
            if adapter is None:
                raise RuntimeError("adapter is unavailable")
            response, loaded = _handle(adapter, loaded, request)
            print(json.dumps(response), flush=True)
            if request.get("operation") == "unload":
                return 0
        except Exception as exc:
            response = {"ok": False, "status": _error_status(exc), "error": str(exc)}
            print(json.dumps(response), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
