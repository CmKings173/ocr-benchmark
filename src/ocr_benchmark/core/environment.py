from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any


def capture_environment() -> dict[str, Any]:
    snapshot: dict[str, Any] = {"timestamp": datetime.now(timezone.utc).isoformat(), "os": platform.platform(), "kernel": platform.release(), "python": sys.version, "architecture": platform.machine(), "cpu": platform.processor()}
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=2, check=False)
        snapshot["nvidia_smi"] = result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        snapshot["nvidia_smi"] = None
    versions = {}
    for package in ("pydantic", "Pillow", "PyYAML", "paddleocr", "torch", "vllm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    snapshot["packages"] = versions
    return snapshot
