from __future__ import annotations

import csv
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Optional


class ResourceMonitor:
    def __init__(self, pid: Optional[int] = None, interval_ms: int = 100):
        self.pid = pid
        self.interval_seconds = max(interval_ms, 10) / 1000
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._nvidia_smi_available: Optional[bool] = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="resource-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> list[dict[str, Any]]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 4))
        return list(self.samples)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append(self.sample())
            self._stop.wait(self.interval_seconds)

    def sample(self) -> dict[str, Any]:
        sample: dict[str, Any] = {"timestamp": time.time(), "rss_bytes": None, "cpu_percent": None, "unified_memory_bytes": None, "gpu_utilization_percent": None, "gpu_memory_bytes": None, "gpu_power_watts": None, "gpu_temperature_c": None}
        try:
            import psutil
            process = psutil.Process(self.pid) if self.pid else psutil.Process()
            processes = [process] + process.children(recursive=True)
            sample["rss_bytes"] = sum(item.memory_info().rss for item in processes if item.is_running())
            sample["cpu_percent"] = sum(item.cpu_percent(interval=None) for item in processes if item.is_running())
        except Exception:
            pass
        try:
            if self._nvidia_smi_available is None:
                self._nvidia_smi_available = shutil.which("nvidia-smi") is not None
            if not self._nvidia_smi_available:
                return sample
            query = "utilization.gpu,memory.used,power.draw,temperature.gpu"
            result = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=1, check=False)
            if result.returncode == 0 and result.stdout.strip():
                values = next(csv.reader([result.stdout.strip()]))
                sample["gpu_utilization_percent"] = float(values[0].strip())
                sample["gpu_memory_bytes"] = int(float(values[1].strip()) * 1024 * 1024)
                sample["gpu_power_watts"] = float(values[2].strip())
                sample["gpu_temperature_c"] = float(values[3].strip())
        except Exception:
            pass
        return sample

    @staticmethod
    def summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
        def peak(key: str) -> Any:
            values = [item[key] for item in samples if item.get(key) is not None]
            return max(values) if values else None

        def mean(key: str) -> Any:
            values = [item[key] for item in samples if item.get(key) is not None]
            return sum(values) / len(values) if values else None

        return {
            "sample_count": len(samples),
            "peak_rss_bytes": peak("rss_bytes"),
            "mean_cpu_percent": mean("cpu_percent"),
            "peak_cpu_percent": peak("cpu_percent"),
            "mean_gpu_utilization_percent": mean("gpu_utilization_percent"),
            "peak_gpu_utilization_percent": peak("gpu_utilization_percent"),
            "peak_unified_memory_bytes": peak("unified_memory_bytes"),
            "peak_gpu_memory_bytes": peak("gpu_memory_bytes"),
            "peak_gpu_power_watts": peak("gpu_power_watts"),
            "peak_gpu_temperature_c": peak("gpu_temperature_c"),
        }
