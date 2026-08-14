from __future__ import annotations

import json
import os
import selectors
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional


class WorkerProtocolError(RuntimeError):
    pass


class SubprocessWorker:
    def __init__(self, timeout_seconds: float = 60.0, model_name: str = "mock", model_config: Optional[dict[str, Any]] = None):
        self.timeout_seconds = timeout_seconds
        self.model_name = model_name
        self.model_config = model_config or {}
        self.process: Optional[subprocess.Popen] = None

    def start(self) -> float:
        started = time.perf_counter()
        src_root = str(Path(__file__).resolve().parents[2])
        environment = os.environ.copy()
        environment["PYTHONPATH"] = src_root + os.pathsep + environment.get("PYTHONPATH", "")
        environment["OCR_BENCH_MODEL"] = self.model_name
        environment["OCR_BENCH_MODEL_CONFIG"] = json.dumps(self.model_config)
        self.process = subprocess.Popen(
            [sys.executable, "-m", "ocr_benchmark.worker_entry"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=environment,
        )
        return (time.perf_counter() - started) * 1000

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.process is None or self.process.stdin is None or self.process.stdout is None:
            raise WorkerProtocolError("worker is not running")
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()
        selector = selectors.DefaultSelector()
        selector.register(self.process.stdout, selectors.EVENT_READ)
        ready = selector.select(timeout=self.timeout_seconds)
        selector.close()
        if not ready:
            self._kill_process()
            raise TimeoutError(f"worker request timed out after {self.timeout_seconds}s")
        line = self.process.stdout.readline()
        if not line:
            stderr = self.process.stderr.read() if self.process.stderr else ""
            self._kill_process()
            raise WorkerProtocolError(f"worker exited without response: {stderr}")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            self._kill_process()
            raise WorkerProtocolError(f"worker returned invalid JSON: {line[:200]!r}") from exc
        if not response.get("ok", False):
            raise WorkerProtocolError(f"{response.get('status', 'EXCEPTION')}: {response.get('error', response)}")
        return response

    def restart(self) -> float:
        self.stop()
        return self.start()

    def _kill_process(self) -> None:
        process = self.process
        if process is None:
            return
        try:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=2)
        finally:
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass
            self.process = None

    def stop(self) -> None:
        if self.process is None:
            return
        process = self.process
        try:
            if process.poll() is None:
                try:
                    self.request({"operation": "unload"})
                except Exception:
                    pass
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=2)
        except Exception:
            if process.poll() is None:
                process.kill()
        finally:
            if process.poll() is None:
                process.kill()
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass
            self.process = None
