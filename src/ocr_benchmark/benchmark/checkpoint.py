from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional


class CheckpointStore:
    """Append-only checkpoint store used to resume deterministic benchmark passes."""

    def __init__(self, path: Optional[Path] = None):
        self.path = path
        self._records: dict[str, dict[str, Any]] = {}
        if path is not None and path.is_file():
            self._load(path)

    def _load(self, path: Path) -> None:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                key = str(payload["key"])
                self._records[key] = payload
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid checkpoint {path}:{line_number}: {exc}") from exc

    def get(self, key: str) -> Optional[dict[str, Any]]:
        return self._records.get(key)

    def append(self, key: str, record: dict[str, Any]) -> None:
        payload = {"key": key, "record": record}
        self._records[key] = payload
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def records(self) -> list[dict[str, Any]]:
        return [payload["record"] for payload in self._records.values()]
