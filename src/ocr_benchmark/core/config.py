from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class BenchmarkConfig(BaseModel):
    warmup_iterations: int = 5
    accuracy_iterations_per_image: int = 1
    performance_min_iterations: int = 100
    performance_repetitions: int = 3
    batch_sizes: list[int] = Field(default_factory=lambda: [1, 4, 8])
    concurrency: list[int] = Field(default_factory=lambda: [1, 2, 4, 8])
    timeout_seconds: float = 60.0
    output_dir: Path = Path("results")
    production_gates: dict[str, Any] = Field(default_factory=dict)
    gate_policy: dict[str, Any] = Field(default_factory=dict)
    score_weights: dict[str, float] = Field(default_factory=dict)
    allowed_tags: list[str] = Field(default_factory=list)


def load_config(path: Path) -> BenchmarkConfig:
    raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    return BenchmarkConfig.model_validate(raw.get("benchmark", raw))


def load_field_schema(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = yaml.safe_load(path.read_text()) or {}
    return payload if isinstance(payload, dict) else {}
