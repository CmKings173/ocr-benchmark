from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def write_leaderboard(rows: list[dict[str, Any]], output_path: Path) -> None:
    def rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
        score = row.get("score")
        score_value = float(score) if isinstance(score, (int, float)) else -1.0
        return (not bool(row.get("eligible")), -score_value, -(row.get("full_label_accuracy") or -1), -(row.get("field_exact_accuracy") or -1))

    ranked = sorted(rows, key=rank_key)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["rank", "model", "eligible", "full_label_accuracy", "critical_field_accuracy", "field_exact_accuracy", "p95_ms", "failure_rate", "score"]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(ranked, 1):
            writer.writerow({field: row.get(field, "N/A") for field in fields} | {"rank": rank})
