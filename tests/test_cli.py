import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


def test_run_all_command_emits_artifacts(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (8, 8), "white").save(image_root / "one.png")
    gt = tmp_path / "ground_truth.json"
    gt.write_text(json.dumps({"samples": [{"image": "one.png", "fields": {"sku": "ABC"}}]}))
    config = tmp_path / "benchmark.yaml"
    config.write_text("benchmark:\n  performance_min_iterations: 1\n  performance_repetitions: 1\n  batch_sizes: [1]\n  concurrency: [1]\n  production_gates: {}\n")
    output = tmp_path / "results"
    result = subprocess.run([sys.executable, "scripts/run_all.py", "--models", "mock", "--dataset", str(image_root), "--ground-truth", str(gt), "--config", str(config), "--output", str(output)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (output / "summary.json").is_file()
    assert (output / "leaderboard.csv").is_file()


def test_multi_model_output_is_isolated(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (8, 8), "white").save(image_root / "one.png")
    gt = tmp_path / "ground_truth.json"
    gt.write_text(json.dumps({"samples": [{"image": "one.png", "fields": {"sku": "ABC"}}]}))
    config = tmp_path / "benchmark.yaml"
    config.write_text("benchmark:\n  performance_min_iterations: 1\n  performance_repetitions: 1\n  batch_sizes: [1]\n  concurrency: [1]\n  production_gates: {}\n")
    output = tmp_path / "results"
    result = subprocess.run([sys.executable, "scripts/run_all.py", "--models", "mock,unknown_model", "--dataset", str(image_root), "--ground-truth", str(gt), "--config", str(config), "--output", str(output)], capture_output=True, text=True, check=False)
    assert result.returncode == 1, result.stdout + result.stderr
    assert (output / "mock" / "summary.json").is_file()
    assert (output / "unknown_model" / "summary.json").is_file()
