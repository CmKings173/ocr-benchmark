import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocr_benchmark.data.validator import DatasetValidationError, load_and_validate_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/images"))
    parser.add_argument("--ground-truth", type=Path, default=Path("data/ground_truth.json"))
    args = parser.parse_args()
    try:
        dataset = load_and_validate_dataset(args.dataset, args.ground_truth)
    except DatasetValidationError as exc:
        print(f"FAIL: {exc}")
        return 1
    print(f"OK: {len(dataset.samples)} samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
