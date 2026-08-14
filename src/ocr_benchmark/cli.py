import argparse
from pathlib import Path

from ocr_benchmark.data.validator import DatasetValidationError, load_and_validate_dataset


def main() -> int:
    parser = argparse.ArgumentParser(prog="ocr-bench")
    parser.add_argument("--dataset", type=Path, default=Path("data"))
    parser.add_argument("--ground-truth", type=Path, default=Path("data/ground_truth.json"))
    args = parser.parse_args()
    try:
        dataset = load_and_validate_dataset(args.dataset, args.ground_truth)
    except DatasetValidationError as exc:
        parser.error(str(exc))
    print(f"dataset valid: {len(dataset.samples)} samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
