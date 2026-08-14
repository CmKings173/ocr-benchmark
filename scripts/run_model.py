import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_all import main as run_all_main


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, default=Path("data/images"))
    parser.add_argument("--ground-truth", type=Path, default=Path("data/ground_truth.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark.yaml"))
    parser.add_argument("--models-config", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    sys.argv = ["run_all.py", "--models", args.model, "--dataset", str(args.dataset), "--ground-truth", str(args.ground_truth), "--config", str(args.config), "--models-config", str(args.models_config)]
    if args.output:
        sys.argv.extend(["--output", str(args.output)])
    return run_all_main()


if __name__ == "__main__":
    raise SystemExit(main())
