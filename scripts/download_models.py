import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Download one pinned Hugging Face model revision into the configured cache.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("model_cache"))
    args = parser.parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("DEPENDENCY_ERROR: install huggingface_hub")
        return 1
    path = snapshot_download(repo_id=args.repo, revision=args.revision, cache_dir=str(args.cache_dir))
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
