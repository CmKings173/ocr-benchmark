import hashlib
import json
from pathlib import Path
from typing import Iterable, Optional

from PIL import Image

from ocr_benchmark.core.schemas import Dataset


class DatasetValidationError(ValueError):
    pass


ALLOWED_TAGS = {
    "clear",
    "blur",
    "rotated",
    "reflection",
    "dark",
    "small_text",
    "damaged",
    "long_distance",
    "vietnamese",
    "vietnamese_mixed",
    "mixed",
    "chinese_simplified",
    "chinese_traditional",
    "japanese",
}


def _safe_image_path(dataset_dir: Path, image_ref: str) -> Path:
    root = dataset_dir.expanduser().resolve()
    relative = Path(image_ref)
    if not image_ref or relative.is_absolute():
        raise DatasetValidationError(f"image path must be relative to dataset root: {image_ref!r}")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise DatasetValidationError(f"image path escapes dataset root: {image_ref!r}") from exc
    return candidate


def load_and_validate_dataset(
    dataset_dir: Path,
    ground_truth_path: Path,
    allowed_tags: Optional[Iterable[str]] = None,
    max_image_bytes: int = 50 * 1024 * 1024,
    max_image_pixels: int = 100_000_000,
) -> Dataset:
    try:
        payload = json.loads(ground_truth_path.read_text())
        if isinstance(payload, dict):
            raw_samples = payload.get("samples")
        elif isinstance(payload, list):
            raw_samples = payload
        else:
            raw_samples = None
        dataset = Dataset.model_validate({"samples": raw_samples})
    except Exception as exc:
        raise DatasetValidationError(f"invalid ground truth: {exc}") from exc

    names = [sample.image for sample in dataset.samples]
    if len(names) != len(set(names)):
        raise DatasetValidationError("duplicate image entries")
    if not dataset.samples:
        raise DatasetValidationError("dataset is empty")

    hashes: dict[str, str] = {}
    valid_tags = set(allowed_tags) if allowed_tags is not None else ALLOWED_TAGS
    for sample in dataset.samples:
        if sample.label_count != 1:
            raise DatasetValidationError(
                f"V1 requires exactly one physical label per image: {sample.image} has {sample.label_count}"
            )
        image_path = _safe_image_path(dataset_dir, sample.image)
        if not image_path.is_file():
            raise DatasetValidationError(f"image not found: {sample.image}")
        if image_path.stat().st_size > max_image_bytes:
            raise DatasetValidationError(f"image too large: {sample.image}")
        try:
            with Image.open(image_path) as image:
                image.verify()
                if image.width * image.height > max_image_pixels:
                    raise DatasetValidationError(f"image dimensions exceed limit: {sample.image}")
        except Exception as exc:
            if isinstance(exc, DatasetValidationError):
                raise
            raise DatasetValidationError(f"unreadable image {sample.image}: {exc}") from exc
        if not sample.fields:
            raise DatasetValidationError(f"fields missing for {sample.image}")
        unknown_tags = set(sample.tags) - valid_tags
        if unknown_tags:
            raise DatasetValidationError(f"invalid tags for {sample.image}: {sorted(unknown_tags)}")
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if digest in hashes:
            raise DatasetValidationError(f"duplicate image content: {sample.image} and {hashes[digest]}")
        hashes[digest] = sample.image
    return dataset
