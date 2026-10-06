"""Image-level train/validation/test splits for cropped animal images."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def split_dataset(
    interim_dir: str | Path,
    processed_dir: str | Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, Path]:
    """Split crops by original image so near-duplicates stay in one fold.

    Writes train.csv / val.csv / test.csv under ``processed_dir`` with the same
    columns as the interim crops manifest, plus a ``split`` column.
    """
    interim_dir = Path(interim_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = interim_dir / "crops_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing crop manifest: {manifest_path}. Run crop first.")

    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    out_paths = {
        "train": processed_dir / "train.csv",
        "val": processed_dir / "val.csv",
        "test": processed_dir / "test.csv",
    }
    if all(p.exists() for p in out_paths.values()):
        print(f"Splits already present in {processed_dir}; skipping split.")
        return out_paths

    with manifest_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    by_image: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_image[row["image_id"]].append(row)

    image_ids = sorted(by_image.keys(), key=lambda x: int(x))
    rng = np.random.default_rng(seed)
    rng.shuffle(image_ids)

    n = len(image_ids)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    # Assign remainder to test so counts always sum to n.
    n_test = n - n_train - n_val
    if n_test < 0:
        n_val += n_test
        n_test = 0

    split_of_image: dict[str, str] = {}
    for i, image_id in enumerate(image_ids):
        if i < n_train:
            split_of_image[image_id] = "train"
        elif i < n_train + n_val:
            split_of_image[image_id] = "val"
        else:
            split_of_image[image_id] = "test"

    fieldnames = list(rows[0].keys()) + ["split"]
    writers = {}
    files = {}
    counts = {"train": 0, "val": 0, "test": 0}
    try:
        for split_name, path in out_paths.items():
            files[split_name] = path.open("w", encoding="utf-8", newline="")
            writers[split_name] = csv.DictWriter(files[split_name], fieldnames=fieldnames)
            writers[split_name].writeheader()

        for image_id, image_rows in by_image.items():
            split_name = split_of_image[image_id]
            for row in image_rows:
                out = dict(row)
                out["split"] = split_name
                writers[split_name].writerow(out)
                counts[split_name] += 1
    finally:
        for handle in files.values():
            handle.close()

    summary_path = processed_dir / "split_summary.json"
    import json

    summary = {
        "seed": seed,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "n_images": {"train": n_train, "val": n_val, "test": n_test, "total": n},
        "n_crops": counts,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        "Split by image_id: "
        f"{n_train} train / {n_val} val / {n_test} test images; "
        f"{counts['train']} / {counts['val']} / {counts['test']} crops"
    )
    return out_paths
