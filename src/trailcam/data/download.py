"""Fetch the ENA24 camera-trap detection dataset."""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

# Proposal / HF docs use davanstrien; community mirror is the same release.
DEFAULT_DATASET_ID = "davanstrien/ena24-detection"
FALLBACK_DATASET_ID = "lila-bc-community/ena24-detection"


def download_ena24(output_dir: str | Path, dataset_id: str = DEFAULT_DATASET_ID) -> Path:
    """Download ENA24 images and annotation index into ``output_dir``.

    Writes:
      - images/{image_id}.jpg
      - annotations.jsonl  (one JSON object per image)
      - categories.json    (category id -> species name)
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    annotations_path = output_dir / "annotations.jsonl"
    categories_path = output_dir / "categories.json"

    if annotations_path.exists() and categories_path.exists():
        print(f"Raw data already present in {output_dir}; skipping download.")
        return output_dir

    try:
        ds = load_dataset(dataset_id, split="train")
    except Exception as exc:  # noqa: BLE001
        if dataset_id == FALLBACK_DATASET_ID:
            raise
        print(f"Failed to load {dataset_id} ({exc}); trying {FALLBACK_DATASET_ID}.")
        ds = load_dataset(FALLBACK_DATASET_ID, split="train")

    # HF schemas vary: Sequence(ClassLabel) vs dict with List(ClassLabel).
    objects_feature = ds.features["objects"]
    if hasattr(objects_feature, "feature"):
        category_feature = objects_feature.feature["category"]
    else:
        category_feature = objects_feature["category"].feature

    categories = {int(i): category_feature.int2str(i) for i in range(category_feature.num_classes)}
    categories_path.write_text(json.dumps(categories, indent=2), encoding="utf-8")
    print(f"Loaded {len(ds)} images; writing to {output_dir} ...", flush=True)

    with annotations_path.open("w", encoding="utf-8") as ann_file:
        for row in tqdm(ds, desc="Downloading ENA24", unit="img"):
            image_id = int(row["image_id"])
            image_path = images_dir / f"{image_id}.jpg"
            if not image_path.exists():
                row["image"].convert("RGB").save(image_path, format="JPEG", quality=95)

            objects = row["objects"]
            record = {
                "image_id": image_id,
                "width": int(row["width"]),
                "height": int(row["height"]),
                "image_path": str(image_path.relative_to(output_dir)),
                "bboxes": [[float(v) for v in bbox] for bbox in objects["bbox"]],
                "category_ids": [int(c) for c in objects["category"]],
                "areas": [float(a) for a in objects.get("area", [])],
            }
            ann_file.write(json.dumps(record) + "\n")

    print(f"Saved {len(ds)} images to {images_dir}", flush=True)
    print(f"Wrote annotations to {annotations_path}", flush=True)
    return output_dir
