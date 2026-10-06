"""Crop bounding boxes from camera-trap images into labeled animal patches."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower()).strip("_")
    return slug or "unknown"


def crop_annotations(raw_dir: str | Path, interim_dir: str | Path) -> Path:
    """Crop COCO xywh boxes from raw images into labeled crop files.

    Writes:
      - crops/{species_slug}/{image_id}_{box_idx}.jpg
      - crops_manifest.csv  (crop_path, image_id, box_idx, species, category_id, bbox)
    """
    raw_dir = Path(raw_dir)
    interim_dir = Path(interim_dir)
    crops_dir = interim_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    annotations_path = raw_dir / "annotations.jsonl"
    categories_path = raw_dir / "categories.json"
    manifest_path = interim_dir / "crops_manifest.csv"

    if not annotations_path.exists():
        raise FileNotFoundError(f"Missing annotations: {annotations_path}. Run download first.")
    if not categories_path.exists():
        raise FileNotFoundError(f"Missing categories: {categories_path}. Run download first.")

    if manifest_path.exists():
        print(f"Crops already present in {interim_dir}; skipping crop.")
        return manifest_path

    categories = {int(k): v for k, v in json.loads(categories_path.read_text(encoding="utf-8")).items()}

    fieldnames = [
        "crop_path",
        "image_id",
        "box_idx",
        "species",
        "category_id",
        "x",
        "y",
        "w",
        "h",
    ]
    n_crops = 0
    n_skipped = 0

    with annotations_path.open(encoding="utf-8") as ann_file, manifest_path.open(
        "w", encoding="utf-8", newline=""
    ) as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
        writer.writeheader()

        lines = ann_file.readlines()
        for line in tqdm(lines, desc="Cropping boxes", unit="img"):
            record = json.loads(line)
            image_id = int(record["image_id"])
            image_path = raw_dir / record["image_path"]
            if not image_path.exists():
                n_skipped += len(record["bboxes"])
                continue

            with Image.open(image_path) as img:
                img = img.convert("RGB")
                width, height = img.size

                for box_idx, (bbox, category_id) in enumerate(
                    zip(record["bboxes"], record["category_ids"], strict=True)
                ):
                    x, y, w, h = (float(v) for v in bbox)
                    if w <= 1 or h <= 1:
                        n_skipped += 1
                        continue

                    left = max(0, int(x))
                    top = max(0, int(y))
                    right = min(width, int(x + w))
                    bottom = min(height, int(y + h))
                    if right <= left or bottom <= top:
                        n_skipped += 1
                        continue

                    category_id = int(category_id)
                    species = categories.get(category_id, f"class_{category_id}")
                    # Skip empty Human class if present with no useful boxes.
                    if species.lower() == "human":
                        n_skipped += 1
                        continue

                    species_dir = crops_dir / _slugify(species)
                    species_dir.mkdir(parents=True, exist_ok=True)
                    crop_name = f"{image_id}_{box_idx}.jpg"
                    crop_path = species_dir / crop_name
                    crop = img.crop((left, top, right, bottom))
                    crop.save(crop_path, format="JPEG", quality=95)

                    writer.writerow(
                        {
                            "crop_path": str(crop_path.relative_to(interim_dir)),
                            "image_id": image_id,
                            "box_idx": box_idx,
                            "species": species,
                            "category_id": category_id,
                            "x": left,
                            "y": top,
                            "w": right - left,
                            "h": bottom - top,
                        }
                    )
                    n_crops += 1

    print(f"Wrote {n_crops} crops to {crops_dir} (skipped {n_skipped})")
    print(f"Manifest: {manifest_path}")
    return manifest_path
