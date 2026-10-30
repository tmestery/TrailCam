"""ENA24 full-frame scenes with annotated animal boxes for accurate demos."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_scene_items(
    raw_dir: Path,
    manifest_path: Path,
    *,
    limit: int = 300,
    species: list[str] | None = None,
    seed: int = 42,
    min_box_side: int = 40,
    split: str = "test",
    processed_dir: Path | None = None,
) -> list[dict]:
    """Load full-image paths + xyxy boxes from the crop manifest.

    Each item is one annotated animal on its source camera-trap image — the same
    crop style the CNN was trained on, but shown in the full scene.

    ``split`` defaults to ``test`` so the demo does not replay training images
    (which often show near-1.0 confidence from memorization). Use ``all`` to
    include every split.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Run: python scripts/prepare_data.py"
        )
    raw_dir = Path(raw_dir)
    images_dir = raw_dir / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing {images_dir}. Run prepare_data first.")

    df = pd.read_csv(manifest_path)
    if species:
        wanted = {s.lower() for s in species}
        df = df[df["species"].str.lower().isin(wanted)]

    if split != "all":
        processed = Path(processed_dir) if processed_dir else raw_dir.parent / "processed"
        split_csv = processed / f"{split}.csv"
        if not split_csv.exists():
            raise FileNotFoundError(
                f"Missing {split_csv} for split={split!r}. Run prepare_data or use --split all."
            )
        split_ids = set(pd.read_csv(split_csv)["image_id"].astype(int))
        df = df[df["image_id"].astype(int).isin(split_ids)]

    # Prefer larger, clearer animals for a nicer demo.
    df = df[(df["w"] >= min_box_side) & (df["h"] >= min_box_side)].copy()
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if limit > 0:
        df = df.head(limit)

    items: list[dict] = []
    for row in df.itertuples(index=False):
        image_path = images_dir / f"{int(row.image_id)}.jpg"
        if not image_path.exists():
            # Some exports use zero-padded names; try a couple fallbacks.
            alt = images_dir / f"{int(row.image_id):05d}.jpg"
            image_path = alt if alt.exists() else image_path
        if not image_path.exists():
            continue
        x, y, bw, bh = float(row.x), float(row.y), float(row.w), float(row.h)
        x1, y1 = int(max(0, np.floor(x))), int(max(0, np.floor(y)))
        x2, y2 = int(np.ceil(x + bw)), int(np.ceil(y + bh))
        items.append(
            {
                "image_path": image_path,
                "box": (x1, y1, x2, y2),
                "species": str(row.species),
            }
        )
    if not items:
        raise FileNotFoundError("No usable ENA24 scenes found (check raw images + manifest).")
    return items
