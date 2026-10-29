"""Crop image datasets for CNN and vision-language evaluation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def load_label_names(processed_dir: Path) -> list[str]:
    """Sorted species names from train/val/test manifests (shared label space)."""
    species: set[str] = set()
    for split in ("train", "val", "test"):
        csv_path = processed_dir / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing {csv_path}. Run prepare_data first.")
        species.update(pd.read_csv(csv_path)["species"].tolist())
    return sorted(species)


class CropImageDataset(Dataset):
    """Load animal crops and integer species labels from a split CSV."""

    def __init__(
        self,
        csv_path: Path,
        interim_dir: Path,
        label_to_id: dict[str, int],
        transform=None,
    ) -> None:
        self.df = pd.read_csv(csv_path)
        self.interim_dir = Path(interim_dir)
        self.label_to_id = label_to_id
        self.transform = transform
        self.paths = [self.interim_dir / p for p in self.df["crop_path"].tolist()]
        self.labels = [label_to_id[s] for s in self.df["species"].tolist()]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        with Image.open(path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label
