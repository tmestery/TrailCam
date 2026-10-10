"""Color and texture descriptors for classical ML baselines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import local_binary_pattern
from tqdm import tqdm


def _hsv_histogram(rgb: np.ndarray, bins: tuple[int, int, int] = (16, 8, 8)) -> np.ndarray:
    hsv = rgb2hsv(rgb)
    hist, _ = np.histogramdd(
        hsv.reshape(-1, 3),
        bins=bins,
        range=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
    )
    hist = hist.astype(np.float64).ravel()
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def _lbp_histogram(rgb: np.ndarray, p: int = 8, r: float = 1.0) -> np.ndarray:
    # Integer dtype avoids skimage numerical-noise warnings on float images.
    gray = (rgb2gray(rgb) * 255.0).astype(np.uint8)
    lbp = local_binary_pattern(gray, P=p, R=r, method="uniform")
    n_bins = p + 2  # uniform LBP pattern count
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=False)
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def extract_features_from_image(
    image_path: str | Path,
    hsv_bins: tuple[int, int, int] = (16, 8, 8),
    lbp_p: int = 8,
    lbp_r: float = 1.0,
) -> np.ndarray:
    """Extract concatenated HSV + LBP features from a single crop image."""
    with Image.open(image_path) as img:
        rgb = np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0
    return np.concatenate(
        [
            _hsv_histogram(rgb, bins=hsv_bins),
            _lbp_histogram(rgb, p=lbp_p, r=lbp_r),
        ]
    )


def extract_classical_features(
    image_paths: list[str] | list[Path],
    hsv_bins: tuple[int, int, int] = (16, 8, 8),
    lbp_p: int = 8,
    lbp_r: float = 1.0,
    show_progress: bool = True,
) -> np.ndarray:
    """Extract engineered color/texture features from crop images."""
    iterator = image_paths
    if show_progress:
        iterator = tqdm(image_paths, desc="Extracting features", unit="img")
    rows = [
        extract_features_from_image(path, hsv_bins=hsv_bins, lbp_p=lbp_p, lbp_r=lbp_r)
        for path in iterator
    ]
    return np.vstack(rows)


def load_split_csv(csv_path: Path, interim_dir: Path) -> tuple[list[Path], np.ndarray]:
    """Load crop paths and species labels from a split CSV."""
    df = pd.read_csv(csv_path)
    paths = [interim_dir / row for row in df["crop_path"].tolist()]
    labels = df["species"].to_numpy()
    return paths, labels


def build_or_load_feature_cache(
    processed_dir: Path,
    interim_dir: Path,
    cache_path: Path,
    hsv_bins: tuple[int, int, int] = (16, 8, 8),
    lbp_p: int = 8,
    lbp_r: float = 1.0,
    force: bool = False,
) -> dict[str, np.ndarray | list[str]]:
    """Build HSV+LBP features for train/val/test, or load from NPZ cache."""
    if cache_path.exists() and not force:
        data = np.load(cache_path, allow_pickle=True)
        return {
            "X_train": data["X_train"],
            "X_val": data["X_val"],
            "X_test": data["X_test"],
            "y_train": data["y_train"],
            "y_val": data["y_val"],
            "y_test": data["y_test"],
            "label_names": data["label_names"].tolist(),
        }

    train_paths, train_species = load_split_csv(processed_dir / "train.csv", interim_dir)
    val_paths, val_species = load_split_csv(processed_dir / "val.csv", interim_dir)
    test_paths, test_species = load_split_csv(processed_dir / "test.csv", interim_dir)

    label_names = sorted(set(train_species) | set(val_species) | set(test_species))
    label_to_id = {name: i for i, name in enumerate(label_names)}

    def encode(species: np.ndarray) -> np.ndarray:
        return np.array([label_to_id[s] for s in species], dtype=np.int64)

    X_train = extract_classical_features(train_paths, hsv_bins=hsv_bins, lbp_p=lbp_p, lbp_r=lbp_r)
    X_val = extract_classical_features(val_paths, hsv_bins=hsv_bins, lbp_p=lbp_p, lbp_r=lbp_r)
    X_test = extract_classical_features(test_paths, hsv_bins=hsv_bins, lbp_p=lbp_p, lbp_r=lbp_r)

    y_train = encode(train_species)
    y_val = encode(val_species)
    y_test = encode(test_species)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        label_names=np.array(label_names, dtype=object),
    )
    print(f"Wrote feature cache to {cache_path}")

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "label_names": label_names,
    }
