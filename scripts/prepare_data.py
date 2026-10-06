#!/usr/bin/env python3
"""Prepare ENA24 data: download, crop bounding boxes, and split."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from trailcam.data.crop import crop_annotations
from trailcam.data.download import download_ena24
from trailcam.data.split import split_dataset
from trailcam.utils.paths import repo_root


def load_config(config_path: Path) -> dict:
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ENA24 crops and splits.")
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root() / "configs" / "default.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download if raw data already exists",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = repo_root()
    raw_dir = root / cfg["paths"]["raw"]
    interim_dir = root / cfg["paths"]["interim"]
    processed_dir = root / cfg["paths"]["processed"]
    seed = int(cfg.get("seed", 42))
    split_cfg = cfg.get("split", {})
    dataset_id = cfg.get("dataset", {}).get("hf_id", "davanstrien/ena24-detection")

    raw_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        download_ena24(raw_dir, dataset_id=dataset_id)
    elif not (raw_dir / "annotations.jsonl").exists():
        raise SystemExit(f"--skip-download set but missing {raw_dir / 'annotations.jsonl'}")

    crop_annotations(raw_dir, interim_dir)
    split_dataset(
        interim_dir,
        processed_dir,
        train_ratio=float(split_cfg.get("train", 0.70)),
        val_ratio=float(split_cfg.get("val", 0.15)),
        test_ratio=float(split_cfg.get("test", 0.15)),
        seed=seed,
    )
    print("Data preparation complete.")


if __name__ == "__main__":
    main()
