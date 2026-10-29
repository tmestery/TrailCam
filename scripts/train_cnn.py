#!/usr/bin/env python3
"""Train the CNN on crop pixels and evaluate on the held-out test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from trailcam.data.dataset import CropImageDataset, load_label_names
from trailcam.eval.metrics import compute_metrics
from trailcam.models.cnn import (
    build_cnn,
    build_transforms,
    get_device,
    predict_loader,
    train_cnn,
)
from trailcam.utils.paths import repo_root


def load_config(config_path: Path) -> dict:
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve(root: Path, maybe_relative: str) -> Path:
    path = Path(maybe_relative)
    return path if path.is_absolute() else root / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TrailCam CNN.")
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root() / "configs" / "default.yaml",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = repo_root()
    seed = int(cfg.get("seed", 42))
    cnn_cfg = cfg.get("cnn", {})

    torch.manual_seed(seed)
    np.random.seed(seed)

    interim_dir = _resolve(root, cfg["paths"]["interim"])
    processed_dir = _resolve(root, cfg["paths"]["processed"])
    results_dir = _resolve(root, cfg["paths"]["results"])
    models_dir = results_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    image_size = int(cnn_cfg.get("image_size", 224))
    batch_size = int(cnn_cfg.get("batch_size", 32))
    epochs = int(cnn_cfg.get("epochs", 25))
    lr = float(cnn_cfg.get("lr", 1e-3))
    patience = int(cnn_cfg.get("patience", 5))
    num_workers = int(cnn_cfg.get("num_workers", 0))
    pretrained = bool(cnn_cfg.get("pretrained", True))

    label_names = load_label_names(processed_dir)
    label_to_id = {name: i for i, name in enumerate(label_names)}

    train_ds = CropImageDataset(
        processed_dir / "train.csv",
        interim_dir,
        label_to_id,
        transform=build_transforms(image_size, train=True),
    )
    val_ds = CropImageDataset(
        processed_dir / "val.csv",
        interim_dir,
        label_to_id,
        transform=build_transforms(image_size, train=False),
    )
    test_ds = CropImageDataset(
        processed_dir / "test.csv",
        interim_dir,
        label_to_id,
        transform=build_transforms(image_size, train=False),
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    device = get_device()
    print(f"Device: {device}")
    print(f"Classes: {len(label_names)} | train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    model = build_cnn(num_classes=len(label_names), pretrained=pretrained)
    checkpoint_path = models_dir / "cnn_resnet18.pt"
    train_info = train_cnn(
        model,
        train_loader,
        val_loader,
        y_train=np.asarray(train_ds.labels),
        device=device,
        epochs=epochs,
        lr=lr,
        patience=patience,
        checkpoint_path=checkpoint_path,
    )

    y_true, y_pred = predict_loader(model, test_loader, device)
    test_metrics = compute_metrics(y_true, y_pred)

    report = {
        "seed": seed,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "label_names": label_names,
        "hyperparameters": {
            "architecture": "resnet18",
            "pretrained": pretrained,
            "image_size": image_size,
            "batch_size": batch_size,
            "lr": lr,
            "epochs": epochs,
            "patience": patience,
            "best_val_macro_f1": train_info["best_val_macro_f1"],
            "epochs_ran": train_info["epochs_ran"],
            "device": train_info["device"],
        },
        "models": {
            "CNN": {
                "accuracy": test_metrics["accuracy"],
                "macro_f1": test_metrics["macro_f1"],
            }
        },
    }

    out_path = results_dir / "cnn_metrics.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print()
    print(f"CNN test accuracy={test_metrics['accuracy']:.4f} macro_f1={test_metrics['macro_f1']:.4f}")
    print(f"Wrote {out_path}")
    print(f"Saved checkpoint {checkpoint_path}")


if __name__ == "__main__":
    main()
