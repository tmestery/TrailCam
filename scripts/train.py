#!/usr/bin/env python3
"""Train classical models and evaluate on the held-out test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python scripts/train.py` without relying on an editable install.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import joblib
import yaml

from trailcam.eval.metrics import compute_metrics
from trailcam.features.classical import build_or_load_feature_cache
from trailcam.models.classical import (
    train_logistic_regression,
    train_perceptron,
    train_svm,
)
from trailcam.utils.paths import repo_root


def load_config(config_path: Path) -> dict:
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve(root: Path, maybe_relative: str) -> Path:
    path = Path(maybe_relative)
    return path if path.is_absolute() else root / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train classical TrailCam models.")
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root() / "configs" / "default.yaml",
    )
    parser.add_argument(
        "--force-features",
        action="store_true",
        help="Recompute feature cache even if it exists",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = repo_root()
    seed = int(cfg.get("seed", 42))
    classical_cfg = cfg.get("classical", {})

    interim_dir = _resolve(root, cfg["paths"]["interim"])
    processed_dir = _resolve(root, cfg["paths"]["processed"])
    results_dir = _resolve(root, cfg["paths"]["results"])
    cache_path = _resolve(
        root,
        classical_cfg.get("feature_cache", "data/processed/classical_features.npz"),
    )

    hsv_bins = tuple(classical_cfg.get("hsv_bins", [16, 8, 8]))
    lbp_p = int(classical_cfg.get("lbp_p", 8))
    lbp_r = float(classical_cfg.get("lbp_r", 1))
    logistic_c = [float(c) for c in classical_cfg.get("logistic_c", [0.1, 1.0, 10.0])]
    svm_c = [float(c) for c in classical_cfg.get("svm_c", [0.1, 1.0, 10.0])]
    svm_gamma_raw = classical_cfg.get("svm_gamma", ["scale", 0.01])
    svm_gamma: list[float | str] = []
    for g in svm_gamma_raw:
        svm_gamma.append(g if isinstance(g, str) else float(g))

    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir = results_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    features = build_or_load_feature_cache(
        processed_dir=processed_dir,
        interim_dir=interim_dir,
        cache_path=cache_path,
        hsv_bins=hsv_bins,  # type: ignore[arg-type]
        lbp_p=lbp_p,
        lbp_r=lbp_r,
        force=args.force_features,
    )

    X_train = features["X_train"]
    X_val = features["X_val"]
    X_test = features["X_test"]
    y_train = features["y_train"]
    y_val = features["y_val"]
    y_test = features["y_test"]

    print("Training Perceptron...")
    perceptron = train_perceptron(X_train, y_train, random_state=seed)
    perceptron_metrics = compute_metrics(y_test, perceptron.predict(X_test))

    print("Training Logistic Regression (val tune C)...")
    logistic, logistic_hp = train_logistic_regression(
        X_train,
        y_train,
        X_val,
        y_val,
        c_grid=logistic_c,
        random_state=seed,
    )
    logistic_metrics = compute_metrics(y_test, logistic.predict(X_test))

    print("Training SVM (val tune C, gamma)...")
    svm, svm_hp = train_svm(
        X_train,
        y_train,
        X_val,
        y_val,
        c_grid=svm_c,
        gamma_grid=svm_gamma,
        random_state=seed,
    )
    svm_metrics = compute_metrics(y_test, svm.predict(X_test))

    joblib.dump(perceptron, models_dir / "perceptron.joblib")
    joblib.dump(logistic, models_dir / "logistic_regression.joblib")
    joblib.dump(svm, models_dir / "svm.joblib")

    report = {
        "seed": seed,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "label_names": features["label_names"],
        "models": {
            "Perceptron": {
                "accuracy": perceptron_metrics["accuracy"],
                "macro_f1": perceptron_metrics["macro_f1"],
            },
            "Logistic Regression": {
                "accuracy": logistic_metrics["accuracy"],
                "macro_f1": logistic_metrics["macro_f1"],
                "hyperparameters": logistic_hp,
            },
            "SVM": {
                "accuracy": svm_metrics["accuracy"],
                "macro_f1": svm_metrics["macro_f1"],
                "hyperparameters": {
                    "C": svm_hp["C"],
                    "gamma": svm_hp["gamma"],
                    "val_macro_f1": svm_hp["val_macro_f1"],
                },
            },
        },
    }

    out_path = results_dir / "classical_metrics.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print(f"{'Model':<22} {'Accuracy':>10} {'Macro-F1':>10}")
    print("-" * 44)
    for name, metrics in report["models"].items():
        print(
            f"{name:<22} {metrics['accuracy']:>10.4f} {metrics['macro_f1']:>10.4f}"
        )
    print()
    print(f"Wrote {out_path}")
    print(f"Saved models under {models_dir}")


if __name__ == "__main__":
    main()
