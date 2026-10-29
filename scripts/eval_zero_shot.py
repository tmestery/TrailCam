#!/usr/bin/env python3
"""Evaluate zero-shot CLIP on the held-out test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import yaml

from trailcam.data.dataset import load_label_names
from trailcam.eval.metrics import compute_metrics
from trailcam.features.classical import load_split_csv
from trailcam.models.llm import classify_zero_shot
from trailcam.utils.paths import repo_root


def load_config(config_path: Path) -> dict:
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve(root: Path, maybe_relative: str) -> Path:
    path = Path(maybe_relative)
    return path if path.is_absolute() else root / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot CLIP evaluation.")
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root() / "configs" / "default.yaml",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = repo_root()
    seed = int(cfg.get("seed", 42))
    clip_cfg = cfg.get("clip", {})

    interim_dir = _resolve(root, cfg["paths"]["interim"])
    processed_dir = _resolve(root, cfg["paths"]["processed"])
    results_dir = _resolve(root, cfg["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)

    model_name = str(clip_cfg.get("model_name", "ViT-B-32"))
    pretrained = str(clip_cfg.get("pretrained", "openai"))
    batch_size = int(clip_cfg.get("batch_size", 64))

    label_names = load_label_names(processed_dir)
    label_to_id = {name: i for i, name in enumerate(label_names)}

    test_paths, test_species = load_split_csv(processed_dir / "test.csv", interim_dir)
    y_true = [label_to_id[s] for s in test_species]

    print(f"Zero-shot CLIP ({model_name}, {pretrained}) on {len(test_paths)} test crops")
    y_pred = classify_zero_shot(
        test_paths,
        label_names,
        model_name=model_name,
        pretrained=pretrained,
        batch_size=batch_size,
    )
    metrics = compute_metrics(y_true, y_pred)

    report = {
        "seed": seed,
        "n_test": len(y_true),
        "label_names": label_names,
        "hyperparameters": {
            "model_name": model_name,
            "pretrained": pretrained,
            "prompt_template": "a camera trap photo of a {species}",
            "fine_tuned": False,
        },
        "models": {
            "Zero-Shot VLM": {
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
            }
        },
    }

    out_path = results_dir / "zero_shot_metrics.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print()
    print(
        f"Zero-Shot VLM test accuracy={metrics['accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
