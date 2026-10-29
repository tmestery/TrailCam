#!/usr/bin/env python3
"""Merge classical / CNN / zero-shot metric JSON files into one report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import yaml

from trailcam.utils.paths import repo_root


def load_config(config_path: Path) -> dict:
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge TrailCam metric reports.")
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root() / "configs" / "default.yaml",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = repo_root()
    results_dir = root / cfg["paths"]["results"]

    sources = {
        "classical": results_dir / "classical_metrics.json",
        "cnn": results_dir / "cnn_metrics.json",
        "zero_shot": results_dir / "zero_shot_metrics.json",
    }

    merged_models: dict = {}
    meta: dict = {"sources": {}}
    for key, path in sources.items():
        if not path.exists():
            print(f"skip missing {path}")
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        meta["sources"][key] = str(path)
        for field in ("seed", "n_train", "n_val", "n_test", "label_names"):
            if field in report and field not in meta:
                meta[field] = report[field]
        merged_models.update(report.get("models", {}))

    order = [
        "Perceptron",
        "Logistic Regression",
        "SVM",
        "CNN",
        "Zero-Shot VLM",
    ]
    ordered = {name: merged_models[name] for name in order if name in merged_models}

    out = {**meta, "models": ordered}
    out_path = results_dir / "all_metrics.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"{'Model':<22} {'Accuracy':>10} {'Macro-F1':>10}")
    print("-" * 44)
    for name, metrics in ordered.items():
        print(
            f"{name:<22} {metrics['accuracy']:>10.4f} {metrics['macro_f1']:>10.4f}"
        )
    print()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
