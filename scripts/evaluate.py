#!/usr/bin/env python3
"""Print classical model metrics from results/classical_metrics.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python scripts/evaluate.py` without relying on an editable install.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import yaml

from trailcam.utils.paths import repo_root


def load_config(config_path: Path) -> dict:
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate / display classical metrics.")
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root() / "configs" / "default.yaml",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="Path to classical_metrics.json (default: results/classical_metrics.json)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = repo_root()
    results_dir = root / cfg["paths"]["results"]
    metrics_path = args.metrics or (results_dir / "classical_metrics.json")

    if not metrics_path.exists():
        raise SystemExit(
            f"Missing {metrics_path}. Run: python scripts/train.py"
        )

    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    models = report.get("models", {})

    print(f"Metrics from {metrics_path}")
    print(f"n_test={report.get('n_test')}  seed={report.get('seed')}")
    print()
    print(f"{'Model':<22} {'Accuracy':>10} {'Macro-F1':>10}")
    print("-" * 44)
    for name, metrics in models.items():
        print(
            f"{name:<22} {metrics['accuracy']:>10.4f} {metrics['macro_f1']:>10.4f}"
        )


if __name__ == "__main__":
    main()
