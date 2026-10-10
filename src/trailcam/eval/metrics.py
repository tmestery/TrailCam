"""Classification metrics (accuracy, macro-F1) for model comparison."""

from __future__ import annotations

from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    """Return accuracy and macro-F1 for imbalanced species labels."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
