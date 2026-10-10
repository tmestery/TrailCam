"""Classical baselines: Perceptron, Logistic Regression, and SVM."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from trailcam.eval.metrics import compute_metrics


def _score_macro_f1(model, X: np.ndarray, y: np.ndarray) -> float:
    pred = model.predict(X)
    return float(compute_metrics(y, pred)["macro_f1"])


def train_perceptron(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> Pipeline:
    """Train a linear Perceptron on engineered features (no hyperparameter search)."""
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", Perceptron(max_iter=1000, random_state=random_state)),
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    c_grid: list[float] | None = None,
    random_state: int = 42,
) -> tuple[Pipeline, dict[str, Any]]:
    """Train L2 logistic regression; select ``C`` by validation macro-F1."""
    if c_grid is None:
        c_grid = [0.1, 1.0, 10.0]

    best_score = -1.0
    best_pipe: Pipeline | None = None
    best_c = c_grid[0]

    for c in c_grid:
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        penalty="l2",
                        C=c,
                        max_iter=2000,
                        solver="lbfgs",
                        random_state=random_state,
                    ),
                ),
            ]
        )
        pipe.fit(X_train, y_train)
        score = _score_macro_f1(pipe, X_val, y_val)
        if score > best_score:
            best_score = score
            best_pipe = pipe
            best_c = c

    assert best_pipe is not None
    return best_pipe, {"C": best_c, "val_macro_f1": best_score}


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    c_grid: list[float] | None = None,
    gamma_grid: list[float | str] | None = None,
    random_state: int = 42,
) -> tuple[Pipeline, dict[str, Any]]:
    """Train an RBF SVM; select ``C`` and ``gamma`` by validation macro-F1."""
    if c_grid is None:
        c_grid = [0.1, 1.0, 10.0]
    if gamma_grid is None:
        gamma_grid = ["scale", 0.01]

    best_score = -1.0
    best_pipe: Pipeline | None = None
    best_params: dict[str, Any] = {}

    for c in c_grid:
        for gamma in gamma_grid:
            pipe = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        SVC(
                            kernel="rbf",
                            C=c,
                            gamma=gamma,
                            random_state=random_state,
                        ),
                    ),
                ]
            )
            pipe.fit(X_train, y_train)
            score = _score_macro_f1(pipe, X_val, y_val)
            if score > best_score:
                best_score = score
                best_pipe = pipe
                best_params = {"C": c, "gamma": gamma, "val_macro_f1": best_score}

    assert best_pipe is not None
    return best_pipe, best_params
