"""Metrics for 0-5 short-answer scores.

Quadratic weighted kappa (QWK) is the standard agreement measure for automated scoring: it rewards
being close, not only being exact. Exact and within-one accuracy are reported because a teacher reads
them more easily.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score

MIN_SCORE, MAX_SCORE = 0, 5


def to_grade(predictions) -> np.ndarray:
    """Continuous predictions to the integer grades a teacher would give."""
    return np.clip(np.rint(np.asarray(predictions, dtype=float)), MIN_SCORE, MAX_SCORE).astype(int)


def evaluate(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), MIN_SCORE, MAX_SCORE)
    grades_true, grades_pred = to_grade(y_true), to_grade(y_pred)
    return {
        "qwk": float(cohen_kappa_score(grades_true, grades_pred, weights="quadratic", labels=list(range(MIN_SCORE, MAX_SCORE + 1)))),
        "pearson": float(pearsonr(y_true, y_pred).statistic) if np.ptp(y_pred) > 0 else 0.0,
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "accuracy": float(np.mean(grades_true == grades_pred)),
        "within_one": float(np.mean(np.abs(grades_true - grades_pred) <= 1)),
    }
