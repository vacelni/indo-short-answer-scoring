import numpy as np
import pytest

from sas.metrics import evaluate, to_grade


def test_perfect_predictions_score_one():
    scores = [0, 1, 2, 3, 4, 5, 5, 0]
    result = evaluate(scores, scores)
    assert result["qwk"] == pytest.approx(1.0)
    assert result["accuracy"] == 1.0
    assert result["rmse"] == 0.0


def test_grades_are_rounded_and_clipped_to_the_scale():
    assert to_grade([-1.2, 0.49, 2.5, 3.51, 7]).tolist() == [0, 0, 2, 4, 5]


def test_within_one_counts_near_misses():
    result = evaluate([0, 2, 4, 5], [1, 2, 2, 5])
    assert result["accuracy"] == 0.5
    assert result["within_one"] == 0.75


def test_constant_predictions_have_zero_correlation_not_nan():
    result = evaluate([0, 3, 5], np.full(3, 2.39))
    assert result["pearson"] == 0.0
    assert result["qwk"] == 0.0
