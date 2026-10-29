import pandas as pd

from sas.data import clean


def test_clean_keeps_valid_scores_and_drops_unusable_rows():
    raw = pd.DataFrame({
        "answer": ["a", "b", "c", "d", "e", "  "],
        "question": ["q"] * 6,
        "reference": ["r"] * 6,
        "score": ["5", "`0", "5?", "-", None, "3"],
    })
    out = clean(raw)
    assert out["answer"].tolist() == ["a", "b"]
    assert out["score"].tolist() == [5, 0]
