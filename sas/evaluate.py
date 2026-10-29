"""Evaluate every model the same way and write results/metrics.json.

Two numbers per model:
  * cross-validation on the 30 training questions, grouped by question (GroupKFold), so no question is
    ever in both the fold's training and validation part;
  * the official test set: 6 questions never seen in training.

    python -m sas.evaluate                 # feature models
    python -m sas.evaluate --save-api      # also fit the API model on all training data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold

from . import data as dataset
from .metrics import evaluate
from .models import EmbeddingModel, LexicalModel, MeanScore

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "metrics.json"
API_MODEL = ROOT / "models" / "scorer.joblib"


def cross_validate(make_model, train, folds: int = 5) -> dict[str, float]:
    predictions = np.zeros(len(train))
    for fit_idx, val_idx in GroupKFold(n_splits=folds).split(train, groups=train["question_id"]):
        fit, val = train.iloc[fit_idx], train.iloc[val_idx]
        model = make_model().fit(fit["answer"], fit["reference"], fit["score"])
        predictions[val_idx] = model.predict(val["answer"], val["reference"])
    return evaluate(train["score"], predictions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-api", action="store_true", help="fit the embedding model on all training data and save it")
    args = parser.parse_args()

    frame = dataset.load()
    train, test = frame[frame.split == "train"].reset_index(drop=True), frame[frame.split == "test"].reset_index(drop=True)

    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    for make_model in (MeanScore, LexicalModel, EmbeddingModel):
        name = make_model().name
        cv = cross_validate(make_model, train)
        model = make_model().fit(train["answer"], train["reference"], train["score"])
        test_scores = evaluate(test["score"], model.predict(test["answer"], test["reference"]))
        results[name] = {"cv_unseen_questions": cv, "official_test": test_scores}
        print(f"{name:48s} CV QWK {cv['qwk']:.3f} | test QWK {test_scores['qwk']:.3f} acc {test_scores['accuracy']:.3f} ±1 {test_scores['within_one']:.3f}")
        if args.save_api and make_model is EmbeddingModel:
            API_MODEL.parent.mkdir(exist_ok=True)
            joblib.dump({"regressor": model.model, "encoder_name": model.encoder_name}, API_MODEL)
            print(f"saved {API_MODEL.relative_to(ROOT)}")

    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
