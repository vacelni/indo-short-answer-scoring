"""HTTP API and a one-page grading tool for teachers.

    uvicorn sas.api:app            # after `python -m sas.evaluate --save-api`

The served model is the lexical + embedding scorer: it runs on a CPU in milliseconds per answer. Each
response includes the evidence behind the score, so a teacher can see why an answer got it.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .features import FEATURE_NAMES
from .metrics import to_grade
from .models import EmbeddingModel

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "scorer.joblib"
STATIC = Path(__file__).resolve().parent / "static"

MAX_ANSWERS = 200


class Scorer(Protocol):
    def score(self, answers: list[str], reference: str) -> tuple[np.ndarray, np.ndarray]: ...


class EmbeddingScorer:
    def __init__(self, path: Path = MODEL_PATH):
        if not path.exists():
            raise RuntimeError(f"{path} not found: run `python -m sas.evaluate --save-api` first")
        # joblib unpickles: only ever load the file this repository's `sas.evaluate --save-api` wrote,
        # never one downloaded or uploaded from elsewhere.
        saved = joblib.load(path)
        self.model = EmbeddingModel(encoder_name=saved["encoder_name"])
        self.model.model = saved["regressor"]

    def score(self, answers, reference):
        features = self.model.features(answers, [reference] * len(answers))
        return np.clip(self.model.model.predict(features), 0, 5), features


@lru_cache(maxsize=1)
def get_scorer() -> Scorer:
    return EmbeddingScorer()


class ScoreRequest(BaseModel):
    reference: str = Field(min_length=1, max_length=4000, description="The teacher's reference answer")
    answers: list[str] = Field(min_length=1, max_length=MAX_ANSWERS, description="Student answers to grade")


class Evidence(BaseModel):
    reference_recall: float = Field(description="Share of the reference's key words that the answer contains")
    char3_dice: float = Field(description="Character-trigram overlap; robust to Indonesian affixes")
    embedding_cosine: float = Field(description="Meaning similarity from a multilingual sentence model")
    blank_or_unsure: bool


class ScoredAnswer(BaseModel):
    answer: str
    score: float = Field(description="Predicted score on the 0-5 scale")
    grade: int = Field(description="Score rounded to a whole grade")
    evidence: Evidence


class ScoreResponse(BaseModel):
    results: list[ScoredAnswer]


app = FastAPI(title="Indonesian short-answer scoring", version="1.0.0")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest, scorer: Scorer = Depends(get_scorer)) -> ScoreResponse:
    answers = [a[:2000] for a in request.answers]
    if not any(a.strip() for a in answers):
        raise HTTPException(status_code=422, detail="Every answer is empty.")
    scores, features = scorer.score(answers, request.reference)
    column = {name: i for i, name in enumerate([*FEATURE_NAMES, "embedding_cosine"])}
    return ScoreResponse(
        results=[
            ScoredAnswer(
                answer=answer,
                score=round(float(s), 2),
                grade=int(to_grade([s])[0]),
                evidence=Evidence(
                    reference_recall=round(float(f[column["reference_recall"]]), 3),
                    char3_dice=round(float(f[column["char3_dice"]]), 3),
                    embedding_cosine=round(float(f[column["embedding_cosine"]]), 3),
                    blank_or_unsure=bool(f[column["is_blank_or_unsure"]]),
                ),
            )
            for answer, s, f in zip(answers, scores, features)
        ]
    )
