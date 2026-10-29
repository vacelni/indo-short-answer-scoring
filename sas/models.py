"""Scoring models, from simplest to strongest. Each one predicts a 0-5 score from (answer, reference)."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import FEATURE_NAMES, lexical_features

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@lru_cache(maxsize=2)
def sentence_encoder(name: str = EMBEDDING_MODEL):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


def cosine_similarity(answers: list[str], references: list[str], encoder_name: str = EMBEDDING_MODEL) -> np.ndarray:
    encoder = sentence_encoder(encoder_name)
    # References repeat across hundreds of answers; encode each distinct one once.
    unique_refs = sorted(set(references))
    ref_vectors = dict(zip(unique_refs, encoder.encode(unique_refs, normalize_embeddings=True, batch_size=64)))
    answer_vectors = encoder.encode(answers, normalize_embeddings=True, batch_size=64)
    return np.array([float(a @ ref_vectors[r]) for a, r in zip(answer_vectors, references)])


def regressor() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0)


class MeanScore:
    """Sanity floor: always predicts the average training score."""

    name = "Mean score (floor)"

    def fit(self, answers, references, scores):
        self.mean_ = float(np.mean(scores))
        return self

    def predict(self, answers, references):
        return np.full(len(answers), self.mean_)


@dataclass
class LexicalModel:
    """Word and character-n-gram overlap with the reference, fed to gradient-boosted trees."""

    name: str = "Lexical overlap + gradient boosting"
    model: HistGradientBoostingRegressor = field(default_factory=regressor)

    def features(self, answers, references) -> np.ndarray:
        return np.array([lexical_features(a, r) for a, r in zip(answers, references)])

    def fit(self, answers, references, scores):
        self.model.fit(self.features(answers, references), scores)
        return self

    def predict(self, answers, references):
        return self.model.predict(self.features(answers, references))


@dataclass
class EmbeddingModel(LexicalModel):
    """Lexical features plus the cosine similarity of multilingual sentence embeddings, which credits
    answers that say the right thing in different words."""

    name: str = "Lexical + multilingual embedding similarity"
    encoder_name: str = EMBEDDING_MODEL

    feature_names = [*FEATURE_NAMES, "embedding_cosine"]

    def features(self, answers, references) -> np.ndarray:
        lexical = super().features(answers, references)
        similarity = cosine_similarity(list(answers), list(references), self.encoder_name)
        return np.column_stack([lexical, similarity])
