import numpy as np
from fastapi.testclient import TestClient

from sas.api import app, get_scorer
from sas.features import FEATURE_NAMES, lexical_features


class FakeScorer:
    """Lexical features plus a fixed similarity, so the API is tested without downloading models."""

    def score(self, answers, reference):
        features = np.array([[*lexical_features(a, reference), 0.5] for a in answers])
        return np.clip(features[:, FEATURE_NAMES.index("reference_recall")] * 5, 0, 5), features


app.dependency_overrides[get_scorer] = FakeScorer
client = TestClient(app)


def test_scores_each_answer_with_its_evidence():
    response = client.post("/score", json={"reference": "fotosintesis menghasilkan oksigen", "answers": ["fotosintesis menghasilkan oksigen", "tidak tahu"]})
    assert response.status_code == 200
    first, second = response.json()["results"]
    assert first["grade"] == 5 and first["evidence"]["reference_recall"] == 1.0
    assert second["grade"] == 0 and second["evidence"]["blank_or_unsure"] is True


def test_rejects_requests_without_answers_or_reference():
    assert client.post("/score", json={"reference": "x", "answers": []}).status_code == 422
    assert client.post("/score", json={"reference": "", "answers": ["a"]}).status_code == 422
    assert client.post("/score", json={"reference": "x", "answers": ["   "]}).status_code == 422


def test_caps_the_number_of_answers_per_request():
    assert client.post("/score", json={"reference": "x", "answers": ["a"] * 201}).status_code == 422


def test_serves_the_teacher_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "Kunci jawaban" in response.text
