# Indonesian Short-Answer Scoring

**Scores students' short answers in Indonesian (0–5) against the teacher's reference answer, and shows why each answer got its score.** It includes an evaluation that tests what a teacher actually needs, grading a *new* question, plus an HTTP API and a one-page grading tool.

[![Tests](https://github.com/vacelni/indo-short-answer-scoring/actions/workflows/tests.yml/badge.svg)](https://github.com/vacelni/indo-short-answer-scoring/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-F7931E?logo=scikitlearn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.16-009688?logo=fastapi&logoColor=white)

## Results

The dataset's official test set has **6 questions that never appear in training** (1,499 graded answers). Every model is also cross-validated on the 30 training questions with `GroupKFold`, so a question is never on both sides of a fold.

| Model | QWK (test) | Pearson | Exact grade | Within ±1 | QWK (5-fold CV) |
|---|---|---|---|---|---|
| Always predict the average (floor) | 0.000 | 0.000 | 15.5% | 54.2% | 0.000 |
| Lexical overlap + gradient boosting | **0.573** | **0.606** | 31.2% | 69.4% | 0.348 |
| + multilingual sentence-embedding similarity | 0.565 | 0.597 | **34.1%** | **69.6%** | **0.370** |

QWK (quadratic weighted kappa) is the standard agreement measure for automated scoring. It rewards being close, not only being exact. The numbers come from `results/metrics.json`, and running `python -m sas.evaluate` again reproduces them exactly.

**How to read this honestly.** On questions the model has never seen, it agrees with human graders moderately: about a third of answers get exactly the same grade and seven in ten are within one point. That is useful for sorting a pile of answers or flagging ones to read first. It is not good enough to replace the teacher, and the tool says so.

## What I changed from my first version

My first attempt, during an internship, reported a much higher accuracy. When I rebuilt it I found why:

- **Data leakage.** The TF-IDF vocabulary was fitted on all answers before the train/test split. Here every feature compares one answer with its own reference only, so nothing is fitted on test data (`sas/features.py`).
- **Seen questions.** Answers were split at random, so the test set held answers to the same questions the model trained on. Here evaluation uses the dataset's own split into unseen questions, plus grouped cross-validation.
- **Mislabelled models.** Two classifiers were reported as regressors. Here there is one regression setup and the standard metrics: QWK, Pearson, RMSE, exact and within-one accuracy.

The lower numbers above are the correct ones.

## How it scores

Each answer is compared with the reference using features that need no training on the question itself:

- **key-word recall and precision**: content words only, Indonesian function words removed;
- **character 3- and 4-gram overlap**: Indonesian builds words with affixes, so *mempercepat* and *percepatan* share most trigrams and no whole word;
- **length ratio** and a flag for blank or "tidak tahu" answers;
- **meaning similarity**: cosine similarity of multilingual sentence embeddings (`paraphrase-multilingual-MiniLM-L12-v2`), which credits a correct answer written in different words.

Gradient-boosted trees turn the features into a 0–5 score. The API returns the score together with this evidence.

`sas/crossencoder.py` fine-tunes **IndoBERT** (`indobenchmark/indobert-base-p1`) as a cross-encoder that reads the question, reference and answer together. Model selection uses held-out *questions*. It needs a CUDA GPU and is not in the table yet.

## Try it

```bash
python -m venv .venv && .venv/Scripts/activate      # or source .venv/bin/activate
pip install -r requirements-dev.txt
python -m sas.data                                   # download the dataset (see below)
python -m sas.evaluate --save-api                    # evaluate, and save the API model
uvicorn sas.api:app                                  # http://localhost:8000
```

Or with Docker, which downloads the data and trains the API model during the build:

```bash
docker build -t sas . && docker run -p 8000:8000 sas
```

`POST /score` takes `{"reference": "...", "answers": ["...", "..."]}` (up to 200 answers) and returns a score, a whole grade and the evidence for each answer. `/` serves a grading page in Indonesian for teachers.

```bash
pytest -q    # 17 tests: metrics, features, data cleaning, API (with a fake model, no downloads)
```

## Data

Haidir & Purwarianti (2020), *Short Answer Grading Using Contextual Word Embedding and Linear Regression*, Jurnal Linguistik Komputasional. Data from [github.com/AgeMagi/tugas-akhir](https://github.com/AgeMagi/tugas-akhir) under **CC BY-NC 4.0**. It is not redistributed here; `sas/data.py` downloads it. Biology and geography answers from Indonesian students, graded 0–5. 336 of 9,163 rows are dropped because the score is missing or unclear (for example `5?` or `-`) or the answer is empty.

## License

Code: [MIT](LICENSE) © Vacelnino Faiz Fadeli. The dataset keeps its own CC BY-NC 4.0 license.
