"""Download and tidy the Indonesian short-answer dataset of Haidir & Purwarianti (2020).

Source: https://github.com/AgeMagi/tugas-akhir (CC BY-NC 4.0). The data is not redistributed in this
repository; `python -m sas.data` downloads it into data/raw/.

The official split matters: the test set holds 6 questions that never appear in training, so a model is
judged on what a teacher actually needs, which is grading a new question from its reference answer alone.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import pandas as pd

BASE_URL = "https://raw.githubusercontent.com/AgeMagi/tugas-akhir/master/data"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DOMAINS = ("saintek", "soshum")  # science (biology) and social studies (geography)
SPLITS = {"train": "", "test": "-test"}
VALID_SCORES = {str(i) for i in range(6)}


def raw_files() -> list[str]:
    return [f"{kind}-{domain}{suffix}.csv" for domain in DOMAINS for suffix in SPLITS.values() for kind in ("question", "score")]


def download(raw_dir: Path = RAW_DIR) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in raw_files():
        target = raw_dir / name
        if not target.exists():
            urllib.request.urlretrieve(f"{BASE_URL}/{name}", target)
            print(f"downloaded {name}")


def load(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """One row per graded answer: split, domain, question_id, question, reference, answer, score (0-5)."""
    missing = [n for n in raw_files() if not (raw_dir / n).exists()]
    if missing:
        raise FileNotFoundError(f"Dataset not found in {raw_dir}. Run `python -m sas.data` first.")

    frames = []
    for domain in DOMAINS:
        for split, suffix in SPLITS.items():
            questions = pd.read_csv(raw_dir / f"question-{domain}{suffix}.csv")
            questions = questions.rename(columns={"type-problem": "number", "pertanyaan": "question", "jawaban": "reference"})
            answers = pd.read_csv(raw_dir / f"score-{domain}{suffix}.csv", dtype={"score": str})
            answers = answers.rename(columns={"type-problem": "number", "jawaban": "answer"})
            merged = answers.merge(questions, on="number", how="left", validate="many_to_one")
            merged.insert(0, "question_id", f"{domain}-{split}-" + merged["number"].astype(str))
            merged.insert(0, "domain", domain)
            merged.insert(0, "split", split)
            frames.append(merged.drop(columns="number"))

    data = pd.concat(frames, ignore_index=True)
    return clean(data)


def clean(data: pd.DataFrame) -> pd.DataFrame:
    """Drop answers that cannot be used for training or evaluation, and say how many."""
    score = data["score"].astype(str).str.strip().str.lstrip("`")
    keep = score.isin(VALID_SCORES) & data["answer"].notna() & (data["answer"].astype(str).str.strip() != "")
    dropped = (~keep).sum()
    if dropped:
        # Missing scores ('nan'), unsure ones ('5?', '-') and empty answers: about 4% of the rows.
        print(f"dropped {dropped} of {len(data)} rows without a usable score or answer", file=sys.stderr)
    data = data[keep].copy()
    data["score"] = score[keep].astype(int)
    for column in ("answer", "question", "reference"):
        data[column] = data[column].astype(str).str.strip()
    return data.reset_index(drop=True)


if __name__ == "__main__":
    download()
    frame = load()
    print(frame.groupby(["split", "domain"]).agg(questions=("question_id", "nunique"), answers=("answer", "size")))
