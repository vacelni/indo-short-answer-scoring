"""Fine-tune IndoBERT as a cross-encoder: it reads the question, the reference and the student's answer
together and regresses the score. Unlike the feature models it can see *which* parts of the reference the
answer covers or contradicts.

    python -m sas.crossencoder              # train, pick the best epoch on held-out questions, test

Needs a CUDA GPU in practice (about 10 minutes on an RTX 3050 laptop GPU).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from . import data as dataset
from .metrics import MAX_SCORE, evaluate

ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = "indobenchmark/indobert-base-p1"
OUT_DIR = ROOT / "models" / "indobert-cross-encoder"
RESULTS = ROOT / "results" / "metrics.json"
NAME = "IndoBERT cross-encoder (fine-tuned)"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pairs(frame) -> tuple[list[str], list[str]]:
    return (frame["question"] + " Kunci: " + frame["reference"]).tolist(), frame["answer"].tolist()


@torch.no_grad()
def predict(model, tokenizer, frame, device, batch_size: int = 32, max_length: int = 256) -> np.ndarray:
    model.eval()
    firsts, seconds = pairs(frame)
    out = []
    for i in range(0, len(firsts), batch_size):
        batch = tokenizer(firsts[i : i + batch_size], seconds[i : i + batch_size], truncation="longest_first",
                          max_length=max_length, padding=True, return_tensors="pt").to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            out.append(model(**batch).logits.squeeze(-1).float().cpu().numpy())
    return np.concatenate(out) * MAX_SCORE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)

    frame = dataset.load()
    train_all, test = frame[frame.split == "train"], frame[frame.split == "test"].reset_index(drop=True)
    # Model selection on whole held-out questions, the same situation as the test set.
    fit_idx, val_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=args.seed)
                            .split(train_all, groups=train_all["question_id"]))
    fit, val = train_all.iloc[fit_idx].reset_index(drop=True), train_all.iloc[val_idx].reset_index(drop=True)
    print(f"fit {fit.question_id.nunique()} questions / {len(fit)} answers, validate {val.question_id.nunique()} / {len(val)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1).to(device)

    firsts, seconds = pairs(fit)
    targets = (fit["score"] / MAX_SCORE).tolist()

    def collate(rows):
        a, b, y = zip(*rows)
        batch = tokenizer(list(a), list(b), truncation="longest_first", max_length=256, padding=True, return_tensors="pt")
        batch["labels"] = torch.tensor(y, dtype=torch.float)
        return batch

    loader = DataLoader(list(zip(firsts, seconds, targets)), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    best_qwk, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in loader:
            batch = batch.to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = model(**batch).loss  # num_labels=1 -> mean squared error
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        val_scores = evaluate(val["score"], predict(model, tokenizer, val, device))
        history.append({"epoch": epoch, **val_scores})
        print(f"epoch {epoch}: validation QWK {val_scores['qwk']:.3f} accuracy {val_scores['accuracy']:.3f}")
        if val_scores["qwk"] > best_qwk:
            best_qwk = val_scores["qwk"]
            model.save_pretrained(OUT_DIR)
            tokenizer.save_pretrained(OUT_DIR)

    best = AutoModelForSequenceClassification.from_pretrained(OUT_DIR).to(device)
    test_scores = evaluate(test["score"], predict(best, tokenizer, test, device))
    print(f"official test: {json.dumps({k: round(v, 3) for k, v in test_scores.items()})}")

    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    results[NAME] = {"validation_by_epoch": history, "official_test": test_scores,
                     "config": {"base_model": BASE_MODEL, **vars(args)}}
    RESULTS.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
