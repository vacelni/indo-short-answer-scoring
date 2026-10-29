"""Hand-built features comparing a student's answer with the reference answer.

Every feature looks at one answer and its reference only, so nothing is fitted on the data and nothing
can leak from the test questions. Character n-grams carry most of the signal: Indonesian builds words
with affixes (men-, -kan, ber-, -an), so 'mempercepat' and 'percepatan' share most of their trigrams
while sharing no whole word.
"""

from __future__ import annotations

import math
import re

FEATURE_NAMES = [
    "reference_recall",
    "answer_precision",
    "word_jaccard",
    "char3_dice",
    "char4_dice",
    "length_ratio",
    "answer_words",
    "is_blank_or_unsure",
]

# Function words that say nothing about whether an answer is right.
STOPWORDS = frozenset(
    """
    yang dan di ke dari pada untuk dengan adalah ini itu atau dalam oleh sebagai juga akan karena
    agar bahwa tersebut suatu sebuah para serta maupun yaitu yakni ialah merupakan dapat bisa ada
    tidak bukan saja hanya sangat lebih paling secara sehingga jika maka kita kami saya mereka ia
    """.split()
)

UNSURE = re.compile(r"^(tidak tahu|tdk tahu|gatau|ga tau|gak tau|nggak tahu|entah|lupa|-|\?+|\.+)$")
WORD = re.compile(r"[a-z0-9]+")


def content_words(text: str) -> list[str]:
    return [w for w in WORD.findall(text.lower()) if w not in STOPWORDS]


def char_ngrams(text: str, n: int) -> set[str]:
    joined = " ".join(WORD.findall(text.lower()))
    padded = f" {joined} "
    return {padded[i : i + n] for i in range(max(0, len(padded) - n + 1))}


def dice(a: set[str], b: set[str]) -> float:
    return 2 * len(a & b) / (len(a) + len(b)) if a and b else 0.0


def lexical_features(answer: str, reference: str) -> list[float]:
    """Values in the order of FEATURE_NAMES."""
    answer_words, reference_words = set(content_words(answer)), set(content_words(reference))
    shared = answer_words & reference_words
    n_answer = len(content_words(answer))
    n_reference = max(1, len(content_words(reference)))
    return [
        len(shared) / len(reference_words) if reference_words else 0.0,
        len(shared) / len(answer_words) if answer_words else 0.0,
        len(shared) / len(answer_words | reference_words) if answer_words | reference_words else 0.0,
        dice(char_ngrams(answer, 3), char_ngrams(reference, 3)),
        dice(char_ngrams(answer, 4), char_ngrams(reference, 4)),
        min(n_answer / n_reference, 3.0),
        math.log1p(n_answer),
        1.0 if not answer.strip() or UNSURE.match(answer.strip().lower()) else 0.0,
    ]
