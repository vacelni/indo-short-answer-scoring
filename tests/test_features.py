import pytest

from sas.features import FEATURE_NAMES, lexical_features

REFERENCE = "Ekosistem adalah hubungan timbal balik antara makhluk hidup dengan lingkungannya"


def named(answer, reference=REFERENCE):
    return dict(zip(FEATURE_NAMES, lexical_features(answer, reference)))


def test_copying_the_reference_maxes_out_overlap():
    f = named(REFERENCE)
    assert f["reference_recall"] == 1.0
    assert f["char3_dice"] == pytest.approx(1.0)
    assert f["is_blank_or_unsure"] == 0.0


def test_stopwords_do_not_count_as_shared_content():
    assert named("adalah dengan yang dan")["reference_recall"] == 0.0


def test_character_ngrams_credit_shared_roots_across_affixes():
    whole_words = named("percepatan", "mempercepat")["word_jaccard"]
    trigrams = named("percepatan", "mempercepat")["char3_dice"]
    assert whole_words == 0.0
    assert trigrams > 0.4


@pytest.mark.parametrize("answer", ["tidak tahu", "Gatau", "-", "???", ""])
def test_blank_and_unsure_answers_are_flagged(answer):
    assert named(answer)["is_blank_or_unsure"] == 1.0
