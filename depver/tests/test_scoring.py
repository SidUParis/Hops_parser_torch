"""Tests for scoring metrics."""

from depver.schema import (
    Alignment,
    Divergence,
    DivergenceType,
    Entity,
    Triple,
    ClauseType,
)
from depver.scoring.metrics import compute_scores


def _make_triple(pred: str, subj: str | None = None, obj: str | None = None) -> Triple:
    return Triple(
        predicate_lemma=pred,
        predicate_form=pred,
        predicate_upos="VERB",
        predicate_id=1,
        subject=Entity(head_lemma=subj, head_form=subj, head_upos="NOUN") if subj else None,
        object=Entity(head_lemma=obj, head_form=obj, head_upos="NOUN") if obj else None,
    )


class TestComputeScores:
    def test_perfect_match(self):
        t = _make_triple("sign", "Macron", "bill")
        alignments = [Alignment(source_triple=t, generated_triple=t, similarity_score=1.0)]
        scores = compute_scores(alignments, [])

        assert scores.triple_precision == 1.0
        assert scores.triple_recall == 1.0
        assert scores.triple_f1 == 1.0
        assert scores.factuality_score == 1.0

    def test_hallucination_lowers_precision(self):
        t1 = _make_triple("sign", "Macron", "bill")
        t2 = _make_triple("rise", "price")
        alignments = [
            Alignment(source_triple=t1, generated_triple=t1, similarity_score=1.0),
            Alignment(source_triple=None, generated_triple=t2, similarity_score=0.0),
        ]
        scores = compute_scores(alignments, [])

        assert scores.triple_precision == 0.5
        assert scores.triple_recall == 1.0
        assert scores.num_hallucinated == 1

    def test_omission_lowers_recall(self):
        t1 = _make_triple("sign", "Macron", "bill")
        t2 = _make_triple("rise", "price")
        alignments = [
            Alignment(source_triple=t1, generated_triple=t1, similarity_score=1.0),
            Alignment(source_triple=t2, generated_triple=None, similarity_score=0.0),
        ]
        scores = compute_scores(alignments, [])

        assert scores.triple_precision == 1.0
        assert scores.triple_recall == 0.5
        assert scores.num_omitted == 1

    def test_severity_weighted_factuality(self):
        t = _make_triple("sign", "Macron", "bill")
        alignments = [Alignment(source_triple=t, generated_triple=t, similarity_score=0.8)]
        divergences = [
            Divergence(
                type=DivergenceType.NEGATION_FLIP,
                description="test",
                source_span="",
                generated_span="",
            )
        ]
        scores = compute_scores(alignments, divergences)
        # Critical divergence (weight 1.0) on 1 generated triple -> factuality = 0.0
        assert scores.factuality_score == 0.0

    def test_empty_inputs(self):
        scores = compute_scores([], [])
        assert scores.triple_precision == 1.0
        assert scores.triple_recall == 1.0
