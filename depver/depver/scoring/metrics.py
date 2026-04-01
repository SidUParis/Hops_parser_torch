"""Aggregate verification metrics."""

from __future__ import annotations

from depver.schema import Alignment, Divergence, DivergenceType, VerificationScores


_SEVERITY_WEIGHTS = {
    "critical": 1.0,
    "high": 0.7,
    "medium": 0.3,
}


def compute_scores(
    alignments: list[Alignment],
    divergences: list[Divergence],
) -> VerificationScores:
    """Compute aggregate verification scores from alignments and divergences."""
    matched = [a for a in alignments if a.source_triple and a.generated_triple]
    hallucinated = [a for a in alignments if a.source_triple is None]
    omitted = [a for a in alignments if a.generated_triple is None]

    total_gen = len(matched) + len(hallucinated)
    total_src = len(matched) + len(omitted)

    precision = len(matched) / total_gen if total_gen > 0 else 1.0
    recall = len(matched) / total_src if total_src > 0 else 1.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Severity-weighted factuality score
    penalty = sum(
        _SEVERITY_WEIGHTS.get(d.type.severity, 0.3) for d in divergences
    )
    max_penalty = max(total_gen, 1)
    factuality = max(0.0, 1.0 - penalty / max_penalty)

    # Count matched triples that have divergences
    divergent_sigs = set()
    for d in divergences:
        if d.type != DivergenceType.OMISSION and d.type != DivergenceType.ENTITY_HALLUCINATION:
            divergent_sigs.add((d.source_span, d.generated_span))

    return VerificationScores(
        triple_precision=precision,
        triple_recall=recall,
        triple_f1=f1,
        factuality_score=factuality,
        num_hallucinated=len(hallucinated),
        num_omitted=len(omitted),
        num_divergent=len(divergent_sigs),
    )
