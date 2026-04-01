"""Triple alignment between source and generated texts."""

from __future__ import annotations

from depver.schema import Triple, Alignment
from depver.comparison.similarity import (
    predicate_similarity,
    entity_similarity,
    polarity_match,
    oblique_similarity,
)


def triple_similarity(g: Triple, s: Triple) -> float:
    """Weighted similarity between two triples."""
    w_pred = 0.30
    w_subj = 0.25
    w_obj = 0.25
    w_pol = 0.10
    w_obl = 0.10

    pred_sim = predicate_similarity(g.predicate_lemma, s.predicate_lemma)

    if g.subject and s.subject:
        subj_sim = entity_similarity(g.subject, s.subject)
    elif g.subject is None and s.subject is None:
        subj_sim = 1.0
    else:
        subj_sim = 0.0

    if g.object and s.object:
        obj_sim = entity_similarity(g.object, s.object)
    elif g.object is None and s.object is None:
        obj_sim = 1.0
    else:
        obj_sim = 0.0

    pol_sim = polarity_match(g.negated, s.negated)
    obl_sim = oblique_similarity(g.obliques, s.obliques)

    return (
        w_pred * pred_sim
        + w_subj * subj_sim
        + w_obj * obj_sim
        + w_pol * pol_sim
        + w_obl * obl_sim
    )


def align_triples(
    source_triples: list[Triple],
    generated_triples: list[Triple],
    threshold: float = 0.4,
) -> list[Alignment]:
    """Align generated triples to source triples using greedy best-match.

    Returns a list of Alignments:
    - Matched pairs (both source and generated present)
    - Hallucinations (source=None, generated present)
    - Omissions (source present, generated=None)
    """
    if not source_triples and not generated_triples:
        return []

    if not source_triples:
        return [
            Alignment(source_triple=None, generated_triple=g, similarity_score=0.0)
            for g in generated_triples
        ]

    if not generated_triples:
        return [
            Alignment(source_triple=s, generated_triple=None, similarity_score=0.0)
            for s in source_triples
        ]

    # Compute similarity matrix
    sim_matrix: list[list[float]] = []
    for g in generated_triples:
        row = [triple_similarity(g, s) for s in source_triples]
        sim_matrix.append(row)

    # Greedy matching: assign best pairs above threshold
    alignments: list[Alignment] = []
    used_source: set[int] = set()
    used_gen: set[int] = set()

    # Collect all pairs, sort by similarity descending
    pairs = [
        (sim_matrix[gi][si], gi, si)
        for gi in range(len(generated_triples))
        for si in range(len(source_triples))
    ]
    pairs.sort(key=lambda x: x[0], reverse=True)

    for score, gi, si in pairs:
        if gi in used_gen or si in used_source:
            continue
        if score < threshold:
            break
        alignments.append(
            Alignment(
                source_triple=source_triples[si],
                generated_triple=generated_triples[gi],
                similarity_score=score,
            )
        )
        used_gen.add(gi)
        used_source.add(si)

    # Unmatched generated → hallucination candidates
    for gi in range(len(generated_triples)):
        if gi not in used_gen:
            alignments.append(
                Alignment(
                    source_triple=None,
                    generated_triple=generated_triples[gi],
                    similarity_score=0.0,
                )
            )

    # Unmatched source → omissions
    for si in range(len(source_triples)):
        if si not in used_source:
            alignments.append(
                Alignment(
                    source_triple=source_triples[si],
                    generated_triple=None,
                    similarity_score=0.0,
                )
            )

    return alignments
