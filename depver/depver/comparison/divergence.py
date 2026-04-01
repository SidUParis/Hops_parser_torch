"""Divergence classification for aligned triples."""

from __future__ import annotations

from depver.schema import (
    Alignment,
    ClauseType,
    Divergence,
    DivergenceType,
    Entity,
    Triple,
)
from depver.comparison.similarity import entity_similarity

# Reporting / attribution verbs (English + French)
REPORTING_VERBS = frozenset({
    "say", "report", "confirm", "deny", "announce", "claim",
    "state", "tell", "declare", "assert", "argue", "suggest",
    "reveal", "disclose", "acknowledge", "warn", "promise",
    "dire", "annoncer", "confirmer", "déclarer", "affirmer",
    "nier", "prétendre", "révéler", "indiquer", "assurer",
})

# Evaluative adjectives that signal editorialization
EVALUATIVE_ADJS = frozenset({
    "controversial", "unprecedented", "shocking", "remarkable",
    "terrible", "amazing", "incredible", "devastating", "historic",
    "alarming", "significant", "massive", "dramatic", "critical",
    "controversé", "inédit", "remarquable", "historique",
    "spectaculaire", "dramatique", "choquant",
})

# Causal markers
CAUSAL_MARKS = frozenset({
    "because", "since", "causing", "leading", "resulting",
    "therefore", "thus", "hence", "consequently",
    "parce", "car", "puisque", "causant", "entraînant",
})


def classify_divergences(alignments: list[Alignment]) -> list[Divergence]:
    """Analyze each alignment and classify any divergences found."""
    divergences: list[Divergence] = []

    for alignment in alignments:
        s = alignment.source_triple
        g = alignment.generated_triple

        if s is None and g is not None:
            divergences.append(Divergence(
                type=DivergenceType.ENTITY_HALLUCINATION,
                description=f"Ungrounded triple in generated text: {g.signature}",
                source_span="",
                generated_span=g.source_text,
            ))
            continue

        if g is None and s is not None:
            divergences.append(Divergence(
                type=DivergenceType.OMISSION,
                description=f"Source triple missing from generated text: {s.signature}",
                source_span=s.source_text,
                generated_span="",
            ))
            continue

        assert s is not None and g is not None
        _check_negation_flip(s, g, divergences)
        _check_verb_substitution(s, g, divergences)
        _check_argument_swap(s, g, divergences)
        _check_modifier_hallucination(s, g, divergences)
        _check_causal_hallucination(s, g, divergences)
        _check_attribution_shift(s, g, divergences)
        _check_temporal_shift(s, g, divergences)
        _check_editorialization(s, g, divergences)

    return divergences


def _check_negation_flip(s: Triple, g: Triple, out: list[Divergence]) -> None:
    if s.negated != g.negated:
        src_pol = "negated" if s.negated else "affirmed"
        gen_pol = "negated" if g.negated else "affirmed"
        out.append(Divergence(
            type=DivergenceType.NEGATION_FLIP,
            description=(
                f"Polarity changed: '{s.predicate_form}' ({src_pol}) "
                f"-> '{g.predicate_form}' ({gen_pol})"
            ),
            source_span=s.predicate_form,
            generated_span=g.predicate_form,
        ))


def _check_verb_substitution(s: Triple, g: Triple, out: list[Divergence]) -> None:
    if s.predicate_lemma.lower() == g.predicate_lemma.lower():
        return
    # Only flag if the argument frame is similar (same subject and/or object)
    subj_match = _entities_match(s.subject, g.subject)
    obj_match = _entities_match(s.object, g.object)
    if subj_match or obj_match:
        out.append(Divergence(
            type=DivergenceType.VERB_SUBSTITUTION,
            description=f"Predicate changed: '{s.predicate_form}' -> '{g.predicate_form}'",
            source_span=s.predicate_form,
            generated_span=g.predicate_form,
        ))


def _check_argument_swap(s: Triple, g: Triple, out: list[Divergence]) -> None:
    if not (s.subject and s.object and g.subject and g.object):
        return
    # Check if subject and object are swapped
    subj_as_obj = entity_similarity(s.subject, g.object) > 0.7
    obj_as_subj = entity_similarity(s.object, g.subject) > 0.7
    if subj_as_obj and obj_as_subj:
        out.append(Divergence(
            type=DivergenceType.ARGUMENT_SWAP,
            description=(
                f"Subject/object swapped: "
                f"'{s.subject.head_form}' <-> '{s.object.head_form}'"
            ),
            source_span=f"{s.subject.head_form} ... {s.object.head_form}",
            generated_span=f"{g.subject.head_form} ... {g.object.head_form}",
        ))


def _check_modifier_hallucination(s: Triple, g: Triple, out: list[Divergence]) -> None:
    """Check if generated entity has modifiers not present in source."""
    for s_ent, g_ent, role in [
        (s.subject, g.subject, "subject"),
        (s.object, g.object, "object"),
    ]:
        if s_ent is None or g_ent is None:
            continue
        if not _entities_match(s_ent, g_ent):
            continue
        s_mods = {m.text.lower() for m in s_ent.modifiers if m.deprel != "det"}
        g_mods = {m.text.lower() for m in g_ent.modifiers if m.deprel != "det"}
        extra = g_mods - s_mods
        if extra:
            out.append(Divergence(
                type=DivergenceType.MODIFIER_HALLUCINATION,
                description=(
                    f"Ungrounded modifier(s) on {role} "
                    f"'{g_ent.head_form}': {', '.join(extra)}"
                ),
                source_span=s_ent.head_form,
                generated_span=f"{' '.join(extra)} {g_ent.head_form}",
            ))


def _check_causal_hallucination(s: Triple, g: Triple, out: list[Divergence]) -> None:
    """Detect when generated text introduces a causal link not in source."""
    # If generated triple is an advcl with causal mark, but source is root/parataxis
    if g.clause_type == ClauseType.ADVCL and g.clause_mark:
        if g.clause_mark.lower() in CAUSAL_MARKS:
            if s.clause_type in (ClauseType.ROOT, ClauseType.PARATAXIS):
                out.append(Divergence(
                    type=DivergenceType.CAUSAL_HALLUCINATION,
                    description=(
                        f"Causal link hallucinated: '{g.clause_mark}' "
                        f"connects '{g.predicate_form}' to main clause, "
                        f"but source has independent clause"
                    ),
                    source_span=s.predicate_form,
                    generated_span=f"{g.clause_mark} {g.predicate_form}",
                ))


def _check_attribution_shift(s: Triple, g: Triple, out: list[Divergence]) -> None:
    """Different subject on a reporting verb."""
    if s.predicate_lemma.lower() not in REPORTING_VERBS:
        return
    if s.subject and g.subject:
        if not _entities_match(s.subject, g.subject):
            out.append(Divergence(
                type=DivergenceType.ATTRIBUTION_SHIFT,
                description=(
                    f"Attribution changed: '{s.subject.head_form}' -> "
                    f"'{g.subject.head_form}' for '{s.predicate_form}'"
                ),
                source_span=s.subject.head_form,
                generated_span=g.subject.head_form,
            ))


def _check_temporal_shift(s: Triple, g: Triple, out: list[Divergence]) -> None:
    """Check if temporal obliques differ."""
    temporal_cases = {"in", "on", "at", "during", "before", "after", "since", "until",
                      "en", "dans", "pendant", "depuis", "avant", "après"}
    s_temps = {
        obl.entity.head_lemma.lower()
        for obl in s.obliques
        if obl.case_marker.lower() in temporal_cases
    }
    g_temps = {
        obl.entity.head_lemma.lower()
        for obl in g.obliques
        if obl.case_marker.lower() in temporal_cases
    }
    if s_temps and g_temps and s_temps != g_temps:
        out.append(Divergence(
            type=DivergenceType.TEMPORAL_SHIFT,
            description=(
                f"Temporal reference changed: "
                f"{', '.join(s_temps)} -> {', '.join(g_temps)}"
            ),
            source_span=", ".join(s_temps),
            generated_span=", ".join(g_temps),
        ))


def _check_editorialization(s: Triple, g: Triple, out: list[Divergence]) -> None:
    """Detect evaluative adjectives added in generated text."""
    for s_ent, g_ent, role in [
        (s.subject, g.subject, "subject"),
        (s.object, g.object, "object"),
    ]:
        if s_ent is None or g_ent is None:
            continue
        if not _entities_match(s_ent, g_ent):
            continue
        s_mods = {m.text.lower() for m in s_ent.modifiers}
        for m in g_ent.modifiers:
            if m.text.lower() not in s_mods and m.text.lower() in EVALUATIVE_ADJS:
                out.append(Divergence(
                    type=DivergenceType.EDITORIALIZATION,
                    description=(
                        f"Evaluative modifier added to {role}: "
                        f"'{m.text}' on '{g_ent.head_form}'"
                    ),
                    source_span=s_ent.head_form,
                    generated_span=f"{m.text} {g_ent.head_form}",
                ))


def _entities_match(a: Entity | None, b: Entity | None) -> bool:
    """Quick check if two entities refer to the same thing."""
    if a is None or b is None:
        return a is None and b is None
    return a.head_lemma.lower() == b.head_lemma.lower()
