"""Triple extraction from parsed dependency graphs."""

from __future__ import annotations

from hopsparser.deptree import DepGraph, DepNode
from depver.schema import Triple, Oblique, ClauseType
from depver.extraction.entities import build_entity
from depver.extraction.walkers import get_children, base_deprel


# Negation lemmas (English + French)
_NEG_LEMMAS = frozenset({
    "not", "never", "no", "nor", "neither", "n't",
    "ne", "pas", "jamais", "aucun", "rien", "guère",
    "nicht", "kein", "nie", "niemals",  # German
})


def extract_triples(graph: DepGraph, sentence_index: int = 0) -> list[Triple]:
    """Extract predicate-argument triples from a parsed dependency graph.

    Walks all nodes, identifies predicates (VERB, copular AUX),
    and collects their arguments by deprel type.
    """
    triples: list[Triple] = []
    for node in graph.nodes:
        if _is_predicate(node):
            triple = _build_triple(graph, node, sentence_index)
            triples.append(triple)
    return triples


def _is_predicate(node: DepNode) -> bool:
    """A node is a predicate if it's a VERB, or an AUX heading a clause."""
    if node.upos == "VERB":
        return True
    if node.upos == "AUX" and base_deprel(node.deprel) in (
        "root", "ccomp", "xcomp", "advcl", "parataxis",
    ):
        return True
    # Nominal predicates in copular constructions: "He is a doctor"
    # The predicate is technically the noun/adj, not the copula
    # We handle this by treating the copula's head as predicate if it has nsubj
    return False


def _build_triple(graph: DepGraph, predicate: DepNode, sentence_index: int) -> Triple:
    """Collect arguments of a predicate from its dependents."""
    children = get_children(graph, predicate.identifier)

    subject = None
    obj = None
    obliques: list[Oblique] = []
    negated = False
    modality = None
    voice = "active"
    clause_mark = None

    for child in children:
        rel = base_deprel(child.deprel)
        full_rel = child.deprel or ""

        if rel == "nsubj":
            subject = build_entity(graph, child)
            if "pass" in full_rel:
                voice = "passive"
        elif rel == "obj" or rel == "iobj":
            if obj is None:  # prefer first obj
                obj = build_entity(graph, child)
        elif rel == "obl":
            case = _find_case(graph, child)
            obliques.append(Oblique(case_marker=case, entity=build_entity(graph, child)))
        elif rel == "advmod" and _is_negation(child):
            negated = True
        elif rel == "aux":
            # Modal auxiliaries
            if child.lemma and child.lemma.lower() in (
                "can", "could", "may", "might", "must", "shall", "should", "will", "would",
                "pouvoir", "devoir", "vouloir", "falloir",  # French
                "können", "müssen", "sollen", "dürfen", "wollen",  # German
            ):
                modality = child.lemma.lower()
        elif rel == "mark":
            clause_mark = child.form.lower()
        elif rel == "expl":
            # Expletive subjects (French "il", English "it"/"there") — skip
            pass
        # Handle passive auxiliary
        if rel == "aux" and "pass" in full_rel:
            voice = "passive"

    # Handle conj: if this predicate is a conjunct, inherit subject from head
    if base_deprel(predicate.deprel) == "conj" and subject is None:
        head_node = _find_node_by_id(graph, predicate.head)
        if head_node is not None:
            for sibling in get_children(graph, head_node.identifier):
                if base_deprel(sibling.deprel) == "nsubj":
                    subject = build_entity(graph, sibling)
                    break

    clause_type = _determine_clause_type(predicate)
    source_text = " ".join(graph.words[1:])  # skip <root>

    return Triple(
        predicate_lemma=predicate.lemma or predicate.form,
        predicate_form=predicate.form,
        predicate_upos=predicate.upos or "VERB",
        predicate_id=predicate.identifier,
        subject=subject,
        object=obj,
        obliques=tuple(obliques),
        negated=negated,
        modality=modality,
        voice=voice,
        clause_type=clause_type,
        clause_mark=clause_mark,
        sentence_index=sentence_index,
        source_text=source_text,
    )


def _find_case(graph: DepGraph, node: DepNode) -> str:
    """Find the case marker (preposition) for an oblique."""
    for child in get_children(graph, node.identifier):
        if base_deprel(child.deprel) == "case":
            return child.form.lower()
    return ""


def _is_negation(node: DepNode) -> bool:
    """Check if an advmod is a negation word."""
    lemma = (node.lemma or node.form).lower()
    return lemma in _NEG_LEMMAS


def _determine_clause_type(node: DepNode) -> ClauseType:
    """Determine the clause type from the node's deprel."""
    rel = base_deprel(node.deprel)
    mapping = {
        "root": ClauseType.ROOT,
        "advcl": ClauseType.ADVCL,
        "xcomp": ClauseType.XCOMP,
        "ccomp": ClauseType.CCOMP,
        "relcl": ClauseType.RELCL,
        "parataxis": ClauseType.PARATAXIS,
    }
    return mapping.get(rel, ClauseType.ROOT)


def _find_node_by_id(graph: DepGraph, node_id: int | None) -> DepNode | None:
    """Find a node by its identifier."""
    if node_id is None:
        return None
    for n in graph.nodes:
        if n.identifier == node_id:
            return n
    return None
