"""Entity extraction from dependency subtrees."""

from __future__ import annotations

from hopsparser.deptree import DepGraph, DepNode
from depver.schema import Entity, Modifier
from depver.extraction.walkers import get_children, subtree_ids, base_deprel


def build_entity(graph: DepGraph, head: DepNode) -> Entity:
    """Build an Entity from a noun and its modifier subtree."""
    children = get_children(graph, head.identifier)
    modifiers: list[Modifier] = []
    nmod_chain: list[Entity] = []

    for child in children:
        rel = base_deprel(child.deprel)
        if rel in ("amod", "nummod", "det"):
            modifiers.append(
                Modifier(
                    text=child.lemma or child.form,
                    deprel=rel,
                    node_id=child.identifier,
                )
            )
        elif rel == "nmod":
            nmod_chain.append(build_entity(graph, child))
        elif rel == "flat" or rel == "appos":
            # Flat names ("Tim Cook") or appositions — add as modifier
            modifiers.append(
                Modifier(
                    text=child.lemma or child.form,
                    deprel=rel,
                    node_id=child.identifier,
                )
            )

    return Entity(
        head_lemma=head.lemma or head.form,
        head_form=head.form,
        head_upos=head.upos or "NOUN",
        modifiers=tuple(modifiers),
        nmod_chain=tuple(nmod_chain),
        node_ids=tuple(subtree_ids(graph, head.identifier)),
    )
