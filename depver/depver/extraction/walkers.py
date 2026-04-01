"""Tree-walking utilities for dependency graphs."""

from __future__ import annotations

from hopsparser.deptree import DepGraph, DepNode


def get_children(graph: DepGraph, node_id: int) -> list[DepNode]:
    """Get all direct children of a node in the dependency tree."""
    return [n for n in graph.nodes if n.head == node_id]


def subtree_ids(graph: DepGraph, node_id: int) -> list[int]:
    """Collect all node IDs in the subtree rooted at node_id (inclusive)."""
    ids = [node_id]
    for child in get_children(graph, node_id):
        ids.extend(subtree_ids(graph, child.identifier))
    return ids


def subtree_text(graph: DepGraph, node_id: int) -> str:
    """Reconstruct the surface text of a subtree, in original word order."""
    ids = sorted(subtree_ids(graph, node_id))
    words = graph.words  # index 0 = <root>
    return " ".join(words[i] for i in ids if i < len(words))


def find_node(graph: DepGraph, node_id: int) -> DepNode | None:
    """Find a node by its identifier."""
    for n in graph.nodes:
        if n.identifier == node_id:
            return n
    return None


def base_deprel(deprel: str | None) -> str:
    """Strip subtype from deprel: 'nsubj:pass' -> 'nsubj'."""
    if deprel is None:
        return ""
    return deprel.split(":")[0]
