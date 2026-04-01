"""Shared fixtures for depver tests."""

from __future__ import annotations

import pytest
from hopsparser.deptree import DepGraph, DepNode, Misc


def make_node(
    identifier: int,
    form: str,
    lemma: str | None = None,
    upos: str | None = None,
    head: int | None = None,
    deprel: str | None = None,
) -> DepNode:
    """Convenience factory for DepNode."""
    return DepNode(
        identifier=identifier,
        form=form,
        lemma=lemma or form.lower(),
        upos=upos,
        xpos=None,
        feats=None,
        head=head,
        deprel=deprel,
        deps=None,
        misc=Misc(),
    )


def make_graph(nodes_spec: list[tuple]) -> DepGraph:
    """Build a DepGraph from a list of (id, form, lemma, upos, head, deprel) tuples."""
    nodes = [
        make_node(
            identifier=spec[0],
            form=spec[1],
            lemma=spec[2],
            upos=spec[3],
            head=spec[4],
            deprel=spec[5],
        )
        for spec in nodes_spec
    ]
    return DepGraph(nodes=nodes)


# --- Fixture graphs ---

@pytest.fixture
def simple_svo_graph():
    """'Macron signed the bill.'"""
    return make_graph([
        # (id, form, lemma, upos, head, deprel)
        (1, "Macron", "Macron", "PROPN", 2, "nsubj"),
        (2, "signed", "sign", "VERB", 0, "root"),
        (3, "the", "the", "DET", 4, "det"),
        (4, "bill", "bill", "NOUN", 2, "obj"),
        (5, ".", ".", "PUNCT", 2, "punct"),
    ])


@pytest.fixture
def negated_graph():
    """'Macron did not sign the bill.'"""
    return make_graph([
        (1, "Macron", "Macron", "PROPN", 4, "nsubj"),
        (2, "did", "do", "AUX", 4, "aux"),
        (3, "not", "not", "PART", 4, "advmod"),
        (4, "sign", "sign", "VERB", 0, "root"),
        (5, "the", "the", "DET", 6, "det"),
        (6, "bill", "bill", "NOUN", 4, "obj"),
        (7, ".", ".", "PUNCT", 4, "punct"),
    ])


@pytest.fixture
def passive_graph():
    """'The bill was signed by Macron.'"""
    return make_graph([
        (1, "The", "the", "DET", 2, "det"),
        (2, "bill", "bill", "NOUN", 4, "nsubj:pass"),
        (3, "was", "be", "AUX", 4, "aux:pass"),
        (4, "signed", "sign", "VERB", 0, "root"),
        (5, "by", "by", "ADP", 6, "case"),
        (6, "Macron", "Macron", "PROPN", 4, "obl"),
        (7, ".", ".", "PUNCT", 4, "punct"),
    ])


@pytest.fixture
def with_oblique_graph():
    """'Macron signed the bill in March.'"""
    return make_graph([
        (1, "Macron", "Macron", "PROPN", 2, "nsubj"),
        (2, "signed", "sign", "VERB", 0, "root"),
        (3, "the", "the", "DET", 4, "det"),
        (4, "bill", "bill", "NOUN", 2, "obj"),
        (5, "in", "in", "ADP", 6, "case"),
        (6, "March", "March", "PROPN", 2, "obl"),
        (7, ".", ".", "PUNCT", 2, "punct"),
    ])


@pytest.fixture
def with_modifier_graph():
    """'Macron signed the controversial climate bill.'"""
    return make_graph([
        (1, "Macron", "Macron", "PROPN", 2, "nsubj"),
        (2, "signed", "sign", "VERB", 0, "root"),
        (3, "the", "the", "DET", 6, "det"),
        (4, "controversial", "controversial", "ADJ", 6, "amod"),
        (5, "climate", "climate", "NOUN", 6, "amod"),
        (6, "bill", "bill", "NOUN", 2, "obj"),
        (7, ".", ".", "PUNCT", 2, "punct"),
    ])


@pytest.fixture
def causal_advcl_graph():
    """'Revenue declined, causing the CEO to resign.'"""
    return make_graph([
        (1, "Revenue", "revenue", "NOUN", 2, "nsubj"),
        (2, "declined", "decline", "VERB", 0, "root"),
        (3, ",", ",", "PUNCT", 7, "punct"),
        (4, "causing", "causing", "SCONJ", 7, "mark"),
        (5, "the", "the", "DET", 6, "det"),
        (6, "CEO", "CEO", "NOUN", 7, "nsubj"),
        (7, "resign", "resign", "VERB", 2, "advcl"),
        (8, ".", ".", "PUNCT", 2, "punct"),
    ])


@pytest.fixture
def two_independent_clauses_graph():
    """'Revenue declined. The CEO resigned.' — as two separate DepGraphs."""
    g1 = make_graph([
        (1, "Revenue", "revenue", "NOUN", 2, "nsubj"),
        (2, "declined", "decline", "VERB", 0, "root"),
        (3, ".", ".", "PUNCT", 2, "punct"),
    ])
    g2 = make_graph([
        (1, "The", "the", "DET", 2, "det"),
        (2, "CEO", "CEO", "NOUN", 3, "nsubj"),
        (3, "resigned", "resign", "VERB", 0, "root"),
        (4, ".", ".", "PUNCT", 3, "punct"),
    ])
    return [g1, g2]


@pytest.fixture
def reporting_verb_graph():
    """'Analysts expect growth.'"""
    return make_graph([
        (1, "Analysts", "analyst", "NOUN", 2, "nsubj"),
        (2, "expect", "report", "VERB", 0, "root"),
        (3, "growth", "growth", "NOUN", 2, "obj"),
        (4, ".", ".", "PUNCT", 2, "punct"),
    ])


@pytest.fixture
def attribution_shifted_graph():
    """'The CEO confirmed growth.'"""
    return make_graph([
        (1, "The", "the", "DET", 2, "det"),
        (2, "CEO", "CEO", "NOUN", 3, "nsubj"),
        (3, "confirmed", "confirm", "VERB", 0, "root"),
        (4, "growth", "growth", "NOUN", 3, "obj"),
        (5, ".", ".", "PUNCT", 3, "punct"),
    ])
