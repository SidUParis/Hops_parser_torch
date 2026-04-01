"""Tests for triple extraction."""

from depver.extraction.triples import extract_triples
from depver.schema import ClauseType


class TestBasicExtraction:
    def test_simple_svo(self, simple_svo_graph):
        triples = extract_triples(simple_svo_graph)
        assert len(triples) == 1

        t = triples[0]
        assert t.predicate_lemma == "sign"
        assert t.subject is not None
        assert t.subject.head_lemma == "Macron"
        assert t.object is not None
        assert t.object.head_lemma == "bill"
        assert not t.negated
        assert t.voice == "active"

    def test_negation(self, negated_graph):
        triples = extract_triples(negated_graph)
        assert len(triples) == 1

        t = triples[0]
        assert t.predicate_lemma == "sign"
        assert t.negated is True
        assert t.subject is not None
        assert t.subject.head_lemma == "Macron"

    def test_passive(self, passive_graph):
        triples = extract_triples(passive_graph)
        assert len(triples) == 1

        t = triples[0]
        assert t.predicate_lemma == "sign"
        assert t.voice == "passive"
        assert t.subject is not None
        assert t.subject.head_lemma == "bill"
        # "by Macron" should be an oblique
        assert len(t.obliques) == 1
        assert t.obliques[0].entity.head_lemma == "Macron"

    def test_with_oblique(self, with_oblique_graph):
        triples = extract_triples(with_oblique_graph)
        assert len(triples) == 1

        t = triples[0]
        assert len(t.obliques) == 1
        assert t.obliques[0].case_marker == "in"
        assert t.obliques[0].entity.head_lemma == "March"


class TestModifiers:
    def test_modifiers_on_object(self, with_modifier_graph):
        triples = extract_triples(with_modifier_graph)
        assert len(triples) == 1

        t = triples[0]
        assert t.object is not None
        assert t.object.head_lemma == "bill"
        mod_texts = {m.text for m in t.object.modifiers if m.deprel != "det"}
        assert "controversial" in mod_texts
        assert "climate" in mod_texts


class TestClauseTypes:
    def test_advcl_clause_type(self, causal_advcl_graph):
        triples = extract_triples(causal_advcl_graph)
        # Should find 2 predicates: "declined" (root) and "resign" (advcl)
        assert len(triples) == 2

        root_triple = next(t for t in triples if t.clause_type == ClauseType.ROOT)
        assert root_triple.predicate_lemma == "decline"

        advcl_triple = next(t for t in triples if t.clause_type == ClauseType.ADVCL)
        assert advcl_triple.predicate_lemma == "resign"
        assert advcl_triple.clause_mark == "causing"

    def test_independent_clauses(self, two_independent_clauses_graph):
        g1, g2 = two_independent_clauses_graph
        t1 = extract_triples(g1)
        t2 = extract_triples(g2)
        assert len(t1) == 1
        assert len(t2) == 1
        assert t1[0].clause_type == ClauseType.ROOT
        assert t2[0].clause_type == ClauseType.ROOT


class TestSignature:
    def test_signature_format(self, simple_svo_graph):
        triples = extract_triples(simple_svo_graph)
        t = triples[0]
        assert t.signature == "sign(Macron, bill)"

    def test_negated_signature(self, negated_graph):
        triples = extract_triples(negated_graph)
        t = triples[0]
        assert t.signature == "!sign(Macron, bill)"
