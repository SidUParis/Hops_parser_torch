"""Tests for triple comparison and divergence detection."""

from depver.extraction.triples import extract_triples
from depver.comparison.matcher import align_triples, triple_similarity
from depver.comparison.divergence import classify_divergences
from depver.schema import DivergenceType
from tests.conftest import make_graph


class TestTripleSimilarity:
    def test_identical_triples(self, simple_svo_graph):
        triples = extract_triples(simple_svo_graph)
        assert len(triples) == 1
        score = triple_similarity(triples[0], triples[0])
        assert score == 1.0

    def test_different_predicate_same_args(self, simple_svo_graph):
        """'Macron signed the bill' vs 'Macron vetoed the bill'."""
        source_triples = extract_triples(simple_svo_graph)

        vetoed_graph = make_graph([
            (1, "Macron", "Macron", "PROPN", 2, "nsubj"),
            (2, "vetoed", "veto", "VERB", 0, "root"),
            (3, "the", "the", "DET", 4, "det"),
            (4, "bill", "bill", "NOUN", 2, "obj"),
            (5, ".", ".", "PUNCT", 2, "punct"),
        ])
        gen_triples = extract_triples(vetoed_graph)

        score = triple_similarity(gen_triples[0], source_triples[0])
        # Should be > 0 (args match) but < 1.0 (predicate differs)
        assert 0.3 < score < 1.0


class TestAlignment:
    def test_perfect_alignment(self, simple_svo_graph):
        triples = extract_triples(simple_svo_graph)
        alignments = align_triples(triples, triples)
        assert len(alignments) == 1
        assert alignments[0].source_triple is not None
        assert alignments[0].generated_triple is not None
        assert alignments[0].similarity_score == 1.0

    def test_hallucination_detection(self, simple_svo_graph):
        """Generated has a triple with no source match."""
        source_triples = extract_triples(simple_svo_graph)

        extra_graph = make_graph([
            (1, "Prices", "price", "NOUN", 2, "nsubj"),
            (2, "rose", "rise", "VERB", 0, "root"),
            (3, ".", ".", "PUNCT", 2, "punct"),
        ])
        gen_triples = extract_triples(simple_svo_graph) + extract_triples(extra_graph)

        alignments = align_triples(source_triples, gen_triples)
        hallucinations = [a for a in alignments if a.source_triple is None]
        assert len(hallucinations) == 1

    def test_omission_detection(self, simple_svo_graph):
        """Source has a triple that generated text doesn't."""
        extra_graph = make_graph([
            (1, "Prices", "price", "NOUN", 2, "nsubj"),
            (2, "rose", "rise", "VERB", 0, "root"),
            (3, ".", ".", "PUNCT", 2, "punct"),
        ])
        source_triples = extract_triples(simple_svo_graph) + extract_triples(extra_graph)
        gen_triples = extract_triples(simple_svo_graph)

        alignments = align_triples(source_triples, gen_triples)
        omissions = [a for a in alignments if a.generated_triple is None]
        assert len(omissions) == 1

    def test_empty_inputs(self):
        assert align_triples([], []) == []


class TestDivergenceClassification:
    def test_negation_flip(self, simple_svo_graph, negated_graph):
        source = extract_triples(simple_svo_graph)
        gen = extract_triples(negated_graph)
        alignments = align_triples(source, gen)
        divergences = classify_divergences(alignments)

        neg_flips = [d for d in divergences if d.type == DivergenceType.NEGATION_FLIP]
        assert len(neg_flips) == 1

    def test_verb_substitution(self, simple_svo_graph):
        gen_graph = make_graph([
            (1, "Macron", "Macron", "PROPN", 2, "nsubj"),
            (2, "vetoed", "veto", "VERB", 0, "root"),
            (3, "the", "the", "DET", 4, "det"),
            (4, "bill", "bill", "NOUN", 2, "obj"),
            (5, ".", ".", "PUNCT", 2, "punct"),
        ])
        source = extract_triples(simple_svo_graph)
        gen = extract_triples(gen_graph)
        alignments = align_triples(source, gen)
        divergences = classify_divergences(alignments)

        verb_subs = [d for d in divergences if d.type == DivergenceType.VERB_SUBSTITUTION]
        assert len(verb_subs) == 1
        assert "sign" in verb_subs[0].source_span.lower() or "signed" in verb_subs[0].source_span.lower()

    def test_argument_swap(self):
        source_graph = make_graph([
            (1, "Google", "Google", "PROPN", 2, "nsubj"),
            (2, "acquired", "acquire", "VERB", 0, "root"),
            (3, "YouTube", "YouTube", "PROPN", 2, "obj"),
            (4, ".", ".", "PUNCT", 2, "punct"),
        ])
        gen_graph = make_graph([
            (1, "YouTube", "YouTube", "PROPN", 2, "nsubj"),
            (2, "acquired", "acquire", "VERB", 0, "root"),
            (3, "Google", "Google", "PROPN", 2, "obj"),
            (4, ".", ".", "PUNCT", 2, "punct"),
        ])
        source = extract_triples(source_graph)
        gen = extract_triples(gen_graph)
        alignments = align_triples(source, gen)
        divergences = classify_divergences(alignments)

        swaps = [d for d in divergences if d.type == DivergenceType.ARGUMENT_SWAP]
        assert len(swaps) == 1

    def test_modifier_hallucination(self, simple_svo_graph, with_modifier_graph):
        source = extract_triples(simple_svo_graph)
        gen = extract_triples(with_modifier_graph)
        alignments = align_triples(source, gen)
        divergences = classify_divergences(alignments)

        mod_halluc = [d for d in divergences if d.type == DivergenceType.MODIFIER_HALLUCINATION]
        assert len(mod_halluc) >= 1

    def test_causal_hallucination(self, two_independent_clauses_graph, causal_advcl_graph):
        g1, g2 = two_independent_clauses_graph
        source = extract_triples(g1) + extract_triples(g2)
        gen = extract_triples(causal_advcl_graph)

        alignments = align_triples(source, gen)
        divergences = classify_divergences(alignments)

        causal = [d for d in divergences if d.type == DivergenceType.CAUSAL_HALLUCINATION]
        assert len(causal) >= 1

    def test_attribution_shift(self, reporting_verb_graph, attribution_shifted_graph):
        source = extract_triples(reporting_verb_graph)
        gen = extract_triples(attribution_shifted_graph)
        alignments = align_triples(source, gen)
        divergences = classify_divergences(alignments)

        attr_shifts = [d for d in divergences if d.type == DivergenceType.ATTRIBUTION_SHIFT]
        assert len(attr_shifts) >= 1
