"""Tests for the verification pipeline (unit-level, no parser needed)."""

from depver.pipeline import DepVerifierWithoutParser


CONLLU_SOURCE = """\
# text = Macron signed the bill.
1\tMacron\tMacron\tPROPN\t_\t_\t2\tnsubj\t_\t_
2\tsigned\tsign\tVERB\t_\t_\t0\troot\t_\t_
3\tthe\tthe\tDET\t_\t_\t4\tdet\t_\t_
4\tbill\tbill\tNOUN\t_\t_\t2\tobj\t_\t_
5\t.\t.\tPUNCT\t_\t_\t2\tpunct\t_\t_

"""

CONLLU_GENERATED_GOOD = """\
# text = Macron signed the bill.
1\tMacron\tMacron\tPROPN\t_\t_\t2\tnsubj\t_\t_
2\tsigned\tsign\tVERB\t_\t_\t0\troot\t_\t_
3\tthe\tthe\tDET\t_\t_\t4\tdet\t_\t_
4\tbill\tbill\tNOUN\t_\t_\t2\tobj\t_\t_
5\t.\t.\tPUNCT\t_\t_\t2\tpunct\t_\t_

"""

CONLLU_GENERATED_NEGATED = """\
# text = Macron did not sign the bill.
1\tMacron\tMacron\tPROPN\t_\t_\t4\tnsubj\t_\t_
2\tdid\tdo\tAUX\t_\t_\t4\taux\t_\t_
3\tnot\tnot\tPART\t_\t_\t4\tadvmod\t_\t_
4\tsign\tsign\tVERB\t_\t_\t0\troot\t_\t_
5\tthe\tthe\tDET\t_\t_\t6\tdet\t_\t_
6\tbill\tbill\tNOUN\t_\t_\t4\tobj\t_\t_
7\t.\t.\tPUNCT\t_\t_\t4\tpunct\t_\t_

"""


class TestDepVerifierWithoutParser:
    def test_perfect_match(self):
        verifier = DepVerifierWithoutParser()
        result = verifier.verify_from_conllu(CONLLU_SOURCE, CONLLU_GENERATED_GOOD)

        assert result.scores.triple_f1 == 1.0
        assert len(result.divergences) == 0

    def test_negation_detected(self):
        verifier = DepVerifierWithoutParser()
        result = verifier.verify_from_conllu(CONLLU_SOURCE, CONLLU_GENERATED_NEGATED)

        assert result.scores.factuality_score < 1.0
        assert any(d.type.value == "negation_flip" for d in result.divergences)

    def test_triple_extraction(self):
        verifier = DepVerifierWithoutParser()
        graphs = verifier.parse_conllu(CONLLU_SOURCE)
        triples = verifier.extract(graphs)

        assert len(triples) == 1
        assert triples[0].predicate_lemma == "sign"
