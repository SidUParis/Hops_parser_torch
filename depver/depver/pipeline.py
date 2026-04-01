"""End-to-end dependency-structure verification pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from hopsparser.parser import BiAffineParser
from hopsparser.deptree import DepGraph

from depver.schema import VerificationResult, Triple
from depver.extraction.triples import extract_triples
from depver.comparison.matcher import align_triples
from depver.comparison.divergence import classify_divergences
from depver.scoring.metrics import compute_scores
from depver.scoring.report import format_report, format_json_report


class DepVerifier:
    """End-to-end dependency-structure verification of LLM outputs."""

    def __init__(self, parser: BiAffineParser):
        self.parser = parser

    @classmethod
    def from_pretrained(cls, model_path: str, device: str = "cpu") -> DepVerifier:
        parser = BiAffineParser.load(Path(model_path)).to(device).eval()
        return cls(parser)

    def parse_text(self, text: str) -> list[DepGraph]:
        """Parse raw text into dependency graphs (one per line)."""
        lines = [l for l in text.strip().splitlines() if l.strip()]
        if not lines:
            return []
        with torch.inference_mode():
            return list(self.parser.parse(lines, raw=True))

    def extract(self, graphs: list[DepGraph]) -> list[Triple]:
        """Extract triples from parsed graphs."""
        triples: list[Triple] = []
        for i, g in enumerate(graphs):
            triples.extend(extract_triples(g, sentence_index=i))
        return triples

    def verify(
        self,
        source_text: str,
        generated_text: str,
        threshold: float = 0.4,
    ) -> VerificationResult:
        """Full pipeline: parse -> extract -> align -> classify -> score."""
        source_graphs = self.parse_text(source_text)
        generated_graphs = self.parse_text(generated_text)

        source_triples = self.extract(source_graphs)
        gen_triples = self.extract(generated_graphs)

        alignments = align_triples(source_triples, gen_triples, threshold=threshold)
        divergences = classify_divergences(alignments)
        scores = compute_scores(alignments, divergences)

        return VerificationResult(
            source_triples=source_triples,
            generated_triples=gen_triples,
            alignments=alignments,
            scores=scores,
            divergences=divergences,
        )

    def verify_batch(
        self,
        pairs: list[tuple[str, str]],
        threshold: float = 0.4,
    ) -> list[VerificationResult]:
        """Verify multiple source-generated pairs."""
        return [self.verify(src, gen, threshold) for src, gen in pairs]


class DepVerifierWithoutParser:
    """Verification pipeline that operates on pre-parsed CoNLL-U data.

    Use this when you want to parse externally (e.g., with Stanza or gold parses)
    and only use depver for triple extraction + comparison.
    """

    def parse_conllu(self, conllu_text: str) -> list[DepGraph]:
        """Parse CoNLL-U formatted text into DepGraphs."""
        lines = conllu_text.strip().splitlines()
        return list(DepGraph.read_conll(lines))

    def extract(self, graphs: list[DepGraph]) -> list[Triple]:
        triples: list[Triple] = []
        for i, g in enumerate(graphs):
            triples.extend(extract_triples(g, sentence_index=i))
        return triples

    def verify_from_conllu(
        self,
        source_conllu: str,
        generated_conllu: str,
        threshold: float = 0.4,
    ) -> VerificationResult:
        """Verify from pre-parsed CoNLL-U strings."""
        source_graphs = self.parse_conllu(source_conllu)
        generated_graphs = self.parse_conllu(generated_conllu)

        source_triples = self.extract(source_graphs)
        gen_triples = self.extract(generated_graphs)

        alignments = align_triples(source_triples, gen_triples, threshold=threshold)
        divergences = classify_divergences(alignments)
        scores = compute_scores(alignments, divergences)

        return VerificationResult(
            source_triples=source_triples,
            generated_triples=gen_triples,
            alignments=alignments,
            scores=scores,
            divergences=divergences,
        )
