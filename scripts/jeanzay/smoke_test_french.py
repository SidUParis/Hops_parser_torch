#!/usr/bin/env python3
"""Quick French smoke test for the DepVer pipeline.

Validates end-to-end: load parser → parse French text → extract triples → compare → report.
No GPU required (runs on CPU, slow but works).

Usage:
    python scripts/jeanzay/smoke_test_french.py --model-path /path/to/UD_French-GSD-camembert
    python scripts/jeanzay/smoke_test_french.py --model-path /path/to/UD_French-GSD-camembert --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from depver.pipeline import DepVerifier
from depver.scoring.report import format_report


# French test pairs: (source, generated, expected_issue)
TEST_PAIRS = [
    # 1. Faithful — no divergence expected
    (
        "Macron a signé le projet de loi sur le climat.",
        "Macron a signé le projet de loi sur le climat.",
        "exact match — should have no divergences",
    ),
    # 2. Verb substitution — signé → rejeté
    (
        "Macron a signé le projet de loi.",
        "Macron a rejeté le projet de loi.",
        "verb substitution: signé → rejeté",
    ),
    # 3. Negation flip — a confirmé → n'a pas confirmé
    (
        "Le ministre a confirmé la réforme.",
        "Le ministre n'a pas confirmé la réforme.",
        "negation flip",
    ),
    # 4. Argument swap
    (
        "La France a battu le Brésil.",
        "Le Brésil a battu la France.",
        "argument swap",
    ),
    # 5. Extra modifier (hallucinated detail)
    (
        "Le gouvernement a annoncé un plan.",
        "Le gouvernement a annoncé un plan controversé.",
        "modifier hallucination: controversé",
    ),
]


def main():
    parser = argparse.ArgumentParser(description="French smoke test for DepVer")
    parser.add_argument("--model-path", type=str, required=True, help="Path to French hopsparser model")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    print("=" * 60)
    print("  DepVer French Smoke Test")
    print("=" * 60)
    print(f"Model:  {args.model_path}")
    print(f"Device: {args.device}")
    print()

    # Step 1: Load model
    print("Loading hopsparser model...")
    try:
        verifier = DepVerifier.from_pretrained(args.model_path, device=args.device)
        print("  OK — model loaded.\n")
    except Exception as e:
        print(f"  FAILED: {e}")
        sys.exit(1)

    # Step 2: Test parsing
    print("Testing parse...")
    try:
        graphs = verifier.parse_text("Le chat dort sur le tapis.")
        assert len(graphs) == 1, f"Expected 1 graph, got {len(graphs)}"
        g = graphs[0]
        print(f"  OK — parsed {len(g.nodes)} tokens:")
        for n in g.nodes:
            print(f"    {n.identifier}\t{n.form}\t{n.upos}\t→ {n.head}\t{n.deprel}")
        print()
    except Exception as e:
        print(f"  FAILED: {e}")
        sys.exit(1)

    # Step 3: Test triple extraction
    print("Testing triple extraction...")
    try:
        triples = verifier.extract(graphs)
        assert len(triples) >= 1, "No triples extracted"
        for t in triples:
            print(f"  {t.signature}")
        print()
    except Exception as e:
        print(f"  FAILED: {e}")
        sys.exit(1)

    # Step 4: Run all test pairs
    print("Running verification on test pairs...")
    print()
    passed = 0
    failed = 0

    for i, (source, generated, description) in enumerate(TEST_PAIRS, 1):
        print(f"--- Test {i}: {description} ---")
        print(f"  Source:    {source}")
        print(f"  Generated: {generated}")

        try:
            result = verifier.verify(source, generated)
            print(f"  Factuality score: {result.scores.factuality_score:.3f}")
            print(f"  Triples (source): {len(result.source_triples)}")
            print(f"  Triples (gen):    {len(result.generated_triples)}")

            if result.divergences:
                for d in result.divergences:
                    print(f"  Divergence: [{d.type.severity.upper()}] {d.type.value}: {d.description}")
            else:
                print("  No divergences detected.")

            passed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

        print()

    # Step 5: Print one full report
    print("=" * 60)
    print("  Full report for test pair 2 (verb substitution)")
    print("=" * 60)
    result = verifier.verify(TEST_PAIRS[1][0], TEST_PAIRS[1][1])
    print(format_report(result))

    # Summary
    print()
    print("=" * 60)
    print(f"  SUMMARY: {passed}/{passed + failed} tests passed")
    if failed == 0:
        print("  Pipeline is working. Ready for evaluation.")
    else:
        print(f"  {failed} test(s) had errors — check output above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
