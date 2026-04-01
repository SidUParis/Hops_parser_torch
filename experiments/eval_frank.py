#!/usr/bin/env python3
"""Evaluate DepVer on the FRANK benchmark.

Usage:
    python experiments/eval_frank.py \
        --model-path models/en_ewt \
        --data-dir data/frank \
        --output-dir results/frank \
        --device cuda

Expects FRANK data in JSONL format (from prepare_data.py).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from depver.pipeline import DepVerifier
from depver.scoring.report import format_json_report


def load_data(data_dir: Path) -> list[dict]:
    """Load FRANK data from JSONL files."""
    records = []
    for jsonl_file in sorted(data_dir.glob("*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    print(f"Loaded {len(records)} examples from {data_dir}")
    return records


def run_depver(
    verifier: DepVerifier,
    records: list[dict],
    threshold: float,
) -> list[dict]:
    """Run DepVer on all records and collect results."""
    results = []
    t0 = time.time()

    for i, record in enumerate(records):
        source = record.get("source", "")
        generated = record.get("generated", "")

        if not source.strip() or not generated.strip():
            continue

        try:
            vr = verifier.verify(source, generated, threshold=threshold)
            report = format_json_report(vr)
            report["id"] = record.get("id", i)
            report["gold_label"] = record.get("label", None)
            results.append(report)
        except Exception as e:
            print(f"  Error on record {i}: {e}", file=sys.stderr)
            continue

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  Processed {i + 1}/{len(records)} ({rate:.1f} ex/s)")

    elapsed = time.time() - t0
    print(f"Processed {len(results)} examples in {elapsed:.1f}s")
    return results


def evaluate(results: list[dict], output_dir: Path) -> None:
    """Compute evaluation metrics and save results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw results
    raw_path = output_dir / "depver_results.jsonl"
    with open(raw_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Raw results saved to {raw_path}")

    # Filter records with gold labels
    labeled = [r for r in results if r.get("gold_label") is not None]
    if not labeled:
        print("No gold labels found — skipping metric computation.")
        return

    # Binary classification: factual (1) vs non-factual (0)
    # Convention: label=1 means factual/correct, label=0 means non-factual
    gold = []
    predicted = []
    scores_list = []

    for r in labeled:
        gl = r["gold_label"]
        # Normalize label to binary
        if isinstance(gl, str):
            gl = 1 if gl.lower() in ("factual", "correct", "1", "true") else 0
        else:
            gl = int(gl)
        gold.append(gl)

        # DepVer prediction: factuality_score > 0.5 => factual
        fs = r["scores"]["factuality_score"]
        scores_list.append(fs)
        predicted.append(1 if fs > 0.5 else 0)

    # Metrics
    p, r, f1, _ = precision_recall_fscore_support(gold, predicted, average="binary", zero_division=0)
    ba = balanced_accuracy_score(gold, predicted)

    try:
        auc = roc_auc_score(gold, scores_list)
    except ValueError:
        auc = float("nan")

    metrics = {
        "num_examples": len(labeled),
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "balanced_accuracy": round(ba, 4),
        "auc_roc": round(auc, 4) if not pd.isna(auc) else None,
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== FRANK Evaluation Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\nMetrics saved to {metrics_path}")

    # Per-divergence-type breakdown
    type_counts: dict[str, int] = {}
    for r in results:
        for d in r.get("divergences", []):
            t = d["type"]
            type_counts[t] = type_counts.get(t, 0) + 1

    if type_counts:
        type_path = output_dir / "divergence_types.json"
        with open(type_path, "w") as f:
            json.dump(type_counts, f, indent=2)
        print(f"\nDivergence type counts saved to {type_path}")
        for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"  {t}: {c}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate DepVer on FRANK")
    parser.add_argument("--model-path", type=str, required=True, help="hopsparser model path")
    parser.add_argument("--data-dir", type=Path, default=Path("data/frank"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/frank"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--max-examples", type=int, default=None, help="Limit examples for debugging")
    args = parser.parse_args()

    print("Loading DepVer model...")
    verifier = DepVerifier.from_pretrained(args.model_path, device=args.device)

    print("Loading FRANK data...")
    records = load_data(args.data_dir)
    if args.max_examples:
        records = records[:args.max_examples]

    print("Running DepVer...")
    results = run_depver(verifier, records, args.threshold)

    print("Evaluating...")
    evaluate(results, args.output_dir)


if __name__ == "__main__":
    main()
