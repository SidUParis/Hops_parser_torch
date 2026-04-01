#!/usr/bin/env python3
"""Run BERTScore baseline on evaluation data.

Usage:
    python experiments/baselines/run_bertscore.py \
        --data-dir data/frank \
        --output-dir results/frank_bertscore
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from bert_score import score as bert_score
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def load_data(data_dir: Path) -> list[dict]:
    records = []
    for jsonl_file in sorted(data_dir.glob("*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser(description="BERTScore baseline")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-type", default="microsoft/deberta-xlarge-mnli")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args()

    records = load_data(args.data_dir)
    if args.max_examples:
        records = records[:args.max_examples]

    sources = [r["source"] for r in records]
    generated = [r["generated"] for r in records]

    print(f"Computing BERTScore for {len(records)} pairs...")
    P, R, F1 = bert_score(
        generated, sources,
        model_type=args.model_type,
        batch_size=args.batch_size,
        verbose=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save per-example scores
    results_path = args.output_dir / "bertscore_results.jsonl"
    with open(results_path, "w") as f:
        for i, r in enumerate(records):
            out = {
                "id": r.get("id", i),
                "bertscore_p": P[i].item(),
                "bertscore_r": R[i].item(),
                "bertscore_f1": F1[i].item(),
                "gold_label": r.get("label", None),
            }
            f.write(json.dumps(out) + "\n")

    # Evaluate
    labeled = [(r, F1[i].item()) for i, r in enumerate(records) if r.get("label") is not None]
    if labeled:
        gold = []
        scores_list = []
        for r, f1_val in labeled:
            gl = r["label"]
            if isinstance(gl, str):
                gl = 1 if gl.lower() in ("factual", "correct", "1", "true", "consistent") else 0
            else:
                gl = int(gl)
            gold.append(gl)
            scores_list.append(f1_val)

        predicted = [1 if s > 0.5 else 0 for s in scores_list]
        p, r, f1_metric, _ = precision_recall_fscore_support(gold, predicted, average="binary", zero_division=0)
        ba = balanced_accuracy_score(gold, predicted)

        try:
            auc = roc_auc_score(gold, scores_list)
        except ValueError:
            auc = float("nan")

        metrics = {
            "method": "bertscore",
            "model": args.model_type,
            "num_examples": len(labeled),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1_metric, 4),
            "balanced_accuracy": round(ba, 4),
            "auc_roc": round(auc, 4) if not pd.isna(auc) else None,
        }

        metrics_path = args.output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print("\n=== BERTScore Results ===")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
