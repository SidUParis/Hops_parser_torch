#!/usr/bin/env python3
"""Run all experiments end-to-end.

Usage:
    python experiments/run_all.py \
        --model-path models/en_ewt \
        --device cuda \
        --output-dir results

This script runs:
  1. DepVer on FRANK
  2. DepVer on AggreFact
  3. BERTScore baseline on FRANK
  4. BERTScore baseline on AggreFact
  5. Comparison summary
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str], desc: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print(f"  WARNING: {desc} exited with code {result.returncode}", file=sys.stderr)


def compare_results(output_dir: Path) -> None:
    """Load all metrics.json files and print comparison."""
    print(f"\n{'='*60}")
    print("  COMPARISON SUMMARY")
    print(f"{'='*60}\n")

    metrics_files = sorted(output_dir.rglob("metrics.json"))
    if not metrics_files:
        print("No metrics files found.")
        return

    rows = []
    for mf in metrics_files:
        with open(mf) as f:
            m = json.load(f)
        m["experiment"] = str(mf.parent.relative_to(output_dir))
        rows.append(m)

    # Print table
    header = f"{'Experiment':<35} {'F1':>8} {'BAcc':>8} {'AUC':>8} {'N':>6}"
    print(header)
    print("-" * len(header))
    for r in rows:
        exp = r.get("experiment", "?")
        f1 = r.get("f1", "?")
        ba = r.get("balanced_accuracy", "?")
        auc = r.get("auc_roc", "?")
        n = r.get("num_examples", "?")
        print(f"{exp:<35} {f1:>8} {ba:>8} {auc:>8} {n:>6}")

    # Save comparison
    comp_path = output_dir / "comparison.json"
    with open(comp_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nComparison saved to {comp_path}")


def main():
    p = argparse.ArgumentParser(description="Run all DepVer experiments")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-examples", type=int, default=None)
    args = p.parse_args()

    py = sys.executable
    max_ex = ["--max-examples", str(args.max_examples)] if args.max_examples else []

    # 1. DepVer on FRANK
    if (args.data_dir / "frank").exists():
        run_cmd([
            py, "experiments/eval_frank.py",
            "--model-path", args.model_path,
            "--data-dir", str(args.data_dir / "frank"),
            "--output-dir", str(args.output_dir / "frank_depver"),
            "--device", args.device,
            *max_ex,
        ], "DepVer on FRANK")

        # BERTScore baseline on FRANK
        run_cmd([
            py, "experiments/baselines/run_bertscore.py",
            "--data-dir", str(args.data_dir / "frank"),
            "--output-dir", str(args.output_dir / "frank_bertscore"),
            *max_ex,
        ], "BERTScore on FRANK")
    else:
        print(f"Skipping FRANK (no data at {args.data_dir / 'frank'})")

    # 2. DepVer on AggreFact
    if (args.data_dir / "aggrefact").exists():
        run_cmd([
            py, "experiments/eval_aggrefact.py",
            "--model-path", args.model_path,
            "--data-dir", str(args.data_dir / "aggrefact"),
            "--output-dir", str(args.output_dir / "aggrefact_depver"),
            "--device", args.device,
            *max_ex,
        ], "DepVer on AggreFact")

        # BERTScore baseline on AggreFact
        run_cmd([
            py, "experiments/baselines/run_bertscore.py",
            "--data-dir", str(args.data_dir / "aggrefact"),
            "--output-dir", str(args.output_dir / "aggrefact_bertscore"),
            *max_ex,
        ], "BERTScore on AggreFact")
    else:
        print(f"Skipping AggreFact (no data at {args.data_dir / 'aggrefact'})")

    # Comparison
    compare_results(args.output_dir)


if __name__ == "__main__":
    main()
