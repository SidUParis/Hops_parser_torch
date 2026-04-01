#!/usr/bin/env python3
"""Download and prepare evaluation datasets for DepVer experiments.

Datasets:
  - FRANK (Pagnoni et al., 2021): Fine-grained factuality annotations for CNN/DM summaries
  - AggreFact (Tang et al., 2023): Aggregated factuality benchmark

Usage:
    python experiments/prepare_data.py --output-dir data/
    python experiments/prepare_data.py --dataset frank --output-dir data/
    python experiments/prepare_data.py --dataset aggrefact --output-dir data/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def prepare_frank(output_dir: Path) -> None:
    """Download and prepare FRANK dataset.

    FRANK provides human annotations of factual errors in CNN/DM and XSum summaries,
    categorized by error type (semantic frame errors, discourse errors, etc.).

    Source: https://github.com/artidoro/frank
    """
    from datasets import load_dataset

    print("Loading FRANK dataset...")
    # FRANK is available via the artidoro/frank HuggingFace dataset
    # If not on HF, we fall back to cloning the repo
    try:
        ds = load_dataset("rbhatt/frank")
    except Exception:
        print("FRANK not on HuggingFace, trying GitHub...")
        _download_frank_github(output_dir)
        return

    frank_dir = output_dir / "frank"
    frank_dir.mkdir(parents=True, exist_ok=True)

    # Convert to our JSONL format
    for split_name in ds:
        split = ds[split_name]
        out_path = frank_dir / f"{split_name}.jsonl"
        with open(out_path, "w") as f:
            for i, row in enumerate(split):
                record = {
                    "id": f"frank_{split_name}_{i}",
                    "source": row.get("article", row.get("document", "")),
                    "generated": row.get("summary", ""),
                    "label": row.get("label", None),
                    "annotations": row.get("annotations", []),
                }
                f.write(json.dumps(record) + "\n")
        print(f"  Wrote {len(split)} examples to {out_path}")

    print(f"FRANK dataset saved to {frank_dir}")


def _download_frank_github(output_dir: Path) -> None:
    """Fallback: download FRANK from GitHub."""
    import subprocess

    frank_dir = output_dir / "frank"
    frank_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = output_dir / "_frank_repo"
    if not repo_dir.exists():
        print("  Cloning FRANK repository...")
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/artidoro/frank.git",
             str(repo_dir)],
            check=True,
        )

    # FRANK data is in JSON files in the repo
    data_dir = repo_dir / "data"
    if not data_dir.exists():
        # Try alternative structure
        data_dir = repo_dir

    for json_file in sorted(data_dir.glob("*.json")):
        with open(json_file) as f:
            data = json.load(f)

        out_path = frank_dir / f"{json_file.stem}.jsonl"
        with open(out_path, "w") as f:
            for i, row in enumerate(data if isinstance(data, list) else [data]):
                record = {
                    "id": f"frank_{json_file.stem}_{i}",
                    "source": row.get("article", row.get("document", "")),
                    "generated": row.get("summary", ""),
                    "label": row.get("label", None),
                    "annotations": row.get("annotations", []),
                }
                f.write(json.dumps(record) + "\n")
        print(f"  Converted {json_file.name} -> {out_path}")

    print(f"FRANK dataset saved to {frank_dir}")


def prepare_aggrefact(output_dir: Path) -> None:
    """Download and prepare AggreFact benchmark.

    AggreFact unifies 9 factuality datasets (CNN/DM + XSum).
    Source: https://github.com/Liyan06/AggreFact
    """
    from datasets import load_dataset

    print("Loading AggreFact dataset...")
    agg_dir = output_dir / "aggrefact"
    agg_dir.mkdir(parents=True, exist_ok=True)

    try:
        ds = load_dataset("Liyan/AggreFact")
    except Exception:
        print("AggreFact not on HuggingFace, trying GitHub...")
        _download_aggrefact_github(output_dir)
        return

    for split_name in ds:
        split = ds[split_name]
        out_path = agg_dir / f"{split_name}.jsonl"
        with open(out_path, "w") as f:
            for i, row in enumerate(split):
                record = {
                    "id": f"aggrefact_{split_name}_{i}",
                    "source": row.get("doc", row.get("article", "")),
                    "generated": row.get("claim", row.get("summary", "")),
                    "label": row.get("label", None),
                    "dataset_origin": row.get("origin", ""),
                    "model_origin": row.get("model", ""),
                }
                f.write(json.dumps(record) + "\n")
        print(f"  Wrote {len(split)} examples to {out_path}")

    print(f"AggreFact dataset saved to {agg_dir}")


def _download_aggrefact_github(output_dir: Path) -> None:
    """Fallback: download AggreFact from GitHub."""
    import subprocess

    agg_dir = output_dir / "aggrefact"
    agg_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = output_dir / "_aggrefact_repo"
    if not repo_dir.exists():
        print("  Cloning AggreFact repository...")
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/Liyan06/AggreFact.git",
             str(repo_dir)],
            check=True,
        )

    data_dir = repo_dir / "data"
    if not data_dir.exists():
        data_dir = repo_dir

    for json_file in sorted(data_dir.glob("*.json")):
        with open(json_file) as f:
            data = json.load(f)

        out_path = agg_dir / f"{json_file.stem}.jsonl"
        with open(out_path, "w") as f:
            rows = data if isinstance(data, list) else [data]
            for i, row in enumerate(rows):
                record = {
                    "id": f"aggrefact_{json_file.stem}_{i}",
                    "source": row.get("doc", row.get("article", "")),
                    "generated": row.get("claim", row.get("summary", "")),
                    "label": row.get("label", None),
                }
                f.write(json.dumps(record) + "\n")
        print(f"  Converted {json_file.name} -> {out_path}")

    print(f"AggreFact dataset saved to {agg_dir}")


def prepare_summeval(output_dir: Path) -> None:
    """Download SummEval for correlation analysis."""
    from datasets import load_dataset

    print("Loading SummEval dataset...")
    se_dir = output_dir / "summeval"
    se_dir.mkdir(parents=True, exist_ok=True)

    try:
        ds = load_dataset("mteb/summeval")
        split = ds[list(ds.keys())[0]]
        out_path = se_dir / "summeval.jsonl"
        with open(out_path, "w") as f:
            for i, row in enumerate(split):
                record = {
                    "id": f"summeval_{i}",
                    "source": row.get("text", ""),
                    "generated": row.get("machine_summaries", [""])[0] if row.get("machine_summaries") else "",
                    "human_scores": {
                        "consistency": row.get("consistency", None),
                        "relevance": row.get("relevance", None),
                        "fluency": row.get("fluency", None),
                        "coherence": row.get("coherence", None),
                    },
                }
                f.write(json.dumps(record) + "\n")
        print(f"  Wrote {len(split)} examples to {out_path}")
    except Exception as e:
        print(f"  Warning: Could not load SummEval: {e}")

    print(f"SummEval dataset saved to {se_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare evaluation datasets for DepVer")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data"),
        help="Output directory (default: data/)",
    )
    parser.add_argument(
        "--dataset", choices=["frank", "aggrefact", "summeval", "all"],
        default="all", help="Which dataset to prepare (default: all)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("frank", "all"):
        prepare_frank(args.output_dir)
    if args.dataset in ("aggrefact", "all"):
        prepare_aggrefact(args.output_dir)
    if args.dataset in ("summeval", "all"):
        prepare_summeval(args.output_dir)

    print("\nDone. Data saved to:", args.output_dir)


if __name__ == "__main__":
    main()
