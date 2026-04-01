"""CLI entry point for depver."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, help="Path to hopsparser model")
    p.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    p.add_argument("--models-dir", default=None, help="Path to depver models (nli/, embedder/)")


def main():
    parser = argparse.ArgumentParser(
        description="DepVer: Dependency-structure verification of LLM outputs",
    )
    subparsers = parser.add_subparsers(dest="command")

    verify_parser = subparsers.add_parser("verify", help="Verify generated text against source")
    _add_common_args(verify_parser)
    verify_parser.add_argument("--source", required=True, help="Source text file")
    verify_parser.add_argument("--generated", required=True, help="Generated text file")
    verify_parser.add_argument("--threshold", type=float, default=0.4)
    verify_parser.add_argument("--format", choices=["text", "json"], default="text")

    batch_parser = subparsers.add_parser("verify-batch", help="Verify a JSONL file of pairs")
    _add_common_args(batch_parser)
    batch_parser.add_argument("--input", required=True, help="JSONL file with source/generated")
    batch_parser.add_argument("--output", required=True, help="Output JSONL file")
    batch_parser.add_argument("--threshold", type=float, default=0.4)

    extract_parser = subparsers.add_parser("extract", help="Extract triples from text")
    _add_common_args(extract_parser)
    extract_parser.add_argument("--input", required=True, help="Input text file")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "verify":
        _cmd_verify(args)
    elif args.command == "verify-batch":
        _cmd_verify_batch(args)
    elif args.command == "extract":
        _cmd_extract(args)


def _cmd_verify(args):
    from depver.pipeline import DepVerifier
    from depver.scoring.report import format_report, format_json_report

    verifier = DepVerifier.from_pretrained(args.model, device=args.device, models_dir=args.models_dir)

    source_text = Path(args.source).read_text()
    generated_text = Path(args.generated).read_text()

    result = verifier.verify(source_text, generated_text, threshold=args.threshold)

    if args.format == "json":
        print(json.dumps(format_json_report(result), indent=2))
    else:
        print(format_report(result))


def _cmd_verify_batch(args):
    from depver.pipeline import DepVerifier
    from depver.scoring.report import format_json_report

    verifier = DepVerifier.from_pretrained(args.model, device=args.device, models_dir=args.models_dir)

    with open(args.input) as f_in, open(args.output, "w") as f_out:
        for line_num, line in enumerate(f_in):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            result = verifier.verify(record["source"], record["generated"], threshold=args.threshold)
            report = format_json_report(result)
            report["id"] = record.get("id", line_num)
            f_out.write(json.dumps(report) + "\n")

            if (line_num + 1) % 100 == 0:
                print(f"Processed {line_num + 1} pairs...", file=sys.stderr)

    print(f"Results written to {args.output}", file=sys.stderr)


def _cmd_extract(args):
    from depver.pipeline import DepVerifier

    verifier = DepVerifier.from_pretrained(args.model, device=args.device, models_dir=args.models_dir)

    text = Path(args.input).read_text()
    graphs = verifier.parse_text(text)
    triples = verifier.extract(graphs)

    for t in triples:
        print(json.dumps({
            "signature": t.signature,
            "predicate": t.predicate_lemma,
            "subject": t.subject.signature if t.subject else None,
            "object": t.object.signature if t.object else None,
            "negated": t.negated,
            "voice": t.voice,
            "clause_type": t.clause_type.name,
            "sentence": t.source_text,
        }))


if __name__ == "__main__":
    main()
