"""Human-readable verification reports."""

from __future__ import annotations

from depver.schema import VerificationResult, Divergence, DivergenceType


def format_report(result: VerificationResult) -> str:
    """Generate a human-readable verification report."""
    lines: list[str] = []

    lines.append("=" * 60)
    lines.append("DEPVER VERIFICATION REPORT")
    lines.append("=" * 60)

    # Scores
    s = result.scores
    lines.append("")
    lines.append(f"Factuality score:    {s.factuality_score:.3f}")
    lines.append(f"Triple precision:    {s.triple_precision:.3f}")
    lines.append(f"Triple recall:       {s.triple_recall:.3f}")
    lines.append(f"Triple F1:           {s.triple_f1:.3f}")
    lines.append(f"Hallucinated:        {s.num_hallucinated}")
    lines.append(f"Omitted:             {s.num_omitted}")
    lines.append(f"Divergent:           {s.num_divergent}")

    # Extracted triples
    lines.append("")
    lines.append(f"Source triples ({len(result.source_triples)}):")
    for t in result.source_triples:
        neg = "[NEG] " if t.negated else ""
        lines.append(f"  {neg}{t.signature}")

    lines.append("")
    lines.append(f"Generated triples ({len(result.generated_triples)}):")
    for t in result.generated_triples:
        neg = "[NEG] " if t.negated else ""
        lines.append(f"  {neg}{t.signature}")

    # Divergences by severity
    if result.divergences:
        lines.append("")
        lines.append("DIVERGENCES:")
        lines.append("-" * 40)

        by_severity = {"critical": [], "high": [], "medium": []}
        for d in result.divergences:
            by_severity[d.type.severity].append(d)

        for severity in ("critical", "high", "medium"):
            divs = by_severity[severity]
            if not divs:
                continue
            lines.append(f"\n  [{severity.upper()}]")
            for d in divs:
                lines.append(f"    {d.type.value}: {d.description}")
                if d.source_span:
                    lines.append(f"      source:    {d.source_span}")
                if d.generated_span:
                    lines.append(f"      generated: {d.generated_span}")
    else:
        lines.append("")
        lines.append("No divergences detected.")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


def format_json_report(result: VerificationResult) -> dict:
    """Generate a JSON-serializable verification report."""
    return {
        "scores": {
            "factuality_score": result.scores.factuality_score,
            "triple_precision": result.scores.triple_precision,
            "triple_recall": result.scores.triple_recall,
            "triple_f1": result.scores.triple_f1,
            "num_hallucinated": result.scores.num_hallucinated,
            "num_omitted": result.scores.num_omitted,
            "num_divergent": result.scores.num_divergent,
        },
        "source_triples": [t.signature for t in result.source_triples],
        "generated_triples": [t.signature for t in result.generated_triples],
        "divergences": [
            {
                "type": d.type.value,
                "severity": d.type.severity,
                "description": d.description,
                "source_span": d.source_span,
                "generated_span": d.generated_span,
            }
            for d in result.divergences
        ],
    }
