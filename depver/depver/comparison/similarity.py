"""Similarity functions for predicates and entities.

All models are loaded from local paths under DEPVER_MODELS_DIR (no network calls).
Set this env var or pass models_dir to init_backends() before use.

Default: $WORK/Projects/Hops_parser_torch/models
Expected layout:
    models/
    ├── nli/          # cross-encoder/nli-deberta-v3-base (huggingface-cli download)
    └── embedder/     # sentence-transformers/all-MiniLM-L6-v2 (optional)
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from depver.schema import Entity


_nli = None       # (model, tokenizer) or False
_embedder = None  # SentenceTransformer or False
_models_dir: Path | None = None


def init_backends(models_dir: str | Path | None = None) -> None:
    """Initialize model backends from a local directory. Call once at startup."""
    global _models_dir, _nli, _embedder
    if models_dir is not None:
        _models_dir = Path(models_dir)
    else:
        _models_dir = Path(os.environ.get(
            "DEPVER_MODELS_DIR",
            os.path.expandvars("$WORK/Projects/Hops_parser_torch/models"),
        ))
    # Reset so they get re-loaded
    _nli = None
    _embedder = None


def _get_models_dir() -> Path:
    global _models_dir
    if _models_dir is None:
        init_backends()
    return _models_dir  # type: ignore


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predicate_similarity(lemma_a: str, lemma_b: str) -> float:
    """Similarity between two predicate lemmas."""
    if lemma_a.lower() == lemma_b.lower():
        return 1.0
    return _nli_similarity(
        f"A person {lemma_a} something.",
        f"A person {lemma_b} something.",
    )


def entity_similarity(a: Entity | None, b: Entity | None) -> float:
    """Similarity between two entities."""
    if a is None and b is None:
        return 1.0
    if a is None or b is None:
        return 0.0

    if a.head_lemma.lower() == b.head_lemma.lower():
        head_sim = 1.0
    else:
        head_sim = _nli_similarity(
            f"They discussed the {a.head_lemma}.",
            f"They discussed the {b.head_lemma}.",
        )

    a_mods = {m.text.lower() for m in a.modifiers if m.deprel != "det"}
    b_mods = {m.text.lower() for m in b.modifiers if m.deprel != "det"}
    mod_overlap = _set_overlap(a_mods, b_mods)

    a_nmod = {e.head_lemma.lower() for e in a.nmod_chain}
    b_nmod = {e.head_lemma.lower() for e in b.nmod_chain}
    nmod_overlap = _set_overlap(a_nmod, b_nmod)

    return 0.6 * head_sim + 0.25 * mod_overlap + 0.15 * nmod_overlap


def polarity_match(neg_a: bool, neg_b: bool) -> float:
    return 1.0 if neg_a == neg_b else 0.0


def oblique_similarity(obliques_a: tuple, obliques_b: tuple) -> float:
    if not obliques_a and not obliques_b:
        return 1.0
    if not obliques_a or not obliques_b:
        return 0.0

    matched = 0.0
    used: set[int] = set()
    for oa in obliques_a:
        best_score = 0.0
        best_idx = -1
        for i, ob in enumerate(obliques_b):
            if i in used:
                continue
            if oa.case_marker == ob.case_marker:
                score = entity_similarity(oa.entity, ob.entity)
                if score > best_score:
                    best_score = score
                    best_idx = i
        if best_idx >= 0 and best_score > 0.3:
            matched += best_score
            used.add(best_idx)

    return matched / max(len(obliques_a), len(obliques_b))


# ---------------------------------------------------------------------------
# NLI backend
# ---------------------------------------------------------------------------


def _nli_similarity(premise: str, hypothesis: str) -> float:
    """Score how much premise entails hypothesis using NLI. Falls back to 0.5."""
    model, tokenizer = _load_nli()
    if model is None:
        return 0.5

    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=128)
    with torch.inference_mode():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0]

    # labels: 0=contradiction, 1=neutral, 2=entailment
    contradiction, neutral, entailment = probs[0].item(), probs[1].item(), probs[2].item()

    if contradiction > 0.7:
        return 0.1
    if entailment > 0.6:
        return 0.85
    return 0.1 * contradiction + 0.4 * neutral + 0.85 * entailment


def _load_nli():
    """Load NLI model from local path. Returns (model, tokenizer) or (None, None)."""
    global _nli
    if _nli is False:
        return None, None
    if _nli is not None:
        return _nli

    nli_path = _get_models_dir() / "nli"
    if not nli_path.exists():
        _nli = False
        return None, None

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(nli_path)
        model = AutoModelForSequenceClassification.from_pretrained(nli_path)
        model.eval()
        _nli = (model, tokenizer)
        return _nli
    except Exception:
        _nli = False
        return None, None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_overlap(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
