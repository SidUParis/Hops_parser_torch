"""Similarity functions for predicates and entities.

Supports multiple backends, auto-detected at runtime:
  1. NLI model (best for predicate entailment — "reported" vs "confirmed")
  2. Sentence-transformers embeddings (good for paraphrase detection)
  3. WordNet (decent for synonyms, limited coverage)
  4. Character overlap (last resort fallback)
"""

from __future__ import annotations

from depver.schema import Entity

# Lazy-loaded backends (None = not tried, False = tried and unavailable)
_nli_model = None
_nli_tokenizer = None
_wordnet = None
_embedder = None


def predicate_similarity(lemma_a: str, lemma_b: str) -> float:
    """Similarity between two predicate lemmas.

    Cascade: exact → NLI entailment → WordNet → embedding cosine → char overlap.
    """
    if lemma_a.lower() == lemma_b.lower():
        return 1.0

    # NLI entailment: "he reported" → "he confirmed"?
    nli_score = _nli_predicate_entailment(lemma_a, lemma_b)
    if nli_score is not None:
        return nli_score

    # WordNet synonyms
    wn_score = _wordnet_similarity(lemma_a, lemma_b)
    if wn_score is not None:
        return wn_score

    # Embedding cosine
    emb_score = _embedding_similarity(lemma_a, lemma_b)
    if emb_score is not None:
        return emb_score * 0.8  # discount embedding-only matches

    # Fallback
    return _char_overlap(lemma_a, lemma_b)


def entity_similarity(a: Entity | None, b: Entity | None) -> float:
    """Similarity between two entities."""
    if a is None and b is None:
        return 1.0
    if a is None or b is None:
        return 0.0

    # Head lemma
    if a.head_lemma.lower() == b.head_lemma.lower():
        head_sim = 1.0
    else:
        # Try NLI: "the reform" entails "the legislative change"?
        nli = _nli_entity_similarity(a.head_lemma, b.head_lemma)
        if nli is not None:
            head_sim = nli
        else:
            emb = _embedding_similarity(a.head_lemma, b.head_lemma)
            head_sim = emb if emb is not None else _char_overlap(a.head_lemma, b.head_lemma)

    # Modifier overlap (ignore determiners)
    a_mods = {m.text.lower() for m in a.modifiers if m.deprel != "det"}
    b_mods = {m.text.lower() for m in b.modifiers if m.deprel != "det"}
    if a_mods or b_mods:
        mod_overlap = len(a_mods & b_mods) / max(len(a_mods | b_mods), 1)
    else:
        mod_overlap = 1.0

    # nmod chain overlap
    a_nmod = {e.head_lemma.lower() for e in a.nmod_chain}
    b_nmod = {e.head_lemma.lower() for e in b.nmod_chain}
    if a_nmod or b_nmod:
        nmod_overlap = len(a_nmod & b_nmod) / max(len(a_nmod | b_nmod), 1)
    else:
        nmod_overlap = 1.0

    return 0.6 * head_sim + 0.25 * mod_overlap + 0.15 * nmod_overlap


def polarity_match(neg_a: bool, neg_b: bool) -> float:
    """Binary: same polarity = 1.0, different = 0.0."""
    return 1.0 if neg_a == neg_b else 0.0


def oblique_similarity(obliques_a: tuple, obliques_b: tuple) -> float:
    """Similarity between oblique argument lists."""
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

    total = max(len(obliques_a), len(obliques_b))
    return matched / total if total > 0 else 1.0


# ---------------------------------------------------------------------------
# Backend: NLI model (DeBERTa-v3-base fine-tuned on MNLI)
# ---------------------------------------------------------------------------
# This is the key learned component. A small NLI model (~180M params) answers:
#   "Does 'Someone reported X' entail 'Someone confirmed X'?"
# This handles predicate entailment, entity coreference-by-description, and
# paraphrase detection — all things rules can't do well.
#
# Model: cross-encoder/nli-deberta-v3-base (~350MB, runs fast on CPU/GPU)
# We frame it as: premise = "A person [verb_a] something" → hypothesis = same with [verb_b]
# Then read the entailment/contradiction/neutral probabilities.
# ---------------------------------------------------------------------------


def _nli_predicate_entailment(lemma_a: str, lemma_b: str) -> float | None:
    """Use NLI to check if predicate A entails predicate B.

    Frames as: "Someone [A] something." entails "Someone [B] something."?
    Returns a score in [0, 1] or None if NLI model unavailable.
    """
    model, tokenizer = _load_nli()
    if model is None:
        return None

    import torch

    premise = f"A person {lemma_a} something."
    hypothesis = f"A person {lemma_b} something."

    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=128)
    with torch.inference_mode():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    # DeBERTa MNLI labels: 0=contradiction, 1=neutral, 2=entailment
    contradiction = probs[0].item()
    neutral = probs[1].item()
    entailment = probs[2].item()

    # Contradictory predicates (reported ↔ denied) → low score
    if contradiction > 0.7:
        return 0.1
    # Entailed predicates (reported ↔ said) → high score
    if entailment > 0.6:
        return 0.85
    # Neutral (reported ↔ worked) → moderate
    if neutral > 0.5:
        return 0.3

    # Mixed — use weighted combination
    return 0.1 * contradiction + 0.4 * neutral + 0.85 * entailment


def _nli_entity_similarity(text_a: str, text_b: str) -> float | None:
    """Use NLI to check if two entity descriptions refer to the same thing.

    "the reform" vs "the legislative change"
    """
    model, tokenizer = _load_nli()
    if model is None:
        return None

    import torch

    premise = f"They discussed the {text_a}."
    hypothesis = f"They discussed the {text_b}."

    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=128)
    with torch.inference_mode():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    entailment = probs[2].item()
    contradiction = probs[0].item()

    if contradiction > 0.7:
        return 0.1
    if entailment > 0.6:
        return 0.85
    return 0.3 + 0.55 * entailment


def _load_nli():
    """Lazy-load the NLI model. Returns (model, tokenizer) or (None, None)."""
    global _nli_model, _nli_tokenizer
    if _nli_model is False:
        return None, None
    if _nli_model is not None:
        return _nli_model, _nli_tokenizer
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_name = "cross-encoder/nli-deberta-v3-base"
        _nli_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _nli_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        _nli_model.eval()
        return _nli_model, _nli_tokenizer
    except (ImportError, OSError):
        _nli_model = False
        _nli_tokenizer = None
        return None, None


# ---------------------------------------------------------------------------
# Backend: WordNet
# ---------------------------------------------------------------------------


def _wordnet_similarity(lemma_a: str, lemma_b: str) -> float | None:
    """Check WordNet synonym/hypernym similarity. Returns None if unavailable."""
    global _wordnet
    if _wordnet is False:
        return None
    try:
        if _wordnet is None:
            from nltk.corpus import wordnet as wn
            _wordnet = wn
        else:
            wn = _wordnet

        synsets_a = wn.synsets(lemma_a.lower(), pos=wn.VERB)
        synsets_b = wn.synsets(lemma_b.lower(), pos=wn.VERB)
        if not synsets_a or not synsets_b:
            # Try nouns too
            synsets_a = wn.synsets(lemma_a.lower())
            synsets_b = wn.synsets(lemma_b.lower())
        if not synsets_a or not synsets_b:
            return None

        # Exact synonym (share a synset)
        set_a = {s.name() for s in synsets_a}
        set_b = {s.name() for s in synsets_b}
        if set_a & set_b:
            return 0.85

        # Path similarity
        best = 0.0
        for sa in synsets_a[:3]:
            for sb in synsets_b[:3]:
                sim = sa.path_similarity(sb)
                if sim is not None and sim > best:
                    best = sim
        if best > 0.3:
            return 0.5 + 0.35 * best
        return None

    except (ImportError, LookupError):
        _wordnet = False
        return None


# ---------------------------------------------------------------------------
# Backend: Sentence-transformers embeddings
# ---------------------------------------------------------------------------


def _embedding_similarity(text_a: str, text_b: str) -> float | None:
    """Cosine similarity via sentence-transformers. Returns None if unavailable."""
    global _embedder
    if _embedder is False:
        return None
    try:
        if _embedder is None:
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer("all-MiniLM-L6-v2")

        embeddings = _embedder.encode([text_a, text_b], convert_to_tensor=True)
        from sentence_transformers.util import cos_sim
        score = cos_sim(embeddings[0], embeddings[1]).item()
        return max(0.0, score)

    except ImportError:
        _embedder = False
        return None


# ---------------------------------------------------------------------------
# Fallback: character overlap
# ---------------------------------------------------------------------------


def _char_overlap(a: str, b: str) -> float:
    """Simple character-level Jaccard as ultimate fallback."""
    a_lower, b_lower = a.lower(), b.lower()
    if a_lower == b_lower:
        return 1.0
    a_chars = set(a_lower)
    b_chars = set(b_lower)
    if not a_chars or not b_chars:
        return 0.0
    return len(a_chars & b_chars) / len(a_chars | b_chars)
