# DepVer: Dependency-Structure Verification of LLM Outputs

## Research Plan & Code Architecture

---

## Part I: The Research

### 1. Thesis

Current hallucination detection methods (uncertainty estimation, self-consistency, NLI classifiers, LLM-as-judge) treat text as flat sequences. They can tell you *something is wrong* but not *what specifically changed*. Dependency parsing decomposes sentences into typed, directed relations — enabling hallucination detection that is both **precise** (identifies the exact relation that was altered) and **interpretable** (classifies the type of error in human-understandable terms).

The only prior work in this direction — DAE (Goyal & Durrett, 2020) — operates at the **arc level** (individual dependency edges) and produces a binary entailed/not-entailed judgment. We propose **triple-level comparison**: extracting predicate-argument structures and comparing them structurally, which captures multi-arc phenomena (causal hallucination, attribution shifts) that arc-level methods miss, and produces a **typed divergence classification** rather than a binary score.

### 2. Research Questions

| # | Question | How We Answer It |
|---|----------|-----------------|
| RQ1 | Does triple-level structural comparison detect hallucination types that token/embedding methods miss? | Compare error detection by type (verb sub, arg swap, negation, causal) across methods |
| RQ2 | Does structural verification improve when combined with embedding-based methods? | Ensemble DepVer + BERTScore, measure F1 lift |
| RQ3 | How much does parser accuracy affect downstream verification quality? | Run with gold parses vs. predicted parses, measure degradation |
| RQ4 | Which divergence types are most reliably detected via structure? | Per-type precision/recall on annotated benchmarks |
| RQ5 | Does the approach generalize across languages? | Evaluate on English + French (+ optionally German) |

### 3. Experimental Design

#### 3.1 Datasets

**Primary evaluation:**
- **FRANK** (Pagnoni et al., 2021) — 2,246 annotated summary-article pairs with fine-grained error typology (semantic frame errors, discourse errors, content verifiability). Maps well to our divergence types.
- **AggreFact** (Tang et al., 2023) — unified benchmark aggregating 9 factuality datasets across CNN/DM and XSum. Binary labels at sentence level.

**Secondary evaluation:**
- **XSumFaith** (Maynez et al., 2020) — intrinsic vs. extrinsic hallucination annotations
- **TRUE** (Honovich et al., 2022) — cross-task (summarization, dialogue, paraphrase)
- **SummEval** (Fabbri et al., 2021) — human consistency ratings (for correlation analysis)

**For French evaluation:**
- We may need to construct a small annotated dataset (200-500 sentence pairs) from French LLM summaries, annotated for factual errors. Alternatively, translate FRANK examples and re-annotate.

#### 3.2 Baselines

| Method | Category | Implementation |
|--------|----------|---------------|
| ROUGE-L | Token overlap | `rouge-score` library |
| BERTScore | Embedding similarity | `bert-score` library |
| FactCC | NLI-based classifier | Authors' checkpoint |
| DAE | Dependency arc entailment | Authors' code (github.com/tagoyal/dae-factuality) |
| QuestEval | QA-based | Authors' code |
| LLM-as-judge | Prompting GPT-4/Claude | Custom prompts |
| SummaC | NLI + segmentation | Authors' code |

#### 3.3 Metrics

**Detection performance:**
- Balanced accuracy, precision, recall, F1 at sentence level (is this sentence factual?)
- ROC-AUC for threshold-free comparison

**Granularity / interpretability:**
- % of detected errors where the system can identify the specific words responsible
- Classification accuracy on FRANK's error typology (if our types map to theirs)

**Correlation with human judgment:**
- Spearman ρ with human factuality scores (SummEval)
- Kendall τ for ranking agreement

**Ablations:**
- Gold parse vs. predicted parse (parser error propagation)
- Individual divergence types on/off (which types contribute most?)
- Strict vs. relaxed entity matching
- With/without WordNet/embedding-based predicate similarity

### 4. Expected Contributions

1. **DepVer** — an open-source tool for structural hallucination detection, integrated with hopsparser
2. **A typed divergence taxonomy** going beyond binary entailment — 10 classified error types detectable via dependency structure
3. **Empirical evidence** showing where structural methods complement embedding-based approaches (and where they don't)
4. **Cross-lingual evaluation** demonstrating the approach works for morphologically rich languages (French)
5. **An interpretability argument** — showing that structural verification produces human-auditable explanations, which NLI/embedding methods cannot

### 5. Timeline

| Phase | Weeks | Deliverable |
|-------|-------|-------------|
| **Phase 1**: Triple extraction | 1-3 | `depver.extraction` module, tested on gold UD parses |
| **Phase 2**: Structural comparison | 4-6 | `depver.comparison` module, tested on synthetic divergences |
| **Phase 3**: Pipeline integration | 7-8 | End-to-end pipeline: text → parse → extract → compare → report |
| **Phase 4**: Evaluation | 9-12 | Results on FRANK, AggreFact; baseline comparisons |
| **Phase 5**: Cross-lingual + ablations | 13-14 | French evaluation, ablation studies |
| **Phase 6**: Paper writing | 15-18 | Paper draft targeting ACL/EMNLP/EACL |

---

## Part II: Code Architecture

### 6. Repository Structure

```
hops_parser/
├── hopsparser/                    # existing parser (untouched)
│   ├── hopsparser/
│   │   ├── parser.py
│   │   ├── deptree.py             # DepGraph, DepNode — we read from these
│   │   ├── lexers.py
│   │   ├── server.py
│   │   └── ...
│   └── pyproject.toml
│
├── depver/                        # NEW: verification library
│   ├── pyproject.toml
│   ├── depver/
│   │   ├── __init__.py
│   │   │
│   │   ├── schema.py             # core data types (Triple, Entity, etc.)
│   │   │
│   │   ├── extraction/           # DepGraph → triples
│   │   │   ├── __init__.py
│   │   │   ├── triples.py        # extract_triples(DepGraph) → list[Triple]
│   │   │   ├── entities.py       # build Entity from subtree
│   │   │   └── walkers.py        # tree-walking utilities
│   │   │
│   │   ├── comparison/           # triple alignment + divergence detection
│   │   │   ├── __init__.py
│   │   │   ├── matcher.py        # align source↔generated triples
│   │   │   ├── similarity.py     # predicate/entity similarity functions
│   │   │   └── divergence.py     # classify divergence type
│   │   │
│   │   ├── scoring/              # aggregate metrics
│   │   │   ├── __init__.py
│   │   │   ├── metrics.py        # precision, recall, F1 over triples
│   │   │   └── report.py         # human-readable report generation
│   │   │
│   │   ├── pipeline.py           # end-to-end verification
│   │   └── server.py             # FastAPI endpoint (optional)
│   │
│   └── tests/
│       ├── conftest.py
│       ├── test_extraction.py
│       ├── test_comparison.py
│       ├── test_scoring.py
│       └── test_pipeline.py
│
├── experiments/                   # evaluation scripts (NOT a package)
│   ├── eval_frank.py
│   ├── eval_aggrefact.py
│   ├── baselines/
│   │   ├── run_bertscore.py
│   │   ├── run_dae.py
│   │   └── run_factcc.py
│   └── analysis/
│       ├── per_type_breakdown.py
│       └── ablation.py
│
├── STRUCTURED_VERIFICATION.md     # existing sketch
└── RESEARCH_PLAN.md               # this file
```

### 7. Core Data Types (`depver/schema.py`)

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


@dataclass(frozen=True)
class Entity:
    """A noun phrase extracted from a dependency subtree."""
    head_lemma: str                          # head noun lemma ("bill")
    head_form: str                           # surface form ("bill")
    head_upos: str                           # POS tag ("NOUN")
    modifiers: tuple[Modifier, ...] = ()     # amod, nummod, det
    nmod_chain: tuple[Entity, ...] = ()      # nested nmod ("CEO of Apple")
    node_ids: tuple[int, ...] = ()           # original DepNode identifiers (for tracing)

    @property
    def signature(self) -> str:
        """Canonical string for quick comparison."""
        mods = "+".join(m.text for m in self.modifiers)
        nmod = ">".join(e.head_lemma for e in self.nmod_chain)
        return f"{self.head_lemma}[{mods}]({nmod})"


@dataclass(frozen=True)
class Modifier:
    """An adjectival, numeric, or determiner modifier."""
    text: str        # lemma or form
    deprel: str      # "amod", "nummod", "det", "advmod"
    node_id: int     # DepNode identifier


class ClauseType(Enum):
    ROOT = auto()       # main clause
    ADVCL = auto()      # adverbial clause (causal, temporal, conditional)
    XCOMP = auto()      # open clausal complement
    CCOMP = auto()      # closed clausal complement
    RELCL = auto()      # relative clause
    PARATAXIS = auto()  # parataxis


@dataclass(frozen=True)
class Triple:
    """A predicate-argument structure extracted from a dependency tree."""
    predicate_lemma: str                     # verb lemma ("sign")
    predicate_form: str                      # surface form ("signed")
    predicate_upos: str                      # typically "VERB" or "AUX"
    predicate_id: int                        # DepNode identifier

    subject: Entity | None = None            # nsubj / nsubj:pass
    object: Entity | None = None             # obj / iobj
    obliques: tuple[Oblique, ...] = ()       # obl (time, place, instrument)

    negated: bool = False                    # advmod "not" / "never" / etc.
    modality: str | None = None              # aux "can", "must", "might"
    voice: str = "active"                    # "active" or "passive" (from nsubj:pass)
    clause_type: ClauseType = ClauseType.ROOT
    clause_mark: str | None = None           # subordinating conjunction: "because", "if", etc.

    # Provenance for interpretability
    sentence_index: int = 0                  # which sentence this came from
    source_text: str = ""                    # original sentence text

    @property
    def signature(self) -> str:
        """Canonical string for quick comparison."""
        subj = self.subject.signature if self.subject else "_"
        obj = self.object.signature if self.object else "_"
        neg = "!" if self.negated else ""
        return f"{neg}{self.predicate_lemma}({subj}, {obj})"


@dataclass(frozen=True)
class Oblique:
    """A prepositional/case-marked argument."""
    case_marker: str     # preposition: "in", "with", "from"
    entity: Entity


class DivergenceType(Enum):
    """Classification of how a generated triple diverges from its source."""
    VERB_SUBSTITUTION = "verb_substitution"          # same args, different predicate
    ARGUMENT_SWAP = "argument_swap"                  # nsubj↔obj reversed
    NEGATION_FLIP = "negation_flip"                  # polarity inverted
    ENTITY_HALLUCINATION = "entity_hallucination"    # entity in gen, not in source
    MODIFIER_HALLUCINATION = "modifier_hallucination"  # added amod/nummod
    CAUSAL_HALLUCINATION = "causal_hallucination"    # advcl causal link invented
    ATTRIBUTION_SHIFT = "attribution_shift"          # different nsubj on reporting verb
    TEMPORAL_SHIFT = "temporal_shift"                 # different time oblique
    EDITORIALIZATION = "editorialization"             # evaluative modifier added
    OMISSION = "omission"                            # source triple missing in generated

    @property
    def severity(self) -> str:
        critical = {self.NEGATION_FLIP}
        high = {self.VERB_SUBSTITUTION, self.ARGUMENT_SWAP,
                self.ENTITY_HALLUCINATION, self.CAUSAL_HALLUCINATION,
                self.ATTRIBUTION_SHIFT}
        if self in critical:
            return "critical"
        elif self in high:
            return "high"
        return "medium"


@dataclass(frozen=True)
class Alignment:
    """A matched pair of source and generated triples."""
    source_triple: Triple | None         # None if hallucination (no source match)
    generated_triple: Triple | None      # None if omission
    similarity_score: float              # 0.0 to 1.0
    divergences: tuple[Divergence, ...] = ()


@dataclass(frozen=True)
class Divergence:
    """A specific divergence found between aligned triples."""
    type: DivergenceType
    description: str                     # human-readable explanation
    source_span: str                     # text from source involved
    generated_span: str                  # text from generated involved


@dataclass
class VerificationResult:
    """Complete output of the verification pipeline."""
    source_triples: list[Triple]
    generated_triples: list[Triple]
    alignments: list[Alignment]
    scores: VerificationScores
    divergences: list[Divergence]


@dataclass
class VerificationScores:
    """Aggregate scores for the verification."""
    triple_precision: float    # |matched gen triples| / |gen triples|
    triple_recall: float       # |matched gen triples| / |source triples|
    triple_f1: float
    factuality_score: float    # weighted score penalizing high-severity divergences
    num_hallucinated: int      # ungrounded generated triples
    num_omitted: int           # unmatched source triples
    num_divergent: int         # matched but with divergences
```

### 8. Module Details

#### 8.1 Extraction (`depver/extraction/triples.py`)

The core function. Walks a `DepGraph` and extracts `Triple` objects.

```python
from hopsparser.deptree import DepGraph, DepNode
from depver.schema import Triple, Entity, Oblique, Modifier, ClauseType

def extract_triples(graph: DepGraph) -> list[Triple]:
    """Extract predicate-argument triples from a parsed dependency graph.

    Strategy:
    1. Find all predicate nodes (VERB, AUX with clausal function)
    2. For each predicate, collect its arguments by deprel
    3. Build Triple with subject, object, obliques, negation, modality
    """
    triples = []
    for node in graph.nodes:
        if not _is_predicate(node):
            continue
        triple = _build_triple(graph, node)
        triples.append(triple)
    return triples


def _is_predicate(node: DepNode) -> bool:
    """A node is a predicate if it's a VERB, or an AUX heading a clause."""
    if node.upos == "VERB":
        return True
    # AUX can be predicate in copular constructions
    if node.upos == "AUX" and node.deprel in ("root", "ccomp", "xcomp", "advcl"):
        return True
    return False


def _build_triple(graph: DepGraph, predicate: DepNode) -> Triple:
    """Collect arguments of a predicate from its dependents."""
    children = _get_children(graph, predicate.identifier)

    subject = None
    obj = None
    obliques = []
    negated = False
    modality = None
    voice = "active"
    clause_mark = None

    for child in children:
        rel = _base_deprel(child.deprel)  # strip subtypes: "nsubj:pass" → "nsubj"
        full_rel = child.deprel

        if rel == "nsubj":
            subject = _build_entity(graph, child)
            if "pass" in (full_rel or ""):
                voice = "passive"
        elif rel == "obj" or rel == "iobj":
            obj = _build_entity(graph, child)
        elif rel == "obl":
            case = _find_case(graph, child)
            obliques.append(Oblique(case_marker=case, entity=_build_entity(graph, child)))
        elif rel == "advmod" and _is_negation(child):
            negated = True
        elif rel == "aux" and child.upos == "AUX":
            modality = child.lemma
        elif rel == "mark":
            clause_mark = child.form

    clause_type = _determine_clause_type(predicate)

    return Triple(
        predicate_lemma=predicate.lemma or predicate.form,
        predicate_form=predicate.form,
        predicate_upos=predicate.upos or "VERB",
        predicate_id=predicate.identifier,
        subject=subject,
        object=obj,
        obliques=tuple(obliques),
        negated=negated,
        modality=modality,
        voice=voice,
        clause_type=clause_type,
        clause_mark=clause_mark,
        source_text=" ".join(graph.words[1:]),  # skip <root>
    )


def _build_entity(graph: DepGraph, head: DepNode) -> Entity:
    """Build an Entity from a noun and its modifier subtree."""
    children = _get_children(graph, head.identifier)
    modifiers = []
    nmod_chain = []

    for child in children:
        rel = _base_deprel(child.deprel)
        if rel in ("amod", "nummod", "det"):
            modifiers.append(Modifier(
                text=child.lemma or child.form,
                deprel=rel,
                node_id=child.identifier,
            ))
        elif rel == "nmod":
            nmod_chain.append(_build_entity(graph, child))

    return Entity(
        head_lemma=head.lemma or head.form,
        head_form=head.form,
        head_upos=head.upos or "NOUN",
        modifiers=tuple(modifiers),
        nmod_chain=tuple(nmod_chain),
        node_ids=tuple(_subtree_ids(graph, head.identifier)),
    )


def _get_children(graph: DepGraph, node_id: int) -> list[DepNode]:
    """Get all direct children of a node in the dependency tree."""
    return [n for n in graph.nodes if n.head == node_id]


def _base_deprel(deprel: str | None) -> str:
    """Strip subtype: 'nsubj:pass' → 'nsubj'."""
    if deprel is None:
        return ""
    return deprel.split(":")[0]


def _find_case(graph: DepGraph, node: DepNode) -> str:
    """Find the case marker (preposition) for an oblique."""
    children = _get_children(graph, node.identifier)
    for child in children:
        if _base_deprel(child.deprel) == "case":
            return child.form
    return ""


def _is_negation(node: DepNode) -> bool:
    """Check if an advmod is a negation word."""
    neg_lemmas = {"not", "never", "no", "nor", "neither", "ne", "pas", "jamais"}
    return (node.lemma or node.form).lower() in neg_lemmas


def _determine_clause_type(node: DepNode) -> ClauseType:
    """Determine the clause type from the node's deprel."""
    rel = _base_deprel(node.deprel)
    mapping = {
        "root": ClauseType.ROOT,
        "advcl": ClauseType.ADVCL,
        "xcomp": ClauseType.XCOMP,
        "ccomp": ClauseType.CCOMP,
        "relcl": ClauseType.RELCL,
        "parataxis": ClauseType.PARATAXIS,
    }
    return mapping.get(rel, ClauseType.ROOT)


def _subtree_ids(graph: DepGraph, node_id: int) -> list[int]:
    """Collect all node IDs in the subtree rooted at node_id."""
    ids = [node_id]
    for child in _get_children(graph, node_id):
        ids.extend(_subtree_ids(graph, child.identifier))
    return ids
```

#### 8.2 Comparison (`depver/comparison/matcher.py`)

Aligns triples between source and generated text.

```python
from depver.schema import Triple, Alignment, Entity
from depver.comparison.similarity import (
    predicate_similarity,
    entity_similarity,
    polarity_match,
)

def align_triples(
    source_triples: list[Triple],
    generated_triples: list[Triple],
    threshold: float = 0.4,
) -> list[Alignment]:
    """Align generated triples to source triples using best-match.

    Uses a greedy algorithm:
    1. Compute similarity matrix between all pairs
    2. Greedily assign best matches above threshold
    3. Unmatched generated triples → hallucination candidates
    4. Unmatched source triples → omissions
    """
    if not source_triples or not generated_triples:
        # All hallucinations or all omissions
        ...

    # Compute similarity matrix
    sim_matrix = []
    for g in generated_triples:
        row = []
        for s in source_triples:
            score = triple_similarity(g, s)
            row.append(score)
        sim_matrix.append(row)

    # Greedy matching (could upgrade to Hungarian for optimal assignment)
    alignments = []
    used_source = set()
    used_gen = set()

    # Sort all pairs by similarity descending
    pairs = [
        (sim_matrix[gi][si], gi, si)
        for gi in range(len(generated_triples))
        for si in range(len(source_triples))
    ]
    pairs.sort(reverse=True)

    for score, gi, si in pairs:
        if gi in used_gen or si in used_source:
            continue
        if score < threshold:
            break
        alignments.append(Alignment(
            source_triple=source_triples[si],
            generated_triple=generated_triples[gi],
            similarity_score=score,
        ))
        used_gen.add(gi)
        used_source.add(si)

    # Unmatched generated → hallucination candidates
    for gi in range(len(generated_triples)):
        if gi not in used_gen:
            alignments.append(Alignment(
                source_triple=None,
                generated_triple=generated_triples[gi],
                similarity_score=0.0,
            ))

    # Unmatched source → omissions
    for si in range(len(source_triples)):
        if si not in used_source:
            alignments.append(Alignment(
                source_triple=source_triples[si],
                generated_triple=None,
                similarity_score=0.0,
            ))

    return alignments


def triple_similarity(g: Triple, s: Triple) -> float:
    """Weighted similarity between two triples."""
    w_pred = 0.35
    w_subj = 0.25
    w_obj  = 0.25
    w_pol  = 0.15

    pred_sim = predicate_similarity(g.predicate_lemma, s.predicate_lemma)

    subj_sim = entity_similarity(g.subject, s.subject) if (g.subject and s.subject) else (
        1.0 if (g.subject is None and s.subject is None) else 0.0
    )

    obj_sim = entity_similarity(g.object, s.object) if (g.object and s.object) else (
        1.0 if (g.object is None and s.object is None) else 0.0
    )

    pol_sim = polarity_match(g.negated, s.negated)

    return w_pred * pred_sim + w_subj * subj_sim + w_obj * obj_sim + w_pol * pol_sim
```

#### 8.3 Comparison (`depver/comparison/similarity.py`)

```python
def predicate_similarity(lemma_a: str, lemma_b: str) -> float:
    """Similarity between two predicate lemmas."""
    # Exact match
    if lemma_a.lower() == lemma_b.lower():
        return 1.0

    # WordNet synonym check (if available)
    if _are_synonyms(lemma_a, lemma_b):
        return 0.8

    # Fallback: embedding cosine similarity
    return _embedding_similarity(lemma_a, lemma_b)


def entity_similarity(a: "Entity | None", b: "Entity | None") -> float:
    """Similarity between two entities."""
    if a is None and b is None:
        return 1.0
    if a is None or b is None:
        return 0.0

    # Head lemma match is most important
    head_sim = 1.0 if a.head_lemma.lower() == b.head_lemma.lower() else _embedding_similarity(
        a.head_lemma, b.head_lemma
    )

    # Modifier overlap
    a_mods = {m.text.lower() for m in a.modifiers if m.deprel != "det"}
    b_mods = {m.text.lower() for m in b.modifiers if m.deprel != "det"}
    if a_mods or b_mods:
        mod_overlap = len(a_mods & b_mods) / max(len(a_mods | b_mods), 1)
    else:
        mod_overlap = 1.0  # both have no modifiers

    return 0.7 * head_sim + 0.3 * mod_overlap


def polarity_match(neg_a: bool, neg_b: bool) -> float:
    """Binary: same polarity = 1.0, different = 0.0."""
    return 1.0 if neg_a == neg_b else 0.0
```

#### 8.4 Divergence Classification (`depver/comparison/divergence.py`)

```python
from depver.schema import Alignment, Divergence, DivergenceType

def classify_divergences(alignments: list[Alignment]) -> list[Divergence]:
    """Analyze each alignment and classify any divergences found."""
    divergences = []

    for alignment in alignments:
        s = alignment.source_triple
        g = alignment.generated_triple

        # Unmatched generated triple → hallucination
        if s is None and g is not None:
            divergences.append(Divergence(
                type=DivergenceType.ENTITY_HALLUCINATION,
                description=f"Generated triple has no source grounding: {g.signature}",
                source_span="",
                generated_span=g.source_text,
            ))
            continue

        # Unmatched source triple → omission
        if g is None and s is not None:
            divergences.append(Divergence(
                type=DivergenceType.OMISSION,
                description=f"Source triple missing from generated text: {s.signature}",
                source_span=s.source_text,
                generated_span="",
            ))
            continue

        # Both exist — check for specific divergences
        assert s is not None and g is not None

        # Negation flip (check first — highest severity)
        if s.negated != g.negated:
            divergences.append(Divergence(
                type=DivergenceType.NEGATION_FLIP,
                description=f"Polarity changed: '{s.predicate_form}' ({'negated' if s.negated else 'affirmed'}) → '{g.predicate_form}' ({'negated' if g.negated else 'affirmed'})",
                source_span=s.predicate_form,
                generated_span=g.predicate_form,
            ))

        # Verb substitution
        if s.predicate_lemma.lower() != g.predicate_lemma.lower():
            if _subjects_match(s, g) and _objects_match(s, g):
                divergences.append(Divergence(
                    type=DivergenceType.VERB_SUBSTITUTION,
                    description=f"Predicate changed: '{s.predicate_form}' → '{g.predicate_form}'",
                    source_span=s.predicate_form,
                    generated_span=g.predicate_form,
                ))

        # Argument swap
        if _is_argument_swap(s, g):
            divergences.append(Divergence(
                type=DivergenceType.ARGUMENT_SWAP,
                description=f"Subject/object swapped: '{s.subject.head_lemma}' ↔ '{s.object.head_lemma}'",
                source_span=f"{s.subject.head_form} ... {s.object.head_form}",
                generated_span=f"{g.subject.head_form} ... {g.object.head_form}",
            ))

        # Modifier hallucination
        _check_modifier_hallucination(s, g, divergences)

        # Causal hallucination
        _check_causal_hallucination(s, g, divergences)

        # Attribution shift
        _check_attribution_shift(s, g, divergences)

    return divergences


REPORTING_VERBS = {"say", "report", "confirm", "deny", "announce", "claim",
                   "state", "tell", "declare", "assert", "argue",
                   "dire", "annoncer", "confirmer", "déclarer"}  # French too


def _is_argument_swap(s, g) -> bool:
    """Check if subject and object are swapped between source and generated."""
    if not (s.subject and s.object and g.subject and g.object):
        return False
    return (
        s.subject.head_lemma.lower() == g.object.head_lemma.lower()
        and s.object.head_lemma.lower() == g.subject.head_lemma.lower()
    )


def _check_attribution_shift(s, g, divergences):
    """Different subject on a reporting verb."""
    if s.predicate_lemma.lower() in REPORTING_VERBS:
        if s.subject and g.subject:
            if s.subject.head_lemma.lower() != g.subject.head_lemma.lower():
                divergences.append(Divergence(
                    type=DivergenceType.ATTRIBUTION_SHIFT,
                    description=f"Attribution changed: '{s.subject.head_form}' → '{g.subject.head_form}' for verb '{s.predicate_form}'",
                    source_span=s.subject.head_form,
                    generated_span=g.subject.head_form,
                ))
```

#### 8.5 Pipeline (`depver/pipeline.py`)

```python
import torch
from hopsparser.parser import BiAffineParser
from hopsparser.deptree import DepGraph
from depver.schema import VerificationResult, VerificationScores
from depver.extraction.triples import extract_triples
from depver.comparison.matcher import align_triples
from depver.comparison.divergence import classify_divergences


class DepVerifier:
    """End-to-end dependency-structure verification."""

    def __init__(self, parser: BiAffineParser):
        self.parser = parser

    @classmethod
    def from_pretrained(cls, model_path: str, device: str = "cpu") -> "DepVerifier":
        parser = BiAffineParser.load(model_path).to(device).eval()
        return cls(parser)

    def parse_text(self, text: str) -> list[DepGraph]:
        """Parse raw text into dependency graphs."""
        with torch.inference_mode():
            return list(self.parser.parse(text.splitlines(), raw=True))

    def extract(self, graphs: list[DepGraph]) -> list["Triple"]:
        """Extract triples from parsed graphs."""
        return [t for g in graphs for t in extract_triples(g)]

    def verify(
        self,
        source_text: str,
        generated_text: str,
        threshold: float = 0.4,
    ) -> VerificationResult:
        """Full pipeline: parse → extract → align → classify → score."""

        # 1. Parse both texts
        source_graphs = self.parse_text(source_text)
        generated_graphs = self.parse_text(generated_text)

        # 2. Extract triples
        source_triples = self.extract(source_graphs)
        gen_triples = self.extract(generated_graphs)

        # 3. Align
        alignments = align_triples(source_triples, gen_triples, threshold=threshold)

        # 4. Classify divergences
        divergences = classify_divergences(alignments)

        # 5. Score
        scores = _compute_scores(alignments, divergences)

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


def _compute_scores(alignments, divergences) -> VerificationScores:
    """Compute aggregate verification scores."""
    matched = [a for a in alignments if a.source_triple and a.generated_triple]
    hallucinated = [a for a in alignments if a.source_triple is None]
    omitted = [a for a in alignments if a.generated_triple is None]

    total_gen = len(matched) + len(hallucinated)
    total_src = len(matched) + len(omitted)

    precision = len(matched) / total_gen if total_gen > 0 else 1.0
    recall = len(matched) / total_src if total_src > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Severity-weighted factuality score
    severity_weights = {"critical": 1.0, "high": 0.7, "medium": 0.3}
    penalty = sum(severity_weights.get(d.type.severity, 0.3) for d in divergences)
    max_penalty = max(total_gen, 1)
    factuality = max(0.0, 1.0 - penalty / max_penalty)

    return VerificationScores(
        triple_precision=precision,
        triple_recall=recall,
        triple_f1=f1,
        factuality_score=factuality,
        num_hallucinated=len(hallucinated),
        num_omitted=len(omitted),
        num_divergent=len([a for a in matched if a.divergences]),
    )
```

### 9. Integration Points with hopsparser

We depend on hopsparser for **one thing only**: turning text into `DepGraph` objects.

```
depver imports:
    hopsparser.parser.BiAffineParser   → .load(), .parse()
    hopsparser.deptree.DepGraph        → .nodes, .words, .deprels, .heads
    hopsparser.deptree.DepNode         → .identifier, .form, .lemma, .upos, .head, .deprel
```

This means:
- **depver is a separate package** — it doesn't modify hopsparser at all
- hopsparser is a dependency, listed in `depver/pyproject.toml`
- If someone wants to use a different parser (Stanza, spaCy), they only need to produce `DepGraph` objects or we add a thin adapter

### 10. Testing Strategy

```
tests/
├── test_extraction.py      # Gold UD trees → known triples
│   - Test verb predicate detection
│   - Test nsubj/obj/obl collection
│   - Test negation detection
│   - Test copular constructions
│   - Test passive voice
│   - Test relative clauses, advcl
│
├── test_comparison.py       # Synthetic triple pairs → known alignments
│   - Test exact match alignment
│   - Test verb substitution detection
│   - Test argument swap detection
│   - Test negation flip detection
│   - Test hallucination (unmatched gen triple)
│   - Test omission (unmatched source triple)
│
├── test_scoring.py          # Known alignments → expected scores
│   - Test precision/recall/F1 computation
│   - Test severity weighting
│
├── test_pipeline.py         # End-to-end with mock parser
│   - Test full verify() flow
│   - Test with real hopsparser (integration test, marked slow)
│
└── conftest.py              # Fixtures: sample DepGraphs, triples, etc.
    - build_graph(): helper to create DepGraph from simple dict spec
    - sample_trees: common test sentences as DepGraphs
```

**Key testing principle:** Unit tests for extraction and comparison use hand-constructed `DepGraph` objects (no parser needed). Integration tests use the real parser and are marked `@pytest.mark.slow`.

### 11. Development Order

```
Step 1: depver/schema.py
        Define all data types. No dependencies.
        Test: just import and instantiate.

Step 2: depver/extraction/triples.py
        Implement extract_triples() for core cases.
        Test: hand-build DepGraphs, verify extracted triples.
        Focus on: nsubj, obj, obl, negation, voice.

Step 3: depver/extraction/triples.py (extended)
        Add advcl, ccomp, xcomp, relcl handling.
        Add copular construction support.
        Test: more complex sentences.

Step 4: depver/comparison/similarity.py
        Implement predicate and entity similarity.
        Start with exact lemma match only (no WordNet/embeddings).
        Test: pairs of entities and predicates.

Step 5: depver/comparison/matcher.py
        Implement greedy triple alignment.
        Test: synthetic source/generated triple lists.

Step 6: depver/comparison/divergence.py
        Implement divergence classification rules.
        Test: aligned pairs with known divergence types.

Step 7: depver/scoring/metrics.py
        Implement scoring.
        Test: known alignments → expected scores.

Step 8: depver/pipeline.py
        Wire everything together.
        Integration test with real hopsparser.

Step 9: depver/comparison/similarity.py (enhanced)
        Add WordNet synonyms.
        Add embedding-based similarity (sentence-transformers).
        Ablation: measure improvement over exact match.

Step 10: experiments/eval_frank.py
         Run on FRANK benchmark.
         Compare against baselines.
```

### 12. Dependencies (`depver/pyproject.toml`)

```toml
[project]
name = "depver"
version = "0.1.0"
description = "Dependency-structure verification of LLM outputs"
requires-python = ">= 3.11"
dependencies = [
    "hopsparser >= 0.8.0",
]

[project.optional-dependencies]
wordnet = ["nltk >= 3.8"]
embeddings = ["sentence-transformers >= 2.0"]
eval = [
    "datasets",       # for loading FRANK, AggreFact
    "pandas",
    "scikit-learn",   # for metrics
    "matplotlib",
]
tests = [
    "pytest >= 7.0",
    "pytest-mock",
]

[project.scripts]
depver = "depver.pipeline:main"
```

### 13. Open Design Questions

| Question | Options | Decision Needed By |
|----------|---------|-------------------|
| Should we support parsers other than hopsparser? | (a) hopsparser only (b) adapter pattern for Stanza/spaCy | Phase 1 — affects extraction interface |
| Greedy vs. Hungarian alignment? | Greedy is simpler; Hungarian is optimal | Phase 2 — greedy first, upgrade if needed |
| How to handle multi-sentence source/generated? | (a) Sentence-level alignment first, then triple-level (b) Pool all triples | Phase 2 |
| WordNet vs. embeddings vs. both for predicate matching? | Start exact-only, add layers | Phase 3 — ablation will tell us |
| French negation / modality words? | Need language-specific lists | Phase 1 if multilingual from start |
| How to handle coordination ("A and B signed")? | conj relation → duplicate triple with each conjunct as subject? | Phase 1 |
