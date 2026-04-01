# Dependency-Structure Verification of LLM Outputs

## Using Syntactic Parsing for Factuality Checking and Hallucination Detection

---

## 1. Problem Statement

Large Language Models generate fluent text, but fluency masks factual errors. Current verification methods rely on:

- **Token-level overlap** (ROUGE, BLEU) — misses semantic distortion with high lexical overlap
- **Embedding similarity** (BERTScore) — captures meaning loosely but ignores relational structure
- **LLM-as-judge** — expensive, non-deterministic, and prone to the same failure modes

None of these capture **who did what to whom**. A sentence can score high on all three metrics while having a swapped subject, a negated verb, or a hallucinated causal link.

**Key insight:** Dependency parsing decomposes sentences into typed, directed relations between words. These relations are comparable algebraically — enabling precise, interpretable verification that text similarity cannot provide.

---

## 2. Core Idea

```
                    ┌──────────────┐
Source text ───────→│              │──→ Source triples    ──┐
                    │  hopsparser  │                        ├──→ Structural comparison ──→ Verdict
Generated text ───→│              │──→ Generated triples ──┘
                    └──────────────┘
```

1. Parse both source and generated text into dependency trees
2. Extract **relational triples** from the trees: `(subject, predicate, object, modifiers)`
3. Compare triples structurally to detect divergence
4. Classify divergence types (factual error, hallucination, omission, editorialization)

---

## 3. Relational Triple Extraction

### 3.1 From Dependency Tree to Triples

Given a dependency tree in CoNLL-U format, extract structured relations by walking the tree from each predicate (verb) node.

**Extraction rules based on Universal Dependencies relations:**

| UD Relation | Role in Triple | Example |
|-------------|---------------|---------|
| `nsubj` | Agent / Subject | *Macron* signed the bill |
| `obj` | Patient / Object | Macron signed *the bill* |
| `iobj` | Recipient | gave *him* the book |
| `obl` | Circumstance (time, location, instrument) | signed in *March* |
| `nmod` | Noun modifier (possession, partitive) | CEO of *Apple* |
| `amod` | Adjectival modifier | the *controversial* bill |
| `advmod` | Adverbial modifier | signed *quickly* |
| `mark` | Clause marker (causation, condition) | *because* revenue grew |
| `advcl` | Adverbial clause (causal, temporal) | Revenue grew, *causing* layoffs |
| `xcomp` / `ccomp` | Clausal complement | said *that he resigned* |
| `neg` / `advmod` with polarity | Negation | did *not* sign |

### 3.2 Triple Schema

```
Triple:
  predicate   : str          # head verb lemma
  subject     : Entity       # nsubj subtree
  object      : Entity       # obj subtree (optional)
  obliques    : list[Oblique] # obl subtrees (time, place, etc.)
  negated     : bool         # presence of negation
  modality    : str | None   # aux: "can", "must", "might", etc.
  clause_type : str          # root, advcl, xcomp, ccomp, relcl

Entity:
  head_lemma  : str          # head noun lemma
  modifiers   : list[str]    # amod, nummod, det values
  nmod_chain  : list[Entity] # nested nmod ("CEO of Apple of America")

Oblique:
  case        : str          # preposition / case marker
  entity      : Entity       # the governed noun phrase
```

### 3.3 Example

**Sentence:** *"Macron signed the controversial climate bill in March."*

**Dependency tree:**
```
signed ──nsubj──→ Macron
signed ──obj────→ bill
bill   ──det────→ the
bill   ──amod───→ controversial
bill   ──amod───→ climate
signed ──obl────→ March
March  ──case───→ in
```

**Extracted triple:**
```
Triple(
  predicate = "sign",
  subject   = Entity(head_lemma="Macron", modifiers=[], nmod_chain=[]),
  object    = Entity(head_lemma="bill", modifiers=["controversial", "climate"], nmod_chain=[]),
  obliques  = [Oblique(case="in", entity=Entity(head_lemma="March"))],
  negated   = False,
  modality  = None,
  clause_type = "root"
)
```

---

## 4. Structural Comparison

### 4.1 Matching Algorithm

Given source triples `S` and generated triples `G`:

```
For each triple g in G:
    Find best matching triple s in S using:
        score = w1 * predicate_match(g, s)
              + w2 * subject_match(g, s)
              + w3 * object_match(g, s)
              + w4 * oblique_match(g, s)
              + w5 * polarity_match(g, s)

    If score > threshold:
        Mark (g, s) as aligned pair
        Check for partial divergences within the pair
    Else:
        Mark g as ungrounded (potential hallucination)

For each triple s in S with no aligned g:
    Mark s as omitted
```

### 4.2 Match Functions

**Predicate matching:**
- Exact lemma match → 1.0
- WordNet synonym → 0.8
- Same VerbNet frame → 0.7
- Embedding cosine similarity > 0.85 → 0.6
- Entailment (signed → agreed, but signed ⊬ vetoed) → requires NLI model

**Entity matching:**
- Head lemma match + overlapping modifiers → 1.0
- Head lemma match, different modifiers → 0.7 (possible editorialization)
- Coreferent (resolved via simple heuristics or coref model) → 0.9
- No match → 0.0

**Polarity matching:**
- Same negation status → 1.0
- Different → 0.0 (critical divergence)

### 4.3 Divergence Classification

| Divergence Type | Structural Signal | Severity |
|----------------|-------------------|----------|
| **Verb substitution** | Same nsubj+obj frame, different predicate | High |
| **Argument swap** | nsubj↔obj swapped between s and g | High |
| **Negation flip** | Negation added or removed | Critical |
| **Entity hallucination** | Entity in g has no correspondent in any s | High |
| **Modifier hallucination** | amod/nummod in g absent from s | Medium |
| **Causal hallucination** | advcl with causal mark in g, independent clauses in s | High |
| **Attribution shift** | Different nsubj on reporting verb (said, confirmed, denied) | High |
| **Temporal shift** | Different obl:tmod or nmod:tmod | Medium |
| **Editorialization** | Added evaluative amod (controversial, unprecedented) | Low-Medium |
| **Omission** | Triple in s with no match in g | Varies |

---

## 5. Concrete Detection Examples

### 5.1 Verb Substitution (Factual Error)

```
Source:    "The company reported a 5% revenue increase."
Generated: "The company denied a 5% revenue increase."

Source triple:    (company, report, increase[5%, revenue])
Generated triple: (company, deny,   increase[5%, revenue])

→ Subject match: ✓ | Object match: ✓ | Predicate match: ✗ (report ≠ deny)
→ Classification: VERB_SUBSTITUTION, severity=HIGH
→ Explanation: "Generated text uses 'denied' where source says 'reported'"
```

### 5.2 Argument Swap (Factual Error)

```
Source:    "Google acquired YouTube."
Generated: "YouTube acquired Google."

Source triple:    (Google, acquire, YouTube)
Generated triple: (YouTube, acquire, Google)

→ Predicate match: ✓ | Arguments: swapped nsubj↔obj
→ Classification: ARGUMENT_SWAP, severity=HIGH
```

### 5.3 Causal Hallucination

```
Source:    "Revenue declined. The CEO resigned."
Generated: "Revenue declined, leading to the CEO's resignation."

Source:     two independent triples, no syntactic link
Generated:  advcl(resigned, declined, mark=leading_to)

→ Causal dependency exists in generated but not in source
→ Classification: CAUSAL_HALLUCINATION, severity=HIGH
```

### 5.4 Modifier Hallucination

```
Source:    "The committee approved the budget."
Generated: "The committee approved the $4.2 billion budget."

Source triple:    (committee, approve, budget[])
Generated triple: (committee, approve, budget[$4.2 billion])

→ Frame match: ✓ | Extra nummod "$4.2 billion" ungrounded
→ Classification: MODIFIER_HALLUCINATION, severity=MEDIUM
```

### 5.5 Negation Flip

```
Source:    "The study confirmed the hypothesis."
Generated: "The study did not confirm the hypothesis."

Source triple:    (study, confirm, hypothesis, negated=False)
Generated triple: (study, confirm, hypothesis, negated=True)

→ Full frame match but polarity inverted
→ Classification: NEGATION_FLIP, severity=CRITICAL
```

---

## 6. Architecture

### 6.1 Module Design

```
depver/                          # "dependency verification"
├── __init__.py
├── extraction/
│   ├── __init__.py
│   ├── triples.py              # DepGraph → list[Triple]
│   ├── entities.py             # subtree → Entity
│   └── schema.py               # Triple, Entity, Oblique dataclasses
├── comparison/
│   ├── __init__.py
│   ├── matcher.py              # align triples between source and generated
│   ├── similarity.py           # predicate/entity similarity functions
│   └── divergence.py           # classify divergence types
├── scoring/
│   ├── __init__.py
│   ├── metrics.py              # precision, recall, F1 over triples
│   └── report.py               # human-readable divergence report
├── pipeline.py                 # end-to-end: text → parse → extract → compare → score
└── server.py                   # optional FastAPI endpoint wrapping pipeline
```

### 6.2 Integration with hopsparser

```python
from hopsparser.parser import BiAffineParser
from hopsparser.deptree import DepGraph
from depver.extraction.triples import extract_triples
from depver.comparison.matcher import align_triples
from depver.comparison.divergence import classify_divergences
from depver.scoring.metrics import compute_scores

# Load parser once
parser = BiAffineParser.load("model_path")

def verify(source_text: str, generated_text: str) -> VerificationResult:
    # 1. Parse
    source_graphs: list[DepGraph] = list(parser.parse(
        source_text.splitlines(), raw=True
    ))
    generated_graphs: list[DepGraph] = list(parser.parse(
        generated_text.splitlines(), raw=True
    ))

    # 2. Extract triples
    source_triples = [t for g in source_graphs for t in extract_triples(g)]
    gen_triples = [t for g in generated_graphs for t in extract_triples(g)]

    # 3. Align and compare
    alignments = align_triples(source_triples, gen_triples)
    divergences = classify_divergences(alignments)

    # 4. Score
    scores = compute_scores(alignments)

    return VerificationResult(
        scores=scores,
        divergences=divergences,
        alignments=alignments,
    )
```

### 6.3 Data Flow

```
Raw text
  │
  ▼
hopsparser.parse(raw=True)
  │
  ▼
list[DepGraph]                    # CoNLL-U: form, lemma, upos, head, deprel per token
  │
  ▼
extract_triples(graph)
  │  Walk from each verb (upos=VERB)
  │  Collect nsubj, obj, obl, modifiers
  │  Resolve negation, modality
  │
  ▼
list[Triple]                      # structured (predicate, subject, object, ...)
  │
  ▼
align_triples(source, generated)
  │  Hungarian algorithm or greedy best-match
  │  Similarity = weighted sum of component matches
  │
  ▼
list[Alignment]                   # paired triples + similarity scores
  │
  ▼
classify_divergences(alignments)
  │  Check each aligned pair for specific divergence patterns
  │  Flag unaligned generated triples as hallucinations
  │
  ▼
VerificationResult                # scores + itemized divergences
```

---

## 7. Evaluation Plan

### 7.1 Datasets

| Dataset | Task | Why Useful |
|---------|------|------------|
| **FRANK** (Pagnoni et al., 2021) | Summarization factuality | Human-annotated factual errors in generated summaries, categorized by type |
| **AggreFact** (Tang et al., 2023) | Aggregated factuality benchmark | Unified benchmark across multiple summarization factuality datasets |
| **TRUE** (Honovich et al., 2022) | Cross-task factuality | Covers summarization, dialogue, paraphrase, fact verification |
| **SummEval** (Fabbri et al., 2021) | Summary quality | Human ratings for consistency, relevance, fluency, coherence |
| **XSumFaith** (Maynez et al., 2020) | Hallucination in abstractive summarization | Annotated intrinsic/extrinsic hallucinations |

### 7.2 Baselines to Compare Against

| Method | Type | Strength | Weakness |
|--------|------|----------|----------|
| ROUGE-L | Token overlap | Fast, standard | Misses semantic divergence |
| BERTScore | Embedding similarity | Captures paraphrase | Ignores relational structure |
| QuestEval | QA-based | Good at content verification | Slow (generates + answers questions) |
| FactCC | NLI-based | Trained for factuality | Binary, not interpretable |
| DAE (Goyal & Durrett, 2021) | Dependency arc entailment | Closest prior work | Per-arc, not per-triple; no hallucination typing |
| LLM-as-judge (GPT-4) | Prompting | Flexible | Expensive, non-deterministic, not reproducible |

### 7.3 Metrics

- **Detection:** Precision / Recall / F1 for identifying sentences with factual errors
- **Classification:** Accuracy of divergence type classification (verb substitution, argument swap, etc.)
- **Correlation:** Spearman/Kendall correlation with human factuality judgments
- **Granularity:** Percentage of errors where the system can point to the specific words responsible (interpretability advantage)

---

## 8. Research Questions

1. **Does structural comparison outperform token/embedding similarity for factual error detection?**
   Hypothesis: Yes, especially for argument swaps, negation flips, and causal hallucinations where surface text remains similar.

2. **Which divergence types are most reliably detected?**
   Hypothesis: Negation flips and argument swaps (clear structural signal) > modifier hallucination (requires grounding judgment) > verb substitution (requires semantic similarity).

3. **How does parser accuracy affect downstream verification?**
   Evaluate with gold parses vs. predicted parses to measure error propagation. If parser errors dominate, the approach loses value.

4. **Can structural verification complement embedding-based methods?**
   Hypothesis: An ensemble (BERTScore + structural) outperforms either alone — they catch different error types.

5. **Does this generalize across languages?**
   hopsparser supports French; UD treebanks exist for 100+ languages. Test on at least 2-3 languages to assess cross-linguistic validity.

---

## 9. Relation to Prior Work

**DAE (Goyal & Durrett, 2021)** is the closest prior work — it uses dependency arcs for entailment checking. Key differences in our approach:

| Aspect | DAE | This Proposal |
|--------|-----|---------------|
| Granularity | Per-arc entailment | Per-triple (predicate-argument structure) |
| Hallucination typing | Binary (entailed / not) | Categorized (verb sub, arg swap, negation, causal, modifier) |
| Matching | Learned NLI model over arcs | Structural alignment + similarity functions |
| Interpretability | Which arc failed | Which relation changed and how |
| Parser | External (Stanza) | Integrated (hopsparser, extensible) |

**Contribution over DAE:** Moving from arc-level to triple-level comparison enables detecting **multi-arc divergences** (e.g., causal hallucination requires seeing that two independent clauses were merged into one advcl structure — no single arc captures this).

---

## 10. Implementation Roadmap

### Phase 1: Triple Extraction (weeks 1-3)
- Implement `extract_triples()` over hopsparser `DepGraph` output
- Handle core relations: nsubj, obj, obl, amod, advmod, negation
- Unit tests against hand-annotated sentences
- Evaluate extraction quality on gold UD treebank parses

### Phase 2: Structural Comparison (weeks 4-6)
- Implement triple alignment (Hungarian algorithm)
- Implement similarity functions (lemma, WordNet, embedding)
- Implement divergence classification rules
- Test on synthetic examples (programmatically generated divergences)

### Phase 3: Evaluation (weeks 7-10)
- Run on FRANK and AggreFact benchmarks
- Compare against baselines (ROUGE, BERTScore, FactCC, DAE)
- Measure correlation with human judgments
- Ablation: gold parses vs. predicted parses
- Ablation: individual divergence type detection accuracy

### Phase 4: Extensions (weeks 11-14)
- Ensemble with BERTScore
- Cross-lingual evaluation (French, German)
- Integration as hopsparser module or standalone tool
- Optional: FastAPI endpoint for verification-as-a-service

---

## 11. Expected Contributions

1. **A typed taxonomy of structural divergences** detectable via dependency parsing, going beyond binary entailment
2. **A triple-level comparison framework** that captures multi-arc semantic changes (causal, attributional)
3. **Interpretable verification** — the system points to exactly which relation changed and how, unlike black-box NLI
4. **Empirical evaluation** showing where structural methods complement embedding-based approaches
5. **An open-source tool** integrated with hopsparser for practical use in LLM output verification pipelines
