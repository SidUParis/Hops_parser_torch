"""Core data types for DepVer."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


@dataclass(frozen=True)
class Modifier:
    """An adjectival, numeric, or determiner modifier."""

    text: str
    deprel: str  # "amod", "nummod", "det", "advmod"
    node_id: int


@dataclass(frozen=True)
class Entity:
    """A noun phrase extracted from a dependency subtree."""

    head_lemma: str
    head_form: str
    head_upos: str
    modifiers: tuple[Modifier, ...] = ()
    nmod_chain: tuple[Entity, ...] = ()
    node_ids: tuple[int, ...] = ()

    @property
    def signature(self) -> str:
        mods = "+".join(m.text for m in self.modifiers if m.deprel != "det")
        nmod = ">".join(e.head_lemma for e in self.nmod_chain)
        parts = [self.head_lemma]
        if mods:
            parts[0] += f"[{mods}]"
        if nmod:
            parts[0] += f"({nmod})"
        return parts[0]


class ClauseType(Enum):
    ROOT = auto()
    ADVCL = auto()
    XCOMP = auto()
    CCOMP = auto()
    RELCL = auto()
    PARATAXIS = auto()


@dataclass(frozen=True)
class Oblique:
    """A prepositional/case-marked argument."""

    case_marker: str
    entity: Entity


@dataclass(frozen=True)
class Triple:
    """A predicate-argument structure extracted from a dependency tree."""

    predicate_lemma: str
    predicate_form: str
    predicate_upos: str
    predicate_id: int

    subject: Entity | None = None
    object: Entity | None = None
    obliques: tuple[Oblique, ...] = ()

    negated: bool = False
    modality: str | None = None
    voice: str = "active"
    clause_type: ClauseType = ClauseType.ROOT
    clause_mark: str | None = None

    sentence_index: int = 0
    source_text: str = ""

    @property
    def signature(self) -> str:
        subj = self.subject.signature if self.subject else "_"
        obj = self.object.signature if self.object else "_"
        neg = "!" if self.negated else ""
        return f"{neg}{self.predicate_lemma}({subj}, {obj})"


class DivergenceType(Enum):
    VERB_SUBSTITUTION = "verb_substitution"
    ARGUMENT_SWAP = "argument_swap"
    NEGATION_FLIP = "negation_flip"
    ENTITY_HALLUCINATION = "entity_hallucination"
    MODIFIER_HALLUCINATION = "modifier_hallucination"
    CAUSAL_HALLUCINATION = "causal_hallucination"
    ATTRIBUTION_SHIFT = "attribution_shift"
    TEMPORAL_SHIFT = "temporal_shift"
    EDITORIALIZATION = "editorialization"
    OMISSION = "omission"

    @property
    def severity(self) -> str:
        critical = {DivergenceType.NEGATION_FLIP}
        high = {
            DivergenceType.VERB_SUBSTITUTION,
            DivergenceType.ARGUMENT_SWAP,
            DivergenceType.ENTITY_HALLUCINATION,
            DivergenceType.CAUSAL_HALLUCINATION,
            DivergenceType.ATTRIBUTION_SHIFT,
        }
        if self in critical:
            return "critical"
        elif self in high:
            return "high"
        return "medium"


@dataclass(frozen=True)
class Divergence:
    """A specific divergence found between aligned triples."""

    type: DivergenceType
    description: str
    source_span: str
    generated_span: str


@dataclass(frozen=True)
class Alignment:
    """A matched pair of source and generated triples."""

    source_triple: Triple | None
    generated_triple: Triple | None
    similarity_score: float
    divergences: tuple[Divergence, ...] = ()


@dataclass
class VerificationScores:
    """Aggregate scores for the verification."""

    triple_precision: float
    triple_recall: float
    triple_f1: float
    factuality_score: float
    num_hallucinated: int
    num_omitted: int
    num_divergent: int


@dataclass
class VerificationResult:
    """Complete output of the verification pipeline."""

    source_triples: list[Triple]
    generated_triples: list[Triple]
    alignments: list[Alignment]
    scores: VerificationScores
    divergences: list[Divergence]
