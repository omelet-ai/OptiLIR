"""Per-domain capability forms: what each declared capability means, as an IR expression keyed by its
registry ref, with the map from form slots ($name) to registry slots.

A domain is one `Vocabulary` plus its `Signature` tuple, collected in a `DomainSpec`; the prompts
and the grounding read each domain from its `DomainSurface`
(`execution_llm.tier2.domains.SURFACES` holds one surface per domain, built from its spec).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from execution_llm.tier2.expr import Node, Vocabulary, normalize, parse


@dataclass(frozen=True)
class SlotSpec:
    registry_ref: str
    unit: str | None  # the spoken unit the registry's value is read in (None: count/fraction)
    kind: str = "scalar"  # scalar | population | table
    sort: str = ""  # DISTANCE | DURATION | COUNT | FRACTION (derived from the unit when empty)

    @property
    def value_sort(self) -> str:
        return self.sort or {"m": "DISTANCE", "s": "DURATION", "h": "DURATION", "day": "COUNT"}.get(
            self.unit or "", "COUNT"
        )


@dataclass(frozen=True)
class Signature:
    ref: str  # registry ref (objective:... / constraint:...)
    text: str  # the entry's meaning in grammar notation
    slots: dict[str, SlotSpec]  # signature slot ref -> registry slot
    vocab: Vocabulary = field(repr=False, compare=False, hash=False, default=None)  # type: ignore[assignment]
    deferred_primitive: str | None = None  # a data primitive the entry needs that the registry defers
    # A rule family: the loose name users give to several structurally different constraints
    # ("the fatigue rule", "the ordering rules"); a frame hole HOLE(constraint, family) names it.
    family: str = ""
    # An opaque entry's form needs OPAQUE(...), which neither the prompt nor the decoder grammar admits,
    # so a parser that follows them never selects it (an IR written by hand with OPAQUE(...) still
    # covers it). The accessor sort OPAQUE in the routing prompt is a different thing: a value with no
    # structure the IR can use.
    opaque: bool = False

    def tree(self) -> Node:
        return parse(self.text, self.vocab)


@dataclass(frozen=True)
class Extra:
    """A user-IR tree the domain can express but no registry entry covers: a recognized but
    unrealized entry, an unrealized configuration of a declared family, or a reading with no
    registry ref."""

    name: str
    text: str  # the slot form; an uncovered frame that unifies with it is this extra
    note: str = ""
    # How such a frame is reported: a meaning outside the served vocabulary is semantic_not_declared
    # (no related refs); a recognized but unrealized entry, or an unrealized configuration of a
    # declared family, is realization_absent with `related_ref` as its related ref, which the
    # registry must declare (otherwise grounding raises GroundingError).
    terminal: str = "semantic_not_declared"
    related_ref: str | None = None


@dataclass(frozen=True)
class DomainSpec:
    vocab: Vocabulary
    signatures: tuple[Signature, ...]
    population_slot_default: Node  # what an unstated population slot means (all riders / all workers)
    extras: tuple[Extra, ...] = ()
    rule_families: Mapping[str, str] = field(default_factory=dict)  # family name -> gloss shown to the parser
    # The measures a plain expression may name without stating its aggregation (e.g. "공평하게" 'fairly',
    # "초과근무 줄여" 'cut overtime'), written UNSPECIFIED(x in P: m(x)); measure ref -> the declared
    # readings offered as Clarify candidates (empty: a Clarify with no candidate).
    plain_readings: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    conventions: tuple[str, ...] = ()  # domain notation conventions shown to the parser, one line each


def signature(vocab: Vocabulary, ref: str, text: str, slots: dict[str, SlotSpec] | None = None, **kw) -> Signature:
    return Signature(ref=ref, text=text, slots=slots or {}, vocab=vocab, **kw)


@dataclass(frozen=True)
class DomainSurface:
    vocab: Vocabulary
    signatures: tuple[Signature, ...]
    spec: DomainSpec
    normalized: Mapping[str, Node] = field(default_factory=dict)

    extras_normalized: tuple[tuple[Extra, Node], ...] = ()  # (Extra, normalized slot-form tree)

    @classmethod
    def build(cls, spec: DomainSpec) -> DomainSurface:
        normalized = {s.ref: normalize(s.tree(), spec.vocab) for s in spec.signatures}
        extras = tuple((x, normalize(parse(x.text, spec.vocab), spec.vocab)) for x in spec.extras)
        return cls(spec.vocab, spec.signatures, spec, normalized, extras)
