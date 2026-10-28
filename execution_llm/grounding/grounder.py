"""Runtime grounding of an emitted IR document: text -> `DecisionBody`.

The parser model writes the IR in the line format of `tier2/emit.py`; this module parses it,
covers every frame against the domain's signatures through `ground()` in `grounding/cover.py`
(a complete cover executes, an unresolved hole clarifies, no complete cover abstains), and applies
the composition line to the decision's `objectives` (an `ObjectiveComposition`). It is a pure
function of (text, registry projection).

`ground_full` also returns what the DecisionBody cannot carry: a stated vehicle count (a fleet
directive that whoever executes the decision resolves against its data) and notices for the default
readings it applied. `ground_parsed` / `ground_ir` return the DecisionBody alone and therefore withhold
an Execute that depends on a fleet directive. The routing rank weights apply only on the caller's opt-in
(`rank_weights`): the IR cannot tell a rank-only priority from an explicit "no trade-off" one, so only a
caller that has read the utterance (`execution_llm.grounding.decide`) can choose the rank-weight reading.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction

from execution_llm.grading import _canonicalize, _check_required_completeness
from execution_llm.grounding.cover import (
    Frame,
    _DuplicateLanding,
    _NoObjective,
    cover,
    ground,
    unit_misfit,
)
from execution_llm.grounding.inputs import SamplerInputs, _Blocked, _clarify, _finalize_abstention
from execution_llm.schema.decision import (
    DecisionBody,
    DecisionFieldInput,
    ExecuteBody,
    Lexicographic,
    ObjectiveTerm,
    UnsupportedBody,
    WeightedSum,
    WeightedTerm,
    _validate_membership,
)
from execution_llm.tier2.emit import Composition, ParsedProblem, parse_problem
from execution_llm.tier2.expr import Node, has_holes, normalize, parse, walk
from execution_llm.tier2.surface import DomainSurface

# Version of the grounding rules (date-based); it changes whenever the same IR can ground to a different
# decision.
# 2026-09-16.1: an unresolved hole clarifies even when only one signature could fill it.
# 2026-10-19.1: unit, population, mark and free-variable guards; a problem with no objective clarifies; one
# constraint stated both hard and soft is Unsupported; fleet directives; routing rank weights; the retail
# stockout objective.
# 2026-10-24.1: UNSPECIFIED aggregations through the plain-reading table, UNEXPRESSED residual lines and
# UNDECIDED bound values.
GROUNDER_VERSION = "2026-10-24.1"

# Rank-only priority ("X first, then Y", no numbers) on a stack that declares no lexicographic method
# executes as a weighted sum with these per-domain weights (routing 10:1). Nurse and retail define none.
# Opt-in per call (`ground_full(rank_weights=True)`); by default such a lexicographic line is
# Unsupported(unsupported_composition).
RANK_WEIGHTS: dict[str, tuple[str, str]] = {"routing": ("10/11", "1/11")}


class GroundingError(ValueError):
    """The IR parsed but cannot be grounded: a population slot that needs catalog members, one
    constraint stated twice with the same strength, a composition line that names a missing objective,
    an unknown rule family, ..."""


@dataclass(frozen=True)
class Notice:
    code: str  # rank_weighted | stockout_objective | unit_misfit | undecided_bound
    text: str  # Korean, at most 120 characters, no identifiers (shown to end users)


RANK_NOTICE = Notice("rank_weighted", "순위만 말씀하셔서 앞선 목표 10, 다음 목표 1의 비중으로 반영했습니다.")
STOCKOUT_NOTICE = Notice("stockout_objective", "품절이 나지 않게 해 달라는 요청을 품절 최소화 목표로 반영했습니다.")
# A stated value whose unit misfits its slot: `cover` replaces it with HOLE(value), so the value is not used.
UNIT_NOTICE = Notice("unit_misfit", "단위가 맞지 않는 값이 있어 그 조건은 반영하지 않았습니다.")
# A bound whose value the user has not decided yet (UNDECIDED) is left out of the problem.
UNDECIDED_NOTICE = Notice(
    "undecided_bound", "값을 아직 정하지 않은 조건은 빼고 반영했습니다. 값을 정하면 추가하겠습니다."
)


@dataclass(frozen=True)
class FleetDirective:
    """A stated vehicle count, `COUNT(x in routes) <= | = | >= N`. The grounder cannot see the fleet:
    whoever executes the decision must keep only the first ``count`` vehicles in data order. ``use_all``
    (stated ">= N", "use all N vehicles") means the decision already holds minimum_order target_order 1
    (added unless a frame already lands there)."""

    count: int
    use_all: bool


@dataclass(frozen=True)
class Grounding:
    """The decision plus what it cannot carry; ``frames`` are the frames actually covered after the
    rewrites below, so a per-frame report that re-covers them agrees with the decision."""

    decision: DecisionBody
    fleet: FleetDirective | None = None
    notices: tuple[Notice, ...] = ()
    frames: tuple[Frame, ...] = ()


def ground_full(
    parsed: ParsedProblem, inputs: SamplerInputs, surface: DomainSurface, *, rank_weights: bool = False
) -> Grounding:
    """The decision with its fleet directive and notices. Raises GroundingError exactly where
    `ground_parsed` does (a same-strength repeated constraint, a malformed composition line, ...).
    ``rank_weights``: the caller read the utterance as a rank-only priority (no "no trade-off" cue), so
    a routing lexicographic line takes the `RANK_WEIGHTS` weights instead of unsupported_composition."""
    if parsed.clarify_intent:
        if parsed.frames or parsed.composition or parsed.no_intent:
            raise GroundingError("CLARIFY(intent) must stand alone")
        return Grounding(_finalize_abstention(_clarify("ambiguous_capability", candidates=[]), inputs))
    if parsed.no_intent:
        body = UnsupportedBody(kind="unsupported", reason_code="request_out_of_scope", related_refs=[])
        return Grounding(_finalize_abstention(body, inputs))
    result = _ground_frames(parsed, inputs, surface, rank_weights)
    if not parsed.residuals:
        return result
    # A stated requirement the grammar cannot hold (an UNEXPRESSED line) makes the whole problem
    # Unsupported(semantic_not_declared, []); a Clarify from an unresolved element of the same problem asks first
    unresolved = any(has_holes(frame.tree) or _unspecified(frame.tree) for frame in result.frames)
    if unresolved and result.decision.kind == "clarify":
        return result
    body = UnsupportedBody(kind="unsupported", reason_code="semantic_not_declared", related_refs=[])
    return Grounding(_finalize_abstention(body, inputs), result.fleet, result.notices, result.frames)


def _ground_frames(
    parsed: ParsedProblem, inputs: SamplerInputs, surface: DomainSurface, rank_weights: bool
) -> Grounding:
    """`ground_full` over the frames, residual lines aside."""
    frames, notices = _rewrite(parsed, surface)
    rule_holes = [frame.tree for frame in frames if frame.tree.get("op") == "frame_hole"]
    if rule_holes:
        # HOLE(constraint, family): the user named a rule loosely; the family's members are the
        # candidates (the frame-level counterpart of a measure hole with several covers)
        families = {tree["family"] for tree in rule_holes}
        if not families <= set(surface.spec.rule_families):
            raise GroundingError(f"unknown rule families: {sorted(families - set(surface.spec.rule_families))}")
        candidates = sorted({s.ref for s in surface.signatures if s.family in families})
        body = _finalize_abstention(_clarify("ambiguous_capability", candidates=candidates), inputs)
        return Grounding(body, notices=notices, frames=frames)
    plain = [node for frame in frames for node in _unspecified(frame.tree)]
    if plain:
        # A quantity named without its aggregation (UNSPECIFIED) clarifies; the domain's plain-reading
        # table offers the declared reading as the single candidate, or none
        refs: set[str] = set()
        for node in plain:
            measure = node["term"].get("ref") if node["term"].get("op") == "measure" else None
            if measure not in surface.spec.plain_readings:
                raise GroundingError(f"UNSPECIFIED over a measure with no plain reading: {measure}")
            refs.update(surface.spec.plain_readings[measure])
        body = _finalize_abstention(_clarify("ambiguous_capability", candidates=sorted(refs)), inputs)
        return Grounding(body, notices=notices, frames=frames)
    fleet = None
    counts = [(frame, *count) for frame in frames if (count := _fleet_count(frame.tree, surface)) is not None]
    if len(counts) == 1:
        count_frame, op, n = counts[0]
        if op == "ge" or count_frame.hardness != "SOFT":
            fleet = FleetDirective(count=n, use_all=op == "ge")
            frames = tuple(frame for frame in frames if frame is not count_frame)
            if fleet.use_all and "constraint:minimum_order" not in {r for f in frames for r in _landed(f, surface)}:
                frames += (Frame(parse(USE_ALL, surface.vocab), count_frame.hardness),)
    # several counts, or a soft fleet size, have no declared reading: those frames stay and cover nothing
    if any(unit_misfit(frame, surface) for frame in frames):
        notices += (UNIT_NOTICE,)
    try:
        grounded = ground(frames, inputs, surface)
    except _NoObjective:
        # A problem whose frames land on no objective asks for the objectives; no default objective is filled in
        body = _clarify("missing_value", inputs_=[DecisionFieldInput(kind="decision_field", field="objectives")])
        return Grounding(_finalize_abstention(body, inputs), fleet, notices, frames)
    except _DuplicateLanding as error:
        if not error.mixed:  # one constraint stated twice with the same strength (an edit) has no declared reading
            raise GroundingError(str(error)) from error
        body = UnsupportedBody(kind="unsupported", reason_code="semantic_not_declared", related_refs=[])
        return Grounding(_finalize_abstention(body, inputs), fleet, notices, frames)
    except _Blocked as error:
        raise GroundingError(str(error)) from error
    gold = grounded.gold
    if not isinstance(gold, ExecuteBody):
        return Grounding(gold, fleet, notices, frames)
    # objectives by landed ref, in frame order: a registry constraint may be realized as a
    # minimized penalty term, so the frame's root op does not decide the kind
    objective_refs = [
        fc.matches[0][0].ref for fc in grounded.covers if fc.matches and fc.matches[0][0].ref.startswith("objective:")
    ]
    composition, rank = parsed.composition, RANK_WEIGHTS.get(inputs.domain)
    if (
        composition is not None
        and composition.method == "lexicographic"
        and "lexicographic" not in inputs.snapshot.allowed_composition_methods
        and rank is not None
        and rank_weights
        and len(objective_refs) == 2
        and sorted(composition.order) == [1, 2]
    ):
        # The IR cannot tell a rank-only statement ("X first, then Y") from an explicit "no trade-off"
        # lexicographic one, so only a caller that read the utterance opts in. The decision does not
        # say whether objective terms are normalized for their units before weighting.
        composition = Composition("weighted_sum", weights=dict(zip(composition.order, rank, strict=True)))
        notices += (RANK_NOTICE,)
    return Grounding(_compose(gold, objective_refs, composition, inputs), fleet, notices, frames)


def ground_parsed(parsed: ParsedProblem, inputs: SamplerInputs, surface: DomainSurface) -> DecisionBody:
    """The DecisionBody alone. A caller that cannot see the fleet directive would run every vehicle,
    so an Execute that depends on one is withheld as Unsupported(semantic_not_declared)."""
    result = ground_full(parsed, inputs, surface)
    if result.fleet is not None and isinstance(result.decision, ExecuteBody):
        body = UnsupportedBody(kind="unsupported", reason_code="semantic_not_declared", related_refs=[])
        return _finalize_abstention(body, inputs)
    return result.decision


_POSITIVE = re.compile(r"^[1-9][0-9]*$")
USE_ALL = "EACH(r in routes): delivery_count(r) >= 1"  # every vehicle serves at least one delivery
STOCKOUT_OBJECTIVE = "MAXIMIZE( in_stock_rate )"
NO_STOCKOUT = "EACH(d in days(horizon)): in_stock(d)"
_DAYS = ({"op": "days", "horizon": "entity:horizon"}, {"op": "pop", "ref": "entity:day"})
_ROUTES = {"op": "pop", "ref": "entity:route"}
_COMPARISONS = ("le", "lt", "eq", "ge", "gt", "ne")


def rewrite_frames(parsed: ParsedProblem, surface: DomainSurface) -> tuple[Frame, ...]:
    """The frames as the decision lands them, for a per-frame report: frames with an UNDECIDED bound
    dropped, the retail stockout reading applied and every vehicle count that reads as a fleet directive
    removed (the decision carries it as a `FleetDirective`, or abstains on it); frame holes pass through
    untouched. Never raises on parsed input, and idempotent."""
    frames, _ = _rewrite(parsed, surface)
    return tuple(frame for frame in frames if _fleet_count(frame.tree, surface) is None)


def is_fleet_frame(tree: Node, surface: DomainSurface) -> bool:
    """A routing vehicle-count statement: COUNT over the whole route population (riders included; a
    WHERE subset is not the fleet) compared with a stated number, whether or not it reads as a fleet
    directive (strict comparisons and invalid numbers included). False for a frame hole and in every
    other domain; never raises."""
    return _count_statement(tree, surface) is not None


def _unspecified(tree: Node) -> list[Node]:
    """The UNSPECIFIED aggregation nodes of a frame."""
    return [n for n in walk(tree) if n.get("op") == "unspecified"]


def _rewrite(parsed: ParsedProblem, surface: DomainSurface) -> tuple[tuple[Frame, ...], tuple[Notice, ...]]:
    """The default readings no signature carries, applied before the cover. Retail: an "avoid stockouts"
    frame becomes the stockout objective in its place when soft or unmarked (once, and not at all when the
    objective is already stated); a hard one becomes the hole-free day form, which no declared constraint
    covers. A frame whose bound is UNDECIDED is dropped, with a notice. Every other frame (a frame hole, a
    vehicle count) passes through."""
    undecided = [tree for tree, _ in parsed.frames if any(n.get("op") == "undecided" for n in walk(tree))]
    if undecided:
        kept = tuple(f for f in parsed.frames if not any(f[0] is tree for tree in undecided))
        frames, notices = _rewrite(ParsedProblem(kept, parsed.composition), surface)
        return frames, (*notices, UNDECIDED_NOTICE)
    given = [Frame(tree, hardness) for tree, hardness in parsed.frames]
    stockouts = [frame.tree.get("op") != "frame_hole" and _stockout(frame.tree, surface) for frame in given]
    stated = any(stockouts) and any(
        "objective:minimize_stockout" in _landed(frame, surface)
        for frame, stockout in zip(given, stockouts, strict=True)
        if not stockout
    )
    frames: list[Frame] = []
    notices: tuple[Notice, ...] = ()
    for frame, stockout in zip(given, stockouts, strict=True):
        if stockout and frame.hardness == "HARD":  # no declared no-stockout constraint; the day form is hole-free
            frame = Frame(parse(NO_STOCKOUT, surface.vocab), "HARD")
        elif stockout:
            notices = (STOCKOUT_NOTICE,)
            frame = Frame(parse(STOCKOUT_OBJECTIVE, surface.vocab))
            if stated or frame in frames:
                continue  # the objective is already there: never ground it twice
        frames.append(frame)
    return tuple(frames), notices


def _normal(tree: Node, surface: DomainSurface) -> Node | None:
    """The normal form, or None for a tree that has none (a cyclic binder chain): such a tree takes no
    default reading, and covering it raises the parse-level error."""
    try:
        return normalize(tree, surface.vocab)
    except ValueError:
        return None


def _landed(frame: Frame, surface: DomainSurface) -> set[str]:
    """The refs the frame covers (none for a tree without a normal form)."""
    try:
        return {signature.ref for signature, _ in cover(frame, surface).matches}
    except ValueError:
        return set()


def _count_statement(tree: Node, surface: DomainSurface) -> Node | None:
    """The normalized comparison of `is_fleet_frame`, or None."""
    if surface.vocab.domain != "routing" or tree.get("op") == "frame_hole":
        return None
    n = _normal(tree, surface)
    if n is None or n.get("op") not in _COMPARISONS:
        return None
    left, right = n["left"], n["right"]
    if left.get("op") == "count" and left.get("over") == _ROUTES and right.get("op") == "value":
        return n
    return None


def _fleet_count(tree: Node, surface: DomainSurface) -> tuple[str, int] | None:
    """(comparator, N) for a fleet directive, `COUNT(x in routes) <=|=|>= N` with N a positive integer
    and no unit. Other vehicle counts (strict, zero, fractional, with a unit) have no declared reading."""
    n = _count_statement(tree, surface)
    if n is None or n["op"] not in ("le", "eq", "ge") or n["right"].get("unit") is not None:
        return None
    if not _POSITIVE.match(n["right"]["num"]):
        return None
    return n["op"], int(n["right"]["num"])


def _stockout(tree: Node, surface: DomainSurface) -> bool:
    """Retail `EACH(x in days(horizon) | days | HOLE(scope)): in_stock(x)`, the "avoid stockouts" form."""
    n = _normal(tree, surface) if surface.vocab.domain == "retail" else None
    if n is None:
        return False
    body = n.get("body") or {}
    return (
        (n.get("op") == "scope_hole" or (n.get("op") == "each" and n.get("over") in _DAYS))
        and body.get("op") == "attr"
        and body.get("ref") == "attr:in_stock"
        and body.get("args") == [{"op": "var", "ref": n.get("var")}]
    )


def _default_weights(objective_refs: list[str], inputs: SamplerInputs) -> dict[str, str]:
    """Weights for an unstated composition, which never clarifies. The semantic projection's
    `default_weights` apply to the full declared set; a subset takes the declared weights of its
    members renormalized to sum to one; otherwise each objective gets 1/n."""
    declared = list(inputs.semantic.objectives)
    weights = list(inputs.semantic.default_weights or ())
    if len(weights) == len(declared) and weights:
        by_ref = {o.ref: Fraction(w) for o, w in zip(declared, weights, strict=True)}
        picked = {ref: by_ref[ref] for ref in objective_refs if ref in by_ref}
        total = sum(picked.values(), Fraction(0))
        if len(picked) == len(objective_refs) and total > 0:
            return {ref: _fraction(w / total) for ref, w in picked.items()}
    return {ref: _fraction(Fraction(1, len(objective_refs))) for ref in objective_refs}


def _fraction(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _stated_weights(objective_refs: list[str], composition: Composition) -> dict[str, str]:
    """Stated weights win; the objectives the user left unweighted share the remainder
    (1 - the stated weights) equally."""
    stated: dict[str, Fraction] = {}
    for index, weight in composition.weights.items():
        if not 1 <= index <= len(objective_refs):
            raise GroundingError(f"COMPOSE names an objective index that has no frame: {index}")
        try:
            stated[objective_refs[index - 1]] = Fraction(weight)
        except (ValueError, ZeroDivisionError) as error:  # e.g. "1/0" from output decoded without the grammar
            raise GroundingError(f"COMPOSE weight is not a valid fraction: {weight!r}") from error
    rest = [ref for ref in objective_refs if ref not in stated]
    out = {ref: _fraction(w) for ref, w in stated.items()}
    if rest:
        remainder = Fraction(1) - sum(stated.values(), Fraction(0))
        if remainder <= 0:
            raise GroundingError("stated weights leave nothing for the unweighted objectives")
        out.update({ref: _fraction(remainder / len(rest)) for ref in rest})
    return out


def _compose(
    body: ExecuteBody, objective_refs: list[str], composition: Composition | None, inputs: SamplerInputs
) -> DecisionBody:
    """Apply the composition line under the stack's declared composition methods (fewer than two
    objectives need none). With no line the objectives take the default weights as a weighted sum. A
    lexicographic line on a stack without that method is Unsupported(unsupported_composition); a weighted
    sum (stated or default) the stack does not declare, or a product, clarifies with
    composition_choice_required."""
    if len(objective_refs) < 2:
        return body
    allowed = set(inputs.snapshot.allowed_composition_methods)
    if composition is None:
        weights = _default_weights(objective_refs, inputs)
        composition = Composition("weighted_sum", weights={i + 1: weights[ref] for i, ref in enumerate(objective_refs)})
    if composition.method == "weighted_sum" and "weighted_sum" in allowed:
        weights = _stated_weights(objective_refs, composition)
        terms = [WeightedTerm(objective_ref=ref, weight=weights[ref]) for ref in objective_refs]
        objectives = WeightedSum(method="weighted_sum", terms=sorted(terms, key=lambda t: t.objective_ref))
    elif composition.method == "lexicographic":
        if "lexicographic" not in allowed:
            body_u = UnsupportedBody(
                kind="unsupported", reason_code="unsupported_composition", related_refs=sorted(objective_refs)
            )
            return _finalize_abstention(body_u, inputs)
        try:
            ordered = [objective_refs[i - 1] for i in composition.order]
        except IndexError as error:
            raise GroundingError("COMPOSE order names a missing objective frame") from error
        objectives = Lexicographic(method="lexicographic", terms=[ObjectiveTerm(objective_ref=r) for r in ordered])
    else:
        clarify = _clarify("composition_choice_required", candidates=objective_refs)
        return _finalize_abstention(clarify, inputs)
    composed = body.model_copy(update={"objectives": objectives})
    _validate_membership(composed, inputs.snapshot)
    _check_required_completeness(composed, inputs.snapshot, inputs.signature)
    canonical = _canonicalize(composed, inputs.snapshot, inputs.signature)
    assert isinstance(canonical, ExecuteBody)
    return canonical


def ground_ir(text: str, inputs: SamplerInputs, surface: DomainSurface) -> DecisionBody:
    """Parse the IR text and ground it. Raises ValueError (parse) or GroundingError."""
    return ground_parsed(parse_problem(text, surface.vocab), inputs, surface)
