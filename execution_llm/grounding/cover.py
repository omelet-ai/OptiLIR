"""Frame cover and grounding: a problem is a list of IR frames; its decision comes from matching each
frame against the domain's capability forms:

- a hole-free frame covered by exactly one signature  -> Execute (ref + bound slots);
- a frame with an unresolved measure/scope hole       -> Clarify, even with zero or one cover;
- a frame with a value hole on a required slot        -> Clarify(missing_value);
- a frame covered by nothing that unifies with a declared `Extra` naming a registry ref
                                                      -> Unsupported(realization_absent);
- any other frame covered by nothing                  -> Unsupported(semantic_not_declared);
- a frame covered by a signature whose data primitive is deferred
                                                      -> Unsupported(required_primitive_absent).

`ground` returns the decision in canonical form (declared defaults filled in by
`grading._canonicalize`), the form of the dataset's `gold_decision`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal

from execution_llm.grading import (
    _canonicalize,
    _check_required_completeness,
    _default_kind,
)
from execution_llm.grounding.inputs import (
    SamplerInputs,
    _Blocked,
    _clarify,
    _execute,
    _finalize_abstention,
)
from execution_llm.schema.decision import (
    ConstraintSelection,
    DecimalValue,
    DecisionBody,
    DurationValue,
    ExecuteBody,
    HardStrength,
    IntegerValue,
    SlotArgument,
    SlotInput,
    SoftStrength,
    TypedValue,
    UnsupportedBody,
    WeightedSum,
    WeightedTerm,
    _validate_membership,
)
from execution_llm.tier2.expr import (
    Match,
    Node,
    canonical_bytes,
    has_holes,
    normalize,
    replace_where,
    unify,
    walk,
)
from execution_llm.tier2.surface import DomainSurface, Signature


@dataclass(frozen=True)
class Frame:
    tree: Node  # the frame's IR, parsed (may carry holes)
    hardness: str | None = None  # HARD | SOFT as marked; None: an objective, or a constraint at its default


@dataclass(frozen=True)
class FrameCover:
    frame: Frame
    normalized: Node
    matches: tuple[tuple[Signature, Match], ...]


@dataclass(frozen=True)
class GroundedTarget:
    """The decision `ground` derives, with the cover of each frame."""

    gold: DecisionBody
    covers: tuple[FrameCover, ...]


class _NoObjective(_Blocked):
    """A complete cover with no objective-landed frame (the runtime grounder asks for the objectives)."""


class _DuplicateLanding(_Blocked):
    """Two frames land on one constraint; ``mixed`` when every repeated constraint is stated with
    different strengths (hard and soft), which the runtime grounder reports as Unsupported."""

    def __init__(self, ref: str, *, mixed: bool) -> None:
        super().__init__(f"two frames land on the same constraint: {ref}")
        self.mixed = mixed


def cover(frame: Frame, surface: DomainSurface) -> FrameCover:
    """The signatures a frame unifies with, under four typing guards: a frame with a free variable
    covers nothing; a [HARD]/[SOFT] mark never lands on an objective; a population slot binds only
    the whole entity set; a value that misfits its slot is unstated, so the frame is covered again
    with HOLE(value) in that place (a missing_value Clarify, never a silent rescale)."""
    normalized = normalize(frame.tree, surface.vocab)
    if _free_variables(normalized):
        return FrameCover(frame, normalized, ())
    whole = canonical_bytes(normalize(surface.spec.population_slot_default, surface.vocab))
    matches = []
    for signature in surface.signatures:
        match = unify(surface.normalized[signature.ref], normalized, surface.vocab)
        if match is None or (frame.hardness is not None and signature.ref.startswith("objective:")):
            continue
        if any(
            signature.slots[k].kind == "population" and canonical_bytes(v) != whole for k, v in match.bindings.items()
        ):
            continue
        matches.append((signature, match))
    misfits = [value for signature, match in matches for value in _misfit_values(signature, match)]
    if misfits:
        holed = replace_where(
            normalized, lambda n: any(n is v for v in misfits), lambda _: {"op": "hole", "kind": "value"}
        )
        return cover(Frame(holed, frame.hardness), surface)
    return FrameCover(frame, normalized, tuple(matches))


def unit_misfit(frame: Frame, surface: DomainSurface) -> bool:
    """Whether `cover` replaced a stated value of this frame by HOLE(value) (its unit misfits the slot):
    only that path re-covers a new Frame, every other path keeps the one it was given."""
    return cover(frame, surface).frame is not frame


def _free_variables(node: Node) -> set[str]:
    bound: set[str] = set()
    used: set[str] = set()
    for n in walk(node):
        if isinstance(n.get("var"), str):
            bound.add(n["var"])
        bound.update(n.get("vars") or ())
        if n.get("op") == "var":
            used.add(n["ref"])
    return used - bound


def _misfit_values(signature: Signature, match: Match) -> list[Node]:
    """Scalar slot values whose base unit is not the slot's, or that are negative. A bare number on a
    COUNT-sorted day seat is a day count (the measure fixes its unit), the one bare value a unit slot takes."""
    out = []
    for spec_ref, value in match.bindings.items():
        spec = signature.slots[spec_ref]
        if spec.kind != "scalar" or value.get("op") != "value":
            continue
        unit = value.get("unit")
        want = signature.vocab.units[spec.unit][0] if spec.unit else None
        day_count = unit is None and spec.unit is not None and spec.value_sort == "COUNT"
        if (unit != want and not day_count) or Decimal(value["num"]) < 0:
            out.append(value)
    return out


def _slot_argument(spec_ref: str, value: Node, signature: Signature, inputs: SamplerInputs) -> SlotArgument | None:
    spec = signature.slots[spec_ref]
    policy = inputs.snapshot.slot_value_policies.get(spec.registry_ref)
    if spec.kind == "population":
        # a population slot bound to the whole entity set is omitted (its canonical spelling) when the
        # registry declares that its empty set means all members; any other population slot needs
        # explicit catalog members, which grounding cannot produce, so the problem is blocked
        if policy is not None and _default_kind(policy) == "all_members":
            return None
        raise _Blocked(
            f"population slot {spec.registry_ref} needs explicit catalog members"
            " (the registry does not declare that its empty set means all members)"
        )
    if spec.kind == "table":
        raise _Blocked(f"table slot {spec.registry_ref} is data, not a decision argument")
    if policy is None:
        raise _Blocked(f"no value policy for {spec.registry_ref}")
    num = value["num"]
    typed: TypedValue
    if policy.value_type == "integer":
        typed = IntegerValue(type="integer", value=int(Decimal(num)))
    elif policy.value_type == "duration":
        # a bare value reaches a duration seat only as a day count (cover's unit guard)
        assert spec.unit is not None  # a duration seat always declares its unit
        typed = DurationValue(type="duration", value=num, unit_ref=value["unit"] or signature.vocab.units[spec.unit][0])
    else:
        typed = DecimalValue(type="decimal", value=num)
    return SlotArgument(slot_ref=spec.registry_ref, value=typed)


def _strength(signature: Signature, frame: Frame, inputs: SamplerInputs) -> HardStrength | SoftStrength:
    entry = next((c for c in inputs.semantic.constraints if c.ref == signature.ref), None)
    if entry is None:
        raise _Blocked(f"the registry does not declare {signature.ref}")
    hardness = (frame.hardness or entry.default_hardness.upper()).lower()
    if hardness == "hard":
        return HardStrength(kind="hard")
    return SoftStrength(kind="soft", weight=entry.soft_weight_schema.default or "1")


def _execute_body(covers: tuple[FrameCover, ...], inputs: SamplerInputs) -> ExecuteBody:
    objectives: list[str] = []
    constraints: list[ConstraintSelection] = []
    for fc in covers:
        signature, match = fc.matches[0]
        if signature.ref.startswith("objective:"):
            objectives.append(signature.ref)
            continue
        arguments = []
        for spec_ref in sorted(match.bindings):
            argument = _slot_argument(spec_ref, match.bindings[spec_ref], signature, inputs)
            if argument is not None:
                arguments.append(argument)
        constraints.append(
            ConstraintSelection(
                constraint_ref=signature.ref,
                strength=_strength(signature, fc.frame, inputs),
                arguments=arguments,
                scope=None,
            )
        )
    refs = [c.constraint_ref for c in constraints]
    repeated = sorted({ref for ref in refs if refs.count(ref) > 1})
    if repeated:  # before the objective check: a hard+soft pair is Unsupported even with no objective
        strengths = [{c.strength.kind for c in constraints if c.constraint_ref == ref} for ref in repeated]
        raise _DuplicateLanding(repeated[0], mixed=all(len(kinds) > 1 for kinds in strengths))
    if not objectives:
        raise _NoObjective("a problem needs an objective frame")
    undeclared = [ref for ref in objectives if ref not in inputs.snapshot.objective_refs]
    if undeclared:  # as `_strength` does for a constraint
        raise _Blocked(f"the registry does not declare {undeclared[0]}")
    # the decision contract keeps selections sorted by their canonical encoding; frames come in utterance order
    constraints.sort(
        key=lambda c: json.dumps(c.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )
    weight = "1/1" if len(objectives) == 1 else f"1/{len(objectives)}"
    blend = WeightedSum(
        method="weighted_sum", terms=[WeightedTerm(objective_ref=r, weight=weight) for r in sorted(objectives)]
    )
    return _execute(blend, constraints)


def _validated(body: ExecuteBody, inputs: SamplerInputs) -> ExecuteBody:
    """The body in canonical form under the comparator, once it passes the grader's
    snapshot-membership and required-slot completeness checks."""

    _validate_membership(body, inputs.snapshot)
    _check_required_completeness(body, inputs.snapshot, inputs.signature)
    canonical = _canonicalize(body, inputs.snapshot, inputs.signature)
    assert isinstance(canonical, ExecuteBody)
    return canonical


def ground(frames: tuple[Frame, ...], inputs: SamplerInputs, surface: DomainSurface) -> GroundedTarget:
    """Cover every frame and derive the decision (see the module docstring). `_NoObjective` and
    `_DuplicateLanding` leave the reading to the caller; any other `_Blocked` means the frames cannot
    be grounded as given."""
    covers = tuple(cover(frame, surface) for frame in frames)
    holed = [fc for fc in covers if has_holes(fc.frame.tree)]
    uncovered = [fc for fc in covers if not fc.matches]

    def abstention(body: DecisionBody) -> GroundedTarget:
        return GroundedTarget(_finalize_abstention(body, inputs), covers)

    # Unsupported: an uncovered frame (classified by `_uncovered_gold`), or a covered frame whose
    # data primitive is deferred.
    deferred = [
        fc for fc in covers if fc.matches and fc.matches[0][0].deferred_primitive and not has_holes(fc.frame.tree)
    ]
    semantic_holes = [
        fc
        for fc in holed
        if any(
            n.get("op") in ("scope_hole", "frame_hole") or (n.get("op") == "hole" and n.get("kind") != "value")
            for n in walk(fc.frame.tree)
        )
    ]
    # Clarify takes precedence over support checks while meaning is unresolved.
    if holed and (
        len(holed) > 1
        or uncovered
        or deferred
        or any(match.missing for fc in semantic_holes for _, match in fc.matches)
    ):
        candidates = sorted({s.ref for fc in semantic_holes or holed for s, _ in fc.matches})
        slots = sorted({s.slots[k].registry_ref for fc in holed for s, m in fc.matches for k in m.missing})
        if not semantic_holes and slots and all(len(fc.matches) == 1 for fc in holed):
            return abstention(_clarify("missing_value", inputs_=[SlotInput(kind="slot", slot_ref=s) for s in slots]))
        return abstention(_clarify("ambiguous_capability", candidates=candidates))
    if uncovered and not holed:
        return abstention(_uncovered_gold(uncovered[0], inputs, surface))
    if deferred:
        ref = deferred[0].matches[0][0].ref
        return abstention(UnsupportedBody(kind="unsupported", reason_code="required_primitive_absent", related_refs=[ref]))
    if holed:
        # one holed frame, and it is covered (an uncovered one clarified above)
        fc = holed[0]
        missing = fc.matches[0][1].missing
        if missing:
            if len(fc.matches) != 1:
                raise _Blocked("a value hole must sit on exactly one signature")
            slot_ref = fc.matches[0][0].slots[missing[0]].registry_ref
            return abstention(_clarify("missing_value", inputs_=[SlotInput(kind="slot", slot_ref=slot_ref)]))
        return abstention(_clarify("ambiguous_capability", candidates=sorted(s.ref for s, _ in fc.matches)))
    return GroundedTarget(_validated(_execute_body(covers, inputs), inputs), covers)


def _uncovered_gold(fc: FrameCover, inputs: SamplerInputs, surface: DomainSurface) -> UnsupportedBody:
    """Classify an uncovered frame. A tree that is a declared recognized entry, or an unrealized
    configuration of a declared family (the domain's `Extra` table, matched by unification), is
    realization_absent with that ref, which the registry must declare. Anything else is outside the
    served vocabulary: semantic_not_declared with no related refs."""

    declared = set(inputs.snapshot.objective_refs) | set(inputs.snapshot.constraint_refs)
    for extra, tree in surface.extras_normalized:
        if extra.terminal == "realization_absent" and unify(tree, fc.normalized, surface.vocab) is not None:
            ref = extra.related_ref or ""
            if ref not in declared:
                raise _Blocked(
                    f"this meaning is recognized as {ref}, which the registry does not declare;"
                    " it can be reported as realization_absent only when the registry declares that ref"
                )
            return UnsupportedBody(kind="unsupported", reason_code="realization_absent", related_refs=[ref])
    return UnsupportedBody(kind="unsupported", reason_code="semantic_not_declared", related_refs=[])
