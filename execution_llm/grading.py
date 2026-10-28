"""Decision-level grading: comparator, outcome classification, and metrics.

The graded emission is a DecisionBody JSON; in this repository it is the grounder's decision for the
model's IR, not text the model writes.

- Comparison: O1 schema validation -> O2 binding existence and entry-relative validity,
  extended with a required-slot completeness check -> six meaning-preserving normalization
  rewrites -> JCS bytes -> O3 byte equality. No further equivalence is folded in.
- Outcome classification: the outcome partition, including a probe that classifies an
  Execute emission as a form defect when it differs from gold only by spelling a paired
  objective as a soft constraint (`GradingSignature.g0_constraint_to_objective`).
- Metrics: per-arm rates (an arm is a gold decision kind: execute, clarify, unsupported),
  which do not depend on the arm mix of the graded set, with a declared-weight remix
  beside them.

The six normalization rewrites, applied to both sides:

1. duration and quantity values rescale to their dimension's base unit
   (`GradingSignature.unit_rescale`), except at COUNT-sorted positions;
2. `weighted_sum` coefficients reduce to exact unit-sum rationals (`normalize_decision`);
3. datetime instants project to the canonical `+09:00` spelling (`normalize_decision`);
4. unbound declared slots and parameters with a `{kind:"value"}` default are filled in;
5. on a polarity-declared slot, an empty list rewrites to omission;
6. members of a polarity-declared (set-typed) list sort by their canonical bytes.

The module reuses `_validate_membership`, `normalize_decision`,
`validate_normalized_decision`, and `jcs_bytes` from `execution_llm.schema.decision`. It
reads exactly three things: the two decisions, the pinned snapshot, and the registry's
structured signature data (`GradingSignature`) -- never the utterance, the parse tree, or
capability prose.

The comparator is total and deterministic: no model calls, no solver I/O, no clock, no
randomness. An exception escaping `grade_turn` is a harness defect, never a grade.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from types import MappingProxyType
from typing import Any, Literal

from pydantic import ValidationError

from execution_llm.schema.decision import (
    _BODY_ADAPTER,
    ClarifyBody,
    DecisionBody,
    ExecuteBody,
    QuerySelection,
    SlotArgument,
    SnapshotConstraints,
    UnsupportedBody,
    ValuePolicy,
    WeightedSum,
    _strict_json,
    _validate_membership,
    _validate_safe_integers,
    _walk_json,
    jcs_bytes,
    normalize_decision,
)

# Stored raw outputs may be re-graded later only under a declared comparator version, so the
# version is part of the grader's identity.
COMPARATOR_VERSION = "1.0"

DefaultKind = Literal["value", "unbounded", "all_members", "required"]
Outcome = Literal[
    "success",
    "rejected",
    "mis_execution",
    "execute_form_defect",
    "unnecessary_clarify",
    "refusal",
    "abstention_payload_defect",
    "abstention_arm_confusion",
]

_KINDS = ("execute", "clarify", "unsupported")


@dataclass(frozen=True)
class GradingSignature:
    """Structured signature data the comparator reads beside the snapshot.

    Defaults come from the snapshot's value policies, which carry the supplied registry's slot
    declarations (see `_default_kind` and `_default_value`). What a policy cannot say is declared
    here: the sort facts (COUNT-sorted positions never rescale; TABLE-sorted slots are outside the
    completeness check), the unit rescale table, and the constraint/objective pairs read by the
    Execute form-defect probe.
    """

    # Unit ref -> (base unit ref, integer factor to base). Units absent from the table are
    # carried verbatim (COUNT kinds, undeclared units). Factors must be positive integers
    # (checked in __post_init__); a fractional factor such as percent would need a Fraction.
    unit_rescale: Mapping[str, tuple[str, int]] = field(default_factory=dict)
    # Positions (slot/parameter/field refs) whose declared sort is a COUNT kind. A value at
    # such a position never rescales; this separates `day` as a count from `day` as a duration.
    count_sorted_positions: frozenset[str] = frozenset()
    # TABLE-sorted slots: "required" there is a demand on the data channel, so these slots
    # are outside the required-slot completeness check.
    table_sorted_slots: frozenset[str] = frozenset()
    # Constraint/objective pairs that spell one preference two ways: constraint ref ->
    # canonical objective ref (in the routing registry, `constraint:distance` and
    # `constraint:duration`). In grading, only the Execute form-defect probe reads it, and
    # only for soft entries of the constraint.
    g0_constraint_to_objective: Mapping[str, str] = field(default_factory=dict)
    # Entry -> declared-parameter edges (the parameters a selected capability entry declares),
    # read by the parameter halves of the completeness check and of rewrite 4 (default
    # materialization). The registry behind the published gold declares no such edges, so this
    # defaults to empty and both parameter halves are no-ops until a registry declares them.
    entry_parameters: Mapping[str, frozenset[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "unit_rescale", MappingProxyType(dict(self.unit_rescale)))
        object.__setattr__(self, "count_sorted_positions", frozenset(self.count_sorted_positions))
        object.__setattr__(self, "table_sorted_slots", frozenset(self.table_sorted_slots))
        object.__setattr__(self, "g0_constraint_to_objective", MappingProxyType(dict(self.g0_constraint_to_objective)))
        object.__setattr__(
            self,
            "entry_parameters",
            MappingProxyType({ref: frozenset(refs) for ref, refs in self.entry_parameters.items()}),
        )
        for unit_ref, (base_ref, factor) in self.unit_rescale.items():
            if type(factor) is not int or factor < 1:
                raise ValueError(f"unit rescale factor must be a positive integer: {unit_ref}")
            if base_ref in self.unit_rescale and self.unit_rescale[base_ref] != (base_ref, 1):
                raise ValueError(f"base unit must map to itself with factor 1: {base_ref}")


@dataclass(frozen=True)
class TurnGrade:
    """One graded turn: its outcome class plus the labels the metrics condition on."""

    outcome: Outcome
    gold_kind: Literal["execute", "clarify", "unsupported"]
    emitted_kind: Literal["execute", "clarify", "unsupported"] | None
    reject_layer: Literal["o1", "o2"] | None = None
    detail: str = ""
    domain: str | None = None
    gold_reason_code: str | None = None


def grade_turn(
    emitted: str | bytes,
    gold: str | bytes | DecisionBody,
    snapshot: SnapshotConstraints,
    signature: GradingSignature,
    *,
    domain: str | None = None,
) -> TurnGrade:
    """Grade one emission against its gold decision under one pinned context."""

    gold_body = gold if isinstance(gold, ExecuteBody | ClarifyBody | UnsupportedBody) else _parse_o1(gold)
    _validate_membership(gold_body, snapshot)
    _check_required_completeness(gold_body, snapshot, signature)
    gold_kind = gold_body.kind
    gold_reason = getattr(gold_body, "reason_code", None)

    def _grade(outcome: Outcome, emitted_kind: str | None, layer: str | None = None, detail: str = "") -> TurnGrade:
        return TurnGrade(
            outcome=outcome,
            gold_kind=gold_kind,
            emitted_kind=emitted_kind,  # type: ignore[arg-type]
            reject_layer=layer,  # type: ignore[arg-type]
            detail=detail,
            domain=domain,
            gold_reason_code=gold_reason,
        )

    # O1 -- schema. Failure terminates with outcome `rejected`; the kind recovered from the raw
    # JSON is used only for classification (None when unreadable), never to repair the emission.
    try:
        emitted_body = _parse_o1(emitted)
    except (ValueError, ValidationError) as error:
        return _grade("rejected", _recover_kind(emitted), "o1", str(error))

    # O2 -- binding existence and entry-relative validity, plus the required-slot completeness
    # check that `_validate_membership` does not perform.
    try:
        _validate_membership(emitted_body, snapshot)
        _check_required_completeness(emitted_body, snapshot, signature)
    except ValueError as error:
        return _grade("rejected", emitted_body.kind, "o2", str(error))

    gold_bytes = jcs_bytes(_canonicalize(gold_body, snapshot, signature))
    emitted_bytes = jcs_bytes(_canonicalize(emitted_body, snapshot, signature))
    equal = emitted_bytes == gold_bytes
    emitted_kind = emitted_body.kind

    if isinstance(emitted_body, ExecuteBody):
        if gold_kind != "execute":
            return _grade("mis_execution", emitted_kind)
        if equal:
            return _grade("success", emitted_kind)
        probe_bytes = _g0_probe_bytes(emitted_body, snapshot, signature)
        if probe_bytes is not None and probe_bytes == gold_bytes:
            return _grade("execute_form_defect", emitted_kind)
        return _grade("mis_execution", emitted_kind)
    if gold_kind == "execute":
        return _grade("unnecessary_clarify" if emitted_kind == "clarify" else "refusal", emitted_kind)
    if emitted_kind != gold_kind:
        return _grade("abstention_arm_confusion", emitted_kind)
    return _grade("success" if equal else "abstention_payload_defect", emitted_kind)


def _parse_o1(content: str | bytes | DecisionBody) -> DecisionBody:
    """The O1 layer of `parse_decision_with_evidence`, split off so the o1/o2 report is exact."""

    if isinstance(content, ExecuteBody | ClarifyBody | UnsupportedBody):
        return content
    raw, _ = _strict_json(content)
    if not isinstance(raw, dict):
        raise ValueError("DecisionBody must be a JSON object")  # noqa: TRY004 - parse failure
    for value in _walk_json(raw):
        if isinstance(value, str) and any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            raise ValueError("JSON strings must contain Unicode scalar values")
    body = _BODY_ADAPTER.validate_python(raw, strict=True)
    _validate_safe_integers(body)
    return body


def _recover_kind(content: str | bytes) -> str | None:
    try:
        raw = json.loads(content)
    except (ValueError, TypeError):
        return None
    if isinstance(raw, dict) and raw.get("kind") in _KINDS:
        return raw["kind"]
    return None


# --- O2 extension: required-slot completeness -----------------------------------------------


def _check_required_completeness(
    body: DecisionBody, snapshot: SnapshotConstraints, signature: GradingSignature
) -> None:
    """Every required-kind slot of a selected entry must be supplied.

    Entry-relative by construction: the declared slot sets of selected constraints and of
    query selections, plus the `entry_parameters` edges of selected capability entries.
    TABLE-sorted slots are outside this check: a missing table is expressed in gold as
    Clarify(data_shortage), not as an emission defect this check can see.
    """

    if not isinstance(body, ExecuteBody):
        return
    declared_parameters = _declared_parameters(body, signature)
    bound_parameters = {parameter.parameter_ref for parameter in body.parameters}
    for parameter_ref in sorted(declared_parameters - bound_parameters):
        policy = snapshot.parameter_value_policies.get(parameter_ref)
        if policy is not None and _default_kind(policy) == "required":
            raise ValueError(f"required parameter {parameter_ref} is not supplied")
    for constraint in body.constraints:
        _require_complete(
            snapshot.constraint_argument_slots.get(constraint.constraint_ref, frozenset()),
            constraint.arguments,
            snapshot.slot_value_policies,
            signature,
            f"constraint {constraint.constraint_ref}",
        )
        if isinstance(constraint.scope, QuerySelection):
            _require_query_complete(constraint.scope, snapshot, signature)
    for selection in body.data_binding.selections:
        if isinstance(selection, QuerySelection):
            _require_query_complete(selection, snapshot, signature)
    for override in body.data_binding.overrides:
        if isinstance(override.selection, QuerySelection):
            _require_query_complete(override.selection, snapshot, signature)


def _declared_parameters(body: ExecuteBody, signature: GradingSignature) -> set[str]:
    selected = {constraint.constraint_ref for constraint in body.constraints}
    selected.update(term.objective_ref for term in body.objectives.terms)
    declared: set[str] = set()
    for ref in selected:
        declared.update(signature.entry_parameters.get(ref, frozenset()))
    return declared


def _require_query_complete(
    selection: QuerySelection, snapshot: SnapshotConstraints, signature: GradingSignature
) -> None:
    _require_complete(
        snapshot.query_argument_slots.get(selection.query_ref, frozenset()),
        selection.arguments,
        snapshot.slot_value_policies,
        signature,
        f"query {selection.query_ref}",
    )


def _require_complete(
    declared: frozenset[str],
    arguments: list[SlotArgument],
    policies: Mapping[str, ValuePolicy],
    signature: GradingSignature,
    position: str,
) -> None:
    bound = {argument.slot_ref for argument in arguments}
    for slot_ref in sorted(declared - bound):
        if slot_ref in signature.table_sorted_slots:
            continue
        policy = policies.get(slot_ref)
        if policy is None:
            continue
        if _default_kind(policy) == "required":
            raise ValueError(f"required slot {slot_ref} is not supplied for {position}")


def _default_kind(policy: ValuePolicy) -> DefaultKind:
    """What leaving a position unbound means, as the registry declares it: an empty-set polarity of
    "all" -> "all_members"; a default -> "value"; not optional -> "required"; else "unbounded"."""
    if policy.empty_set_semantics == "all":
        return "all_members"
    if policy.default is not None:
        return "value"
    if not policy.optional:
        return "required"
    return "unbounded"


# --- Normalization: six meaning-preserving rewrites, applied to both sides ----------------


def _canonicalize(body: DecisionBody, snapshot: SnapshotConstraints, signature: GradingSignature) -> DecisionBody:
    """Apply rewrites 1 and 4-6, then `normalize_decision` (rewrites 2-3) and its form checks."""

    if not isinstance(body, ExecuteBody):
        return normalize_decision(body)
    data = body.model_dump(mode="json")
    for entry in data["constraints"]:
        entry["arguments"] = _rewrite_arguments(
            entry["arguments"],
            snapshot.constraint_argument_slots.get(entry["constraint_ref"], frozenset()),
            snapshot.slot_value_policies,
            signature,
        )
        if entry["scope"] is not None and entry["scope"]["kind"] == "query":
            _rewrite_query(entry["scope"], snapshot, signature)
    # Rewrite 4's parameter half: materialize {kind:"value"} defaults of parameters declared
    # by a selected capability entry, then rescale everything through the same path.
    bound_parameters = {parameter["parameter_ref"] for parameter in data["parameters"]}
    for parameter_ref in sorted(_declared_parameters(body, signature) - bound_parameters):
        policy = snapshot.parameter_value_policies.get(parameter_ref)
        if policy is not None and _default_kind(policy) == "value":
            data["parameters"].append({"parameter_ref": parameter_ref, "value": _default_value(parameter_ref, policy)})
    data["parameters"].sort(key=lambda parameter: parameter["parameter_ref"])
    for parameter in data["parameters"]:
        parameter["value"] = _rewrite_value(
            parameter["value"],
            parameter["parameter_ref"],
            snapshot.parameter_value_policies.get(parameter["parameter_ref"]),
            signature,
        )
    for selection in data["data_binding"]["selections"]:
        if selection["kind"] == "query":
            _rewrite_query(selection, snapshot, signature)
    for override in data["data_binding"]["overrides"]:
        override["value"] = _rewrite_value(
            override["value"],
            override["field_ref"],
            snapshot.field_value_policies.get(override["field_ref"]),
            signature,
        )
        if override["selection"]["kind"] == "query":
            _rewrite_query(override["selection"], snapshot, signature)
    # Rewrites change bytes, and these two collections sort by their full canonical encoding.
    # Members two rewrites just made byte-identical fold to one: the collections are sets
    # (sorted, duplicate-free by construction), so a duplicate carries no meaning, and without
    # the fold the re-validation would raise -- breaking the comparator's totality guarantee
    # over schema-legal emissions (an exception escaping `grade_turn` is a harness defect).
    data["constraints"] = _sorted_folded(data["constraints"])
    data["data_binding"]["overrides"] = _sorted_folded(data["data_binding"]["overrides"])
    return normalize_decision(_BODY_ADAPTER.validate_python(data, strict=True))


def _sorted_folded(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    folded: list[dict[str, Any]] = []
    previous: str | None = None
    for value in sorted(values, key=_sort_key):
        encoded = _sort_key(value)
        if encoded != previous:
            folded.append(value)
            previous = encoded
    return folded


def _rewrite_query(selection: dict[str, Any], snapshot: SnapshotConstraints, signature: GradingSignature) -> None:
    selection["arguments"] = _rewrite_arguments(
        selection["arguments"],
        snapshot.query_argument_slots.get(selection["query_ref"], frozenset()),
        snapshot.slot_value_policies,
        signature,
    )


def _rewrite_arguments(
    arguments: list[dict[str, Any]],
    declared: frozenset[str],
    policies: Mapping[str, ValuePolicy],
    signature: GradingSignature,
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for argument in arguments:
        policy = policies.get(argument["slot_ref"])
        # Rewrite 5: on a polarity-declared slot the empty-list spelling rewrites to
        # omission, the canonical form.
        if (
            policy is not None
            and policy.empty_set_semantics is not None
            and argument["value"] == {"type": "list", "values": []}
        ):
            continue
        kept.append(argument)
    bound = {argument["slot_ref"] for argument in kept}
    # Rewrite 4: materialize {kind:"value"} defaults for unbound declared slots.
    # Polarity-declared slots are exempt -- their canonical empty spelling is omission.
    for slot_ref in sorted(declared - bound):
        policy = policies.get(slot_ref)
        if policy is None or policy.empty_set_semantics is not None:
            continue
        if _default_kind(policy) == "value":
            kept.append({"slot_ref": slot_ref, "value": _default_value(slot_ref, policy)})
    for argument in kept:
        argument["value"] = _rewrite_value(
            argument["value"], argument["slot_ref"], policies.get(argument["slot_ref"]), signature
        )
    kept.sort(key=lambda argument: argument["slot_ref"])
    return kept


def _default_value(slot_ref: str, policy: ValuePolicy) -> dict[str, Any]:
    """The registry's declared default as TypedValue JSON; a duration or quantity default is in the
    position's one declared unit."""
    if policy.default is None:
        raise ValueError(f"no declared value default for {slot_ref}")
    if policy.value_type == "integer":
        return {"type": "integer", "value": int(policy.default)}
    if policy.value_type == "decimal":
        return {"type": "decimal", "value": policy.default}
    if policy.value_type in ("duration", "quantity") and len(policy.unit_refs) == 1:
        (unit_ref,) = policy.unit_refs
        return {"type": policy.value_type, "value": policy.default, "unit_ref": unit_ref}
    raise ValueError(f"the value default of {slot_ref} needs exactly one declared unit")


def _rewrite_value(
    value: dict[str, Any],
    position_ref: str,
    policy: ValuePolicy | None,
    signature: GradingSignature,
) -> dict[str, Any]:
    if value["type"] == "list":
        values = [_rescale_scalar(member, position_ref, signature) for member in value["values"]]
        # Rewrite 6: set-member order is not meaning on polarity-declared (set-typed)
        # slots; members sort by their canonical bytes. Order-semantic lists keep their order.
        if policy is not None and policy.empty_set_semantics is not None:
            values.sort(key=_sort_key)
        return {"type": "list", "values": values}
    if value["type"] == "range":
        rewritten = dict(value)
        for bound in ("lower", "upper"):
            if rewritten[bound] is not None:
                rewritten[bound] = _rescale_scalar(rewritten[bound], position_ref, signature)
        return rewritten
    return _rescale_scalar(value, position_ref, signature)


def _rescale_scalar(value: dict[str, Any], position_ref: str, signature: GradingSignature) -> dict[str, Any]:
    """Rewrite 1: rescale to the dimension's base unit; COUNT-sorted positions never rescale."""

    if value["type"] not in ("duration", "quantity") or position_ref in signature.count_sorted_positions:
        return value
    target = signature.unit_rescale.get(value["unit_ref"])
    if target is None:
        return value
    base_ref, factor = target
    return {**value, "value": _rescale_decimal(value["value"], factor), "unit_ref": base_ref}


def _rescale_decimal(text: str, factor: int) -> str:
    # Exact integer arithmetic on the canonical DecimalString spelling: no `decimal` context
    # is read, so grading cannot round, trap, or drift under an ambient context change.
    sign = "-" if text.startswith("-") else ""
    integer, _, fraction = text.lstrip("-").partition(".")
    scaled = int(integer + fraction) * factor
    if scaled == 0:
        return "0"
    quotient, remainder = divmod(scaled, 10 ** len(fraction)) if fraction else (scaled, 0)
    fraction_text = str(remainder).rjust(len(fraction), "0").rstrip("0") if remainder else ""
    return f"{sign}{quotient}.{fraction_text}" if fraction_text else f"{sign}{quotient}"


def _sort_key(value: Any) -> str:
    # Matches the `_require_sorted_unique` encoding in `execution_llm.schema.decision`, so
    # re-sorted collections satisfy the model validators byte-for-byte.
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


# --- Execute form-defect classification probe ---------------------------------------------


def _g0_probe_bytes(body: ExecuteBody, snapshot: SnapshotConstraints, signature: GradingSignature) -> bytes | None:
    """Move soft constraints named in `g0_constraint_to_objective` onto their paired objectives.

    Returns the canonical bytes of the rewritten body, or None when the probe does not apply
    (no soft paired constraint, or objectives that are not a weighted sum) or cannot legally
    rewrite the body. Mechanical and classification-only: byte equality with gold after the
    probe -- and only that -- classifies the failure as `execute_form_defect` rather than
    `mis_execution` (it is still a failure). Any residual difference leaves the turn
    classified as `mis_execution`.
    """

    if not isinstance(body.objectives, WeightedSum) or not signature.g0_constraint_to_objective:
        return None
    data = body.model_dump(mode="json")
    kept: list[dict[str, Any]] = []
    moved = False
    for entry in data["constraints"]:
        objective_ref = signature.g0_constraint_to_objective.get(entry["constraint_ref"])
        if objective_ref is None or entry["strength"]["kind"] != "soft":
            kept.append(entry)
            continue
        weight = Fraction(entry["strength"]["weight"])
        if weight <= 0:
            return None
        data["objectives"]["terms"].append(
            {"objective_ref": objective_ref, "weight": f"{weight.numerator}/{weight.denominator}"}
        )
        moved = True
    if not moved:
        return None
    data["constraints"] = kept
    data["objectives"]["terms"].sort(key=lambda term: term["objective_ref"])
    try:
        probe = _BODY_ADAPTER.validate_python(data, strict=True)
        return jcs_bytes(_canonicalize(probe, snapshot, signature))
    except (ValueError, ValidationError):
        # A duplicate objective ref or an over-limit term count is not a re-spelling of the pair.
        return None


# --- Metrics: exact rates; arm-conditional rates do not depend on the arm mix -------------


@dataclass(frozen=True)
class RateBlock:
    """The rate set over one set of graded turns. All rates are exact Fractions; None means 0/0."""

    n: int
    success_count: int
    headline_success: Fraction | None
    selective_accuracy: Fraction | None
    unreadable_reject_count: int
    bindable_arm_coverage: Fraction | None
    emission_rate_over_n: Fraction | None
    mis_execution_by_gold_arm: Mapping[str, Fraction | None]
    mis_execution_over_n: Fraction | None
    rejection_rate: Fraction | None
    rejection_o1_rate: Fraction | None
    rejection_o2_rate: Fraction | None
    unnecessary_clarify_rate: Fraction | None
    refusal_rate: Fraction | None
    execute_form_defect_rate: Fraction | None
    justified_abstention_rate: Fraction | None
    justified_abstention_by_gold_arm: Mapping[str, Fraction | None]
    abstention_failure_slices: Mapping[str, Mapping[str, Fraction | None]]
    wrong_emit_on_should_abstain: Fraction | None
    clarify_precision: Fraction | None
    clarify_recall: Fraction | None
    gold_arm_shares: Mapping[str, Fraction | None]


@dataclass(frozen=True)
class MetricsReport:
    overall: RateBlock
    by_gold_arm: Mapping[str, RateBlock]
    by_domain: Mapping[str, RateBlock]
    by_gold_reason: Mapping[str, RateBlock]


def compute_metrics(turns: list[TurnGrade]) -> MetricsReport:
    """Report the rate set overall, per gold arm, per domain, and per gold reason code."""

    arms = sorted({turn.gold_kind for turn in turns})
    domains = sorted({turn.domain for turn in turns if turn.domain is not None})
    reasons = sorted({turn.gold_reason_code for turn in turns if turn.gold_reason_code is not None})
    return MetricsReport(
        overall=_rate_block(turns),
        by_gold_arm={key: _rate_block([turn for turn in turns if turn.gold_kind == key]) for key in arms},
        by_domain={key: _rate_block([turn for turn in turns if turn.domain == key]) for key in domains},
        by_gold_reason={key: _rate_block([turn for turn in turns if turn.gold_reason_code == key]) for key in reasons},
    )


def remix_mis_execution(block: RateBlock, arm_weights: Mapping[str, Fraction]) -> Fraction:
    """Remix the three arm-conditional mis-execution rates under declared arm weights.

    The weights must be non-negative and sum to exactly one. With the graded set's own
    `gold_arm_shares` the remix equals `mis_execution_over_n`; with production arm weights it
    estimates the deployed rate. Choosing production weights is outside this module.
    """

    if any(weight < 0 for weight in arm_weights.values()) or sum(arm_weights.values()) != 1:
        raise ValueError("remix weights must be non-negative and sum exactly to one")
    total = Fraction(0)
    for arm, weight in arm_weights.items():
        if weight == 0:
            continue
        rate = block.mis_execution_by_gold_arm.get(arm)
        if rate is None:
            raise ValueError(f"no {arm}-arm turns to carry a nonzero remix weight")
        total += weight * rate
    return total


def _rate_block(turns: list[TurnGrade]) -> RateBlock:
    def ratio(numerator: int, denominator: int) -> Fraction | None:
        return None if denominator == 0 else Fraction(numerator, denominator)

    n = len(turns)
    successes = [turn for turn in turns if turn.outcome == "success"]
    gold = {kind: [turn for turn in turns if turn.gold_kind == kind] for kind in _KINDS}
    gold_execute = gold["execute"]
    gold_abstention = gold["clarify"] + gold["unsupported"]
    # Emission-side denominators keep rejected emissions whose kind is readable; only rejects
    # with an unreadable kind (emitted_kind None) fall out, and their count is reported as
    # `unreadable_reject_count`.
    emitted_execute = [turn for turn in turns if turn.emitted_kind == "execute"]
    emitted_clarify = [turn for turn in turns if turn.emitted_kind == "clarify"]
    unreadable = sum(1 for turn in turns if turn.emitted_kind is None)
    rejected = [turn for turn in turns if turn.outcome == "rejected"]
    mis = [turn for turn in turns if turn.outcome == "mis_execution"]
    mis_by_arm = {kind: ratio(sum(1 for turn in mis if turn.gold_kind == kind), len(gold[kind])) for kind in _KINDS}
    coverage_numerator = sum(
        1 for turn in gold_execute if turn.emitted_kind == "execute" and turn.outcome != "rejected"
    )
    slices = {}
    for kind in ("clarify", "unsupported"):
        arm = gold[kind]
        slices[kind] = {
            "payload_defect": ratio(sum(1 for t in arm if t.outcome == "abstention_payload_defect"), len(arm)),
            "arm_confusion": ratio(sum(1 for t in arm if t.outcome == "abstention_arm_confusion"), len(arm)),
            "mis_execution": ratio(sum(1 for t in arm if t.outcome == "mis_execution"), len(arm)),
            "rejected": ratio(sum(1 for t in arm if t.outcome == "rejected"), len(arm)),
        }
    clarify_successes = sum(1 for turn in successes if turn.gold_kind == "clarify")
    return RateBlock(
        n=n,
        success_count=len(successes),
        headline_success=ratio(len(successes), n),
        selective_accuracy=ratio(sum(1 for turn in emitted_execute if turn.outcome == "success"), len(emitted_execute)),
        unreadable_reject_count=unreadable,
        bindable_arm_coverage=ratio(coverage_numerator, len(gold_execute)),
        emission_rate_over_n=ratio(len(emitted_execute), n),
        mis_execution_by_gold_arm=MappingProxyType(mis_by_arm),
        mis_execution_over_n=ratio(len(mis), n),
        rejection_rate=ratio(len(rejected), n),
        rejection_o1_rate=ratio(sum(1 for turn in rejected if turn.reject_layer == "o1"), n),
        rejection_o2_rate=ratio(sum(1 for turn in rejected if turn.reject_layer == "o2"), n),
        unnecessary_clarify_rate=ratio(
            sum(1 for turn in gold_execute if turn.outcome == "unnecessary_clarify"), len(gold_execute)
        ),
        refusal_rate=ratio(sum(1 for turn in gold_execute if turn.outcome == "refusal"), len(gold_execute)),
        execute_form_defect_rate=ratio(
            sum(1 for turn in gold_execute if turn.outcome == "execute_form_defect"), len(gold_execute)
        ),
        justified_abstention_rate=ratio(
            sum(1 for turn in successes if turn.gold_kind != "execute"), len(gold_abstention)
        ),
        justified_abstention_by_gold_arm=MappingProxyType(
            {
                kind: ratio(sum(1 for turn in successes if turn.gold_kind == kind), len(gold[kind]))
                for kind in ("clarify", "unsupported")
            }
        ),
        abstention_failure_slices=MappingProxyType({kind: MappingProxyType(v) for kind, v in slices.items()}),
        wrong_emit_on_should_abstain=ratio(sum(1 for turn in mis if turn.gold_kind != "execute"), len(gold_abstention)),
        clarify_precision=ratio(clarify_successes, len(emitted_clarify)),
        clarify_recall=ratio(clarify_successes, len(gold["clarify"])),
        gold_arm_shares=MappingProxyType({kind: ratio(len(gold[kind]), n) for kind in _KINDS}),
    )


__all__ = [  # noqa: RUF022 - public contract order
    "COMPARATOR_VERSION",
    "GradingSignature",
    "TurnGrade",
    "grade_turn",
    "RateBlock",
    "MetricsReport",
    "compute_metrics",
    "remix_mis_execution",
]
