"""DecisionBody, the decision the grounder writes (decision schema version 3.0).

Strict JSON parsing, membership checks against a SnapshotConstraints, normalization and
canonical JSON serialization.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from fractions import Fraction
from types import MappingProxyType
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    WithJsonSchema,
    field_validator,
    model_validator,
)

SAFE_INT = 9_007_199_254_740_991
_DECIMAL = re.compile(r"^(?:0|-?[1-9][0-9]*|(?:-?(?:0|[1-9][0-9]*))\.[0-9]*[1-9])$")
_RATIONAL = re.compile(r"^(?:[1-9][0-9]*)/(?:[1-9][0-9]*)$")
_DATETIME = re.compile(
    r"^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}):([0-5]\d)"
    r"(Z|\+(?:[01]\d|2[0-3]):[0-5]\d|"
    r"-(?:00:(?:0[1-9]|[1-5]\d)|(?:0[1-9]|1\d|2[0-3]):[0-5]\d))$"
)

Ref = Annotated[str, Field(strict=True, min_length=1)]
DecimalString = Annotated[str, Field(strict=True, pattern=_DECIMAL.pattern)]
RationalWeightString = Annotated[str, Field(strict=True, pattern=_RATIONAL.pattern)]
CanonicalInt = Annotated[
    int,
    Field(strict=True),
    WithJsonSchema({"type": "integer", "minimum": -SAFE_INT, "maximum": SAFE_INT}),
]


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


def _plain(value: Any) -> Any:
    return value.model_dump(mode="json") if isinstance(value, BaseModel) else value


def _require_sorted_unique(values: Sequence[Any], *, key: str | None = None) -> None:
    encoded = [
        (
            str(getattr(value, key))
            if key
            else json.dumps(_plain(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
        for value in values
    ]
    if encoded != sorted(encoded) or len(encoded) != len(set(encoded)):
        raise ValueError("collection must already be sorted and duplicate-free")


class BooleanValue(_ClosedModel):
    type: Literal["boolean"]
    value: bool


class IntegerValue(_ClosedModel):
    type: Literal["integer"]
    value: CanonicalInt


class DecimalValue(_ClosedModel):
    type: Literal["decimal"]
    value: DecimalString


class TextValue(_ClosedModel):
    type: Literal["text"]
    value: str


class EnumValue(_ClosedModel):
    type: Literal["enum"]
    value_ref: Ref


class DateTimeValue(_ClosedModel):
    type: Literal["datetime"]
    value: Annotated[str, Field(pattern=_DATETIME.pattern)]

    @field_validator("value")
    @classmethod
    def canonical_datetime(cls, value: str) -> str:
        _project_datetime(value)
        return value


def _project_datetime(value: str) -> str:
    match = _DATETIME.fullmatch(value)
    if not match or match.group(3) == "60" or match.group(4) == "-00:00":
        raise ValueError("datetime must be a known-offset whole-second instant")
    try:
        instant = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("invalid datetime") from exc
    return instant.astimezone(timezone(timedelta(hours=9))).isoformat(timespec="seconds")


class DurationValue(_ClosedModel):
    type: Literal["duration"]
    value: DecimalString
    unit_ref: Ref


class QuantityValue(_ClosedModel):
    type: Literal["quantity"]
    value: DecimalString
    unit_ref: Ref


class RefValue(_ClosedModel):
    type: Literal["ref"]
    value_ref: Ref


ScalarValue: TypeAlias = Annotated[
    BooleanValue
    | IntegerValue
    | DecimalValue
    | TextValue
    | EnumValue
    | DateTimeValue
    | DurationValue
    | QuantityValue
    | RefValue,
    Field(discriminator="type"),
]


class ListValue(_ClosedModel):
    type: Literal["list"]
    values: list[ScalarValue]


class RangeValue(_ClosedModel):
    type: Literal["range"]
    lower: ScalarValue | None
    upper: ScalarValue | None
    lower_inclusive: bool
    upper_inclusive: bool


TypedValue: TypeAlias = Annotated[ScalarValue | ListValue | RangeValue, Field(discriminator="type")]


class SlotArgument(_ClosedModel):
    slot_ref: Ref
    value: TypedValue


class AllSelection(_ClosedModel):
    kind: Literal["all"]
    entity_ref: Ref


class IdsSelection(_ClosedModel):
    kind: Literal["ids"]
    entity_ref: Ref
    ids: Annotated[list[Ref], Field(min_length=1)]

    @field_validator("ids")
    @classmethod
    def sorted_ids(cls, values: list[str]) -> list[str]:
        _require_sorted_unique(values)
        return values


class QuerySelection(_ClosedModel):
    kind: Literal["query"]
    entity_ref: Ref
    query_ref: Ref
    arguments: list[SlotArgument]

    @field_validator("arguments")
    @classmethod
    def sorted_arguments(cls, values: list[SlotArgument]) -> list[SlotArgument]:
        _require_sorted_unique(values, key="slot_ref")
        return values


ScopeSelection: TypeAlias = Annotated[AllSelection | IdsSelection | QuerySelection, Field(discriminator="kind")]
DataSelection: TypeAlias = Annotated[IdsSelection | QuerySelection, Field(discriminator="kind")]


class HardStrength(_ClosedModel):
    kind: Literal["hard"]


class SoftStrength(_ClosedModel):
    kind: Literal["soft"]
    weight: DecimalString

    @field_validator("weight")
    @classmethod
    def non_negative_weight(cls, value: str) -> str:
        # A soft weight is non-negative regardless of what a manifest declares.
        # DecimalString itself admits signed spellings, and a null declared minimum records
        # only that the source declares no floor; it never permits a weight below zero.
        # The check runs at parse time, before normalization.
        if value.startswith("-"):
            raise ValueError("soft constraint weight must be a non-negative decimal")
        return value


ConstraintStrength: TypeAlias = Annotated[HardStrength | SoftStrength, Field(discriminator="kind")]


class ConstraintSelection(_ClosedModel):
    constraint_ref: Ref
    strength: ConstraintStrength
    arguments: list[SlotArgument]
    scope: ScopeSelection | None

    @field_validator("arguments")
    @classmethod
    def sorted_arguments(cls, values: list[SlotArgument]) -> list[SlotArgument]:
        _require_sorted_unique(values, key="slot_ref")
        return values


class ParameterSelection(_ClosedModel):
    parameter_ref: Ref
    value: TypedValue


class FieldOverride(_ClosedModel):
    field_ref: Ref
    selection: ScopeSelection
    value: TypedValue


class DataBinding(_ClosedModel):
    selections: list[DataSelection]
    overrides: list[FieldOverride]

    @field_validator("selections")
    @classmethod
    def sorted_selections(cls, values: list[DataSelection]) -> list[DataSelection]:
        _require_sorted_unique(values, key="entity_ref")
        return values

    @field_validator("overrides")
    @classmethod
    def sorted_overrides(cls, values: list[FieldOverride]) -> list[FieldOverride]:
        _require_sorted_unique(values)
        return values


class WeightedTerm(_ClosedModel):
    objective_ref: Ref
    weight: RationalWeightString


class ObjectiveTerm(_ClosedModel):
    objective_ref: Ref


class WeightedSum(_ClosedModel):
    method: Literal["weighted_sum"]
    terms: Annotated[list[WeightedTerm], Field(min_length=1)]

    @field_validator("terms")
    @classmethod
    def sorted_terms(cls, values: list[WeightedTerm]) -> list[WeightedTerm]:
        _require_sorted_unique(values, key="objective_ref")
        return values


class Lexicographic(_ClosedModel):
    method: Literal["lexicographic"]
    terms: Annotated[list[ObjectiveTerm], Field(min_length=1)]

    @field_validator("terms")
    @classmethod
    def unique_terms(cls, values: list[ObjectiveTerm]) -> list[ObjectiveTerm]:
        refs = [term.objective_ref for term in values]
        if len(refs) != len(set(refs)):
            raise ValueError("objective refs must be unique")
        return values


class Pareto(_ClosedModel):
    method: Literal["pareto"]
    terms: Annotated[list[ObjectiveTerm], Field(min_length=1)]

    @field_validator("terms")
    @classmethod
    def sorted_terms(cls, values: list[ObjectiveTerm]) -> list[ObjectiveTerm]:
        _require_sorted_unique(values, key="objective_ref")
        return values


ObjectiveComposition: TypeAlias = Annotated[WeightedSum | Lexicographic | Pareto, Field(discriminator="method")]


class DecisionFieldInput(_ClosedModel):
    kind: Literal["decision_field"]
    field: Literal["objectives", "objective_composition"]


class SlotInput(_ClosedModel):
    kind: Literal["slot"]
    slot_ref: Ref


class ParameterInput(_ClosedModel):
    kind: Literal["parameter"]
    parameter_ref: Ref


RequiredInput: TypeAlias = Annotated[DecisionFieldInput | SlotInput | ParameterInput, Field(discriminator="kind")]


class ExecuteBody(_ClosedModel):
    kind: Literal["execute"]
    objectives: ObjectiveComposition
    constraints: list[ConstraintSelection]
    parameters: list[ParameterSelection]
    data_binding: DataBinding

    @field_validator("constraints")
    @classmethod
    def sorted_constraints(cls, values: list[ConstraintSelection]) -> list[ConstraintSelection]:
        _require_sorted_unique(values)
        return values

    @field_validator("parameters")
    @classmethod
    def sorted_parameters(cls, values: list[ParameterSelection]) -> list[ParameterSelection]:
        _require_sorted_unique(values, key="parameter_ref")
        return values


class ClarifyBody(_ClosedModel):
    kind: Literal["clarify"]
    reason_code: Literal[
        "ambiguous_capability",
        "missing_value",
        "ambiguous_scope",
        "missing_composition",
        "unit_ambiguity",
        "data_shortage",
        "composition_choice_required",
        "hardness_ambiguity",
        "form_choice_required",
    ]
    candidate_refs: list[Ref]
    required_inputs: list[RequiredInput]
    missing_field_refs: list[Ref]

    @field_validator("candidate_refs", "required_inputs", "missing_field_refs")
    @classmethod
    def sorted_diagnostics(cls, values: list[Any]) -> list[Any]:
        _require_sorted_unique(values)
        return values

    @model_validator(mode="after")
    def exact_reason_payload(self) -> ClarifyBody:
        populated = {
            "candidate_refs": bool(self.candidate_refs),
            "required_inputs": bool(self.required_inputs),
            "missing_field_refs": bool(self.missing_field_refs),
        }
        allowed = {
            "ambiguous_capability": {"candidate_refs"},
            "ambiguous_scope": {"candidate_refs"},
            "missing_value": {"required_inputs"},
            "missing_composition": {"required_inputs"},
            "unit_ambiguity": {"candidate_refs", "required_inputs"},
            "data_shortage": {"missing_field_refs"},
            "composition_choice_required": {"candidate_refs"},
            "hardness_ambiguity": {"candidate_refs"},
            "form_choice_required": {"candidate_refs", "required_inputs"},
        }[self.reason_code]
        if any(value and name not in allowed for name, value in populated.items()):
            raise ValueError("diagnostic collection is forbidden for reason_code")
        required_nonempty = {
            "missing_value": {"required_inputs"},
            "missing_composition": {"required_inputs"},
            "unit_ambiguity": {"candidate_refs", "required_inputs"},
            "data_shortage": {"missing_field_refs"},
            "composition_choice_required": {"candidate_refs"},
            "hardness_ambiguity": {"candidate_refs"},
            "form_choice_required": {"candidate_refs", "required_inputs"},
        }.get(self.reason_code, set())
        if any(not populated[name] for name in required_nonempty):
            raise ValueError("reason_code is missing required diagnostic evidence")
        if self.reason_code == "missing_composition" and self.required_inputs != [
            DecisionFieldInput(kind="decision_field", field="objective_composition")
        ]:
            raise ValueError("missing_composition has one exact required input")
        return self

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: Any, handler: Any) -> dict[str, Any]:
        schema = handler(core_schema)
        shapes = {
            "ambiguous_capability": (None, 0, 0),
            "ambiguous_scope": (None, 0, 0),
            "missing_value": (0, 1, 0),
            "missing_composition": (0, 1, 0),
            "unit_ambiguity": (1, 1, 0),
            "data_shortage": (0, 0, 1),
            "composition_choice_required": (1, 0, 0),
            "hardness_ambiguity": (1, 0, 0),
            "form_choice_required": (1, 1, 0),
        }
        schema["allOf"] = []
        for reason, minimums in shapes.items():
            properties = {
                name: ({"maxItems": 0} if minimum == 0 else {"minItems": minimum})
                for name, minimum in zip(
                    ("candidate_refs", "required_inputs", "missing_field_refs"),
                    minimums,
                    strict=True,
                )
                if minimum is not None
            }
            if reason == "missing_composition":
                properties["required_inputs"] = {
                    "items": {"const": {"kind": "decision_field", "field": "objective_composition"}},
                    "minItems": 1,
                    "maxItems": 1,
                }
            schema["allOf"].append(
                {
                    "if": {"properties": {"reason_code": {"const": reason}}},
                    "then": {"properties": properties},
                }
            )
        return schema


class UnsupportedBody(_ClosedModel):
    kind: Literal["unsupported"]
    reason_code: Literal[
        "semantic_not_declared",
        "request_out_of_scope",
        "required_primitive_absent",
        "unsupported_composition",
        "realization_absent",
    ]
    related_refs: list[Ref]

    @field_validator("related_refs")
    @classmethod
    def sorted_refs(cls, values: list[str]) -> list[str]:
        _require_sorted_unique(values)
        return values


DecisionBody: TypeAlias = Annotated[ExecuteBody | ClarifyBody | UnsupportedBody, Field(discriminator="kind")]
_BODY_ADAPTER = TypeAdapter(DecisionBody)


ScalarType = Literal["boolean", "integer", "decimal", "text", "enum", "datetime", "duration", "quantity", "ref"]


def is_canonical_decimal(value: str) -> bool:
    """True when ``value`` is a canonical ``DecimalString`` spelling (one shared pattern)."""
    return _DECIMAL.fullmatch(value) is not None


@dataclass(frozen=True, slots=True)
class ValuePolicy:
    """Closed value facts for one manifest-owned value position."""

    value_type: ScalarType
    container: Literal["scalar", "list", "range"] = "scalar"
    enum_refs: frozenset[str] = frozenset()
    unit_refs: frozenset[str] = frozenset()
    ref_target_entity_refs: frozenset[str] = frozenset()
    # Facts the pinned source declares for this position: whether it is optional, its numeric
    # default and bounds, and, for list positions, what an empty list means (``all`` members or
    # ``none``). ``None`` means the source declares no such fact, and none is invented; it never
    # means "unbounded by default".
    optional: bool = False
    default: str | None = None
    minimum: str | None = None
    maximum: str | None = None
    empty_set_semantics: Literal["all", "none"] | None = None

    def __post_init__(self) -> None:
        if self.value_type not in {
            "boolean",
            "integer",
            "decimal",
            "text",
            "enum",
            "datetime",
            "duration",
            "quantity",
            "ref",
        } or self.container not in {"scalar", "list", "range"}:
            raise ValueError("unknown value policy type or container")
        object.__setattr__(self, "enum_refs", frozenset(self.enum_refs))
        object.__setattr__(self, "unit_refs", frozenset(self.unit_refs))
        object.__setattr__(self, "ref_target_entity_refs", frozenset(self.ref_target_entity_refs))
        required = {
            "enum": (self.enum_refs, self.unit_refs, self.ref_target_entity_refs),
            "duration": (self.unit_refs, self.enum_refs, self.ref_target_entity_refs),
            "quantity": (self.unit_refs, self.enum_refs, self.ref_target_entity_refs),
            "ref": (self.ref_target_entity_refs, self.enum_refs, self.unit_refs),
        }.get(self.value_type)
        if required is None:
            if self.enum_refs or self.unit_refs or self.ref_target_entity_refs:
                raise ValueError("scalar policy has irrelevant enum, unit, or ref targets")
        elif not required[0] or required[1] or required[2]:
            raise ValueError("scalar policy is missing or mixing its position-specific refs")
        if self.empty_set_semantics is not None and self.container != "list":
            raise ValueError("empty-set polarity applies only to list-valued positions")
        numeric = self.value_type in {"integer", "decimal", "duration", "quantity"}
        for bound in (self.default, self.minimum, self.maximum):
            if bound is None:
                continue
            if not numeric:
                raise ValueError("numeric default or bounds declared on a non-numeric position")
            if not _DECIMAL.fullmatch(bound):
                raise ValueError("value policy default and bounds must be canonical decimals")
        low = None if self.minimum is None else Decimal(self.minimum)
        high = None if self.maximum is None else Decimal(self.maximum)
        mid = None if self.default is None else Decimal(self.default)
        if low is not None and high is not None and low > high:
            raise ValueError("value policy minimum exceeds its maximum")
        if mid is not None and ((low is not None and mid < low) or (high is not None and mid > high)):
            raise ValueError("value policy default falls outside its declared bounds")


class SnapshotConstraints:
    """Immutable membership facts, optionally carrying verified compiler provenance.

    ``source_generation_context_digest`` is supplied only by the snapshot compiler in
    ``execution_llm.schema.projections``. Standalone validation (``parse_decision``) may
    leave it unset; ``accept_assistant_decision`` in ``execution_llm.schema.context``
    requires it to equal the digest of the expected generation context.
    """

    __slots__ = (  # noqa: RUF023 - grouped by canonical ref and policy categories
        "semantic_refs",
        "objective_refs",
        "constraint_refs",
        "entity_refs",
        "field_refs",
        "query_refs",
        "slot_refs",
        "parameter_refs",
        "catalog_ids_by_entity",
        "enum_refs",
        "unit_refs",
        "constraint_argument_slots",
        "constraint_scope_entity_refs",
        "capability_required_fields",
        "query_argument_slots",
        "query_entity_refs",
        "field_entity_refs",
        "slot_value_policies",
        "parameter_value_policies",
        "field_value_policies",
        "allowed_composition_methods",
        "max_weighted_terms",
        "source_generation_context_digest",
    )

    def __init__(
        self,
        *,
        objective_refs: Iterable[str] = (),
        constraint_refs: Iterable[str] = (),
        entity_refs: Iterable[str] = (),
        field_refs: Iterable[str] = (),
        query_refs: Iterable[str] = (),
        slot_refs: Iterable[str] = (),
        parameter_refs: Iterable[str] = (),
        catalog_ids_by_entity: Mapping[str, Iterable[str]],
        enum_refs: Iterable[str] = (),
        unit_refs: Iterable[str] = (),
        constraint_argument_slots: Mapping[str, Iterable[str]] = MappingProxyType({}),
        constraint_scope_entity_refs: Mapping[str, Iterable[str]] = MappingProxyType({}),
        capability_required_fields: Mapping[str, Iterable[str]] = MappingProxyType({}),
        query_argument_slots: Mapping[str, Iterable[str]] = MappingProxyType({}),
        query_entity_refs: Mapping[str, str] = MappingProxyType({}),
        field_entity_refs: Mapping[str, str] = MappingProxyType({}),
        slot_value_policies: Mapping[str, ValuePolicy] = MappingProxyType({}),
        parameter_value_policies: Mapping[str, ValuePolicy] = MappingProxyType({}),
        field_value_policies: Mapping[str, ValuePolicy] = MappingProxyType({}),
        allowed_composition_methods: Iterable[str] = ("weighted_sum", "lexicographic", "pareto"),
        max_weighted_terms: int | None = None,
        source_generation_context_digest: str | None = None,
    ) -> None:
        self.objective_refs = frozenset(objective_refs)
        self.constraint_refs = frozenset(constraint_refs)
        self.entity_refs = frozenset(entity_refs)
        self.field_refs = frozenset(field_refs)
        self.query_refs = frozenset(query_refs)
        self.slot_refs = frozenset(slot_refs)
        self.parameter_refs = frozenset(parameter_refs)
        self.enum_refs = frozenset(enum_refs)
        self.unit_refs = frozenset(unit_refs)
        self.semantic_refs = frozenset().union(
            self.objective_refs,
            self.constraint_refs,
            self.entity_refs,
            self.field_refs,
            self.query_refs,
            self.slot_refs,
            self.parameter_refs,
            self.enum_refs,
            self.unit_refs,
        )
        self.catalog_ids_by_entity = MappingProxyType(
            {entity: frozenset(ids) for entity, ids in catalog_ids_by_entity.items()}
        )
        self.constraint_argument_slots = _frozen_set_map(constraint_argument_slots)
        self.constraint_scope_entity_refs = _frozen_set_map(constraint_scope_entity_refs)
        self.capability_required_fields = _frozen_set_map(capability_required_fields)
        self.query_argument_slots = _frozen_set_map(query_argument_slots)
        self.query_entity_refs = MappingProxyType(dict(query_entity_refs))
        self.field_entity_refs = MappingProxyType(dict(field_entity_refs))
        self.slot_value_policies = MappingProxyType(dict(slot_value_policies))
        self.parameter_value_policies = MappingProxyType(dict(parameter_value_policies))
        self.field_value_policies = MappingProxyType(dict(field_value_policies))
        self.allowed_composition_methods = frozenset(allowed_composition_methods)
        if self.allowed_composition_methods - {"weighted_sum", "lexicographic", "pareto"}:
            raise ValueError("snapshot contains an unknown composition method")
        if max_weighted_terms is not None and (type(max_weighted_terms) is not int or max_weighted_terms < 0):
            raise ValueError("maximum weighted terms must be a non-negative integer")
        self.max_weighted_terms = max_weighted_terms
        if source_generation_context_digest is not None and not re.fullmatch(
            r"[0-9a-f]{64}", source_generation_context_digest
        ):
            raise ValueError("source generation context digest must be lowercase SHA-256 hex")
        self.source_generation_context_digest = source_generation_context_digest
        _validate_snapshot_constraints(self)

    def __setattr__(self, name: str, value: Any) -> None:
        if hasattr(self, name):
            raise AttributeError("SnapshotConstraints is immutable")
        object.__setattr__(self, name, value)


def _frozen_set_map(values: Mapping[str, Iterable[str]]) -> Mapping[str, frozenset[str]]:
    return MappingProxyType({key: frozenset(items) for key, items in values.items()})


def _validate_snapshot_constraints(snapshot: SnapshotConstraints) -> None:
    kinds = (
        snapshot.objective_refs,
        snapshot.constraint_refs,
        snapshot.entity_refs,
        snapshot.field_refs,
        snapshot.query_refs,
        snapshot.slot_refs,
        snapshot.parameter_refs,
        snapshot.enum_refs,
        snapshot.unit_refs,
    )
    if sum(map(len, kinds)) != len(set().union(*kinds)):
        raise ValueError("semantic refs must belong to exactly one declared kind")
    if set(snapshot.catalog_ids_by_entity) - snapshot.entity_refs:
        raise ValueError("catalog entity keys must be declared entity refs")
    checks = (
        (snapshot.constraint_argument_slots, snapshot.constraint_refs, snapshot.slot_refs),
        (snapshot.constraint_scope_entity_refs, snapshot.constraint_refs, snapshot.entity_refs),
        (snapshot.query_argument_slots, snapshot.query_refs, snapshot.slot_refs),
    )
    for mapping, owners, values in checks:
        if set(mapping) - owners or set().union(*mapping.values(), set()) - values:
            raise ValueError("snapshot ownership policy contains a wrong-kind ref")
    # Capability -> required-field edges. Owners span both capability kinds because every domain
    # declares the relation on objectives as well as constraints.
    if (
        set(snapshot.capability_required_fields) - (snapshot.objective_refs | snapshot.constraint_refs)
        or set().union(*snapshot.capability_required_fields.values(), set()) - snapshot.field_refs
    ):
        raise ValueError("capability required fields contain a wrong-kind ref")
    if (
        set(snapshot.query_entity_refs) - snapshot.query_refs
        or set(snapshot.query_entity_refs.values()) - snapshot.entity_refs
    ):
        raise ValueError("query ownership policy contains a wrong-kind ref")
    if (
        set(snapshot.field_entity_refs) - snapshot.field_refs
        or set(snapshot.field_entity_refs.values()) - snapshot.entity_refs
    ):
        raise ValueError("field ownership policy contains a wrong-kind ref")
    policy_checks = (
        (snapshot.slot_value_policies, snapshot.slot_refs),
        (snapshot.parameter_value_policies, snapshot.parameter_refs),
        (snapshot.field_value_policies, snapshot.field_refs),
    )
    for policies, refs in policy_checks:
        if set(policies) - refs:
            raise ValueError("value policy contains a wrong-kind owner ref")
        for policy in policies.values():
            if policy.enum_refs - snapshot.enum_refs or policy.unit_refs - snapshot.unit_refs:
                raise ValueError("value policy contains an undeclared enum or unit ref")
            if policy.ref_target_entity_refs - snapshot.entity_refs:
                raise ValueError("value policy contains an undeclared target entity ref")


@dataclass(frozen=True, slots=True)
class _ObjectPairs:
    pairs: list[tuple[str, Any]]


@dataclass(frozen=True, slots=True)
class _NumberToken:
    lexeme: str
    value: int | Decimal


@dataclass(frozen=True, slots=True)
class JsonKeyOccurrence:
    path: str
    key: str


@dataclass(frozen=True, slots=True)
class JsonNumberToken:
    path: str
    lexeme: str


@dataclass(frozen=True, slots=True)
class JsonEvidence:
    key_occurrences: tuple[JsonKeyOccurrence, ...]
    number_tokens: tuple[JsonNumberToken, ...]


@dataclass(frozen=True, slots=True)
class DecisionParseResult:
    body: DecisionBody
    evidence: JsonEvidence


def _strict_json(content: str | bytes) -> tuple[Any, JsonEvidence]:
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="strict")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("assistant content must be one JSON value")

    def integer(token: str) -> _NumberToken:
        if token == "-0":
            raise ValueError("raw -0 is forbidden")
        return _NumberToken(token, int(token))

    def finite_decimal(token: str) -> _NumberToken:
        return _NumberToken(token, Decimal(token))

    def reject_nonfinite(token: str) -> Any:
        raise ValueError(f"non-finite JSON number token: {token}")

    try:
        parsed = json.loads(
            content,
            object_pairs_hook=_ObjectPairs,
            parse_int=integer,
            parse_float=finite_decimal,
            parse_constant=reject_nonfinite,
        )
    except json.JSONDecodeError as exc:
        raise ValueError("assistant content must be exactly one strict JSON value") from exc
    keys: list[JsonKeyOccurrence] = []
    numbers: list[JsonNumberToken] = []
    materialized = _materialize_json(parsed, key_evidence=keys, number_evidence=numbers)
    return materialized, JsonEvidence(tuple(keys), tuple(numbers))


def _materialize_json(
    value: Any,
    path: str = "$",
    *,
    key_evidence: list[JsonKeyOccurrence],
    number_evidence: list[JsonNumberToken],
) -> Any:
    if isinstance(value, _ObjectPairs):
        result: dict[str, Any] = {}
        for key, child in value.pairs:
            key_path = _json_pointer(path, key)
            key_evidence.append(JsonKeyOccurrence(key_path, key))
            if key in result:
                raise ValueError(f"duplicate object key at {key_path}")
            result[key] = _materialize_json(
                child,
                key_path,
                key_evidence=key_evidence,
                number_evidence=number_evidence,
            )
        return result
    if isinstance(value, list):
        return [
            _materialize_json(
                child,
                _json_pointer(path, str(index)),
                key_evidence=key_evidence,
                number_evidence=number_evidence,
            )
            for index, child in enumerate(value)
        ]
    if isinstance(value, _NumberToken):
        number_evidence.append(JsonNumberToken(path, value.lexeme))
        return value.value
    return value


def _json_pointer(path: str, segment: str) -> str:
    return f"{path}/{segment.replace('~', '~0').replace('/', '~1')}"


def _validate_membership(body: DecisionBody, snapshot: SnapshotConstraints) -> None:
    if isinstance(body, ExecuteBody):
        if body.objectives.method not in snapshot.allowed_composition_methods:
            raise ValueError("objective composition method is not declared by the snapshot")
        if (
            isinstance(body.objectives, WeightedSum)
            and snapshot.max_weighted_terms is not None
            and len(body.objectives.terms) > snapshot.max_weighted_terms
        ):
            raise ValueError("weighted objective composition exceeds the snapshot term limit")
        for term in body.objectives.terms:
            _require_ref(term.objective_ref, snapshot.objective_refs, "objective")
        for constraint in body.constraints:
            _require_ref(constraint.constraint_ref, snapshot.constraint_refs, "constraint")
            _validate_arguments(
                constraint.arguments,
                snapshot.constraint_argument_slots.get(constraint.constraint_ref, frozenset()),
                snapshot,
            )
            allowed_scope_entities = snapshot.constraint_scope_entity_refs.get(constraint.constraint_ref, frozenset())
            if constraint.scope is None:
                if allowed_scope_entities:
                    raise ValueError("constraint scope is required by its closed policy")
            else:
                _validate_selection(constraint.scope, snapshot)
                if constraint.scope.entity_ref not in allowed_scope_entities:
                    raise ValueError("constraint scope uses a disallowed entity ref")
        for parameter in body.parameters:
            _require_ref(parameter.parameter_ref, snapshot.parameter_refs, "parameter")
            _validate_owned_value(
                parameter.value,
                snapshot.parameter_value_policies.get(parameter.parameter_ref),
                snapshot,
                f"parameter {parameter.parameter_ref}",
            )
        for selection in body.data_binding.selections:
            _validate_selection(selection, snapshot)
        for override in body.data_binding.overrides:
            _require_ref(override.field_ref, snapshot.field_refs, "field")
            _validate_selection(override.selection, snapshot)
            if snapshot.field_entity_refs.get(override.field_ref) != override.selection.entity_ref:
                raise ValueError("field override selection has the wrong owner entity")
            _validate_owned_value(
                override.value,
                snapshot.field_value_policies.get(override.field_ref),
                snapshot,
                f"field {override.field_ref}",
            )
    elif isinstance(body, ClarifyBody):
        _validate_clarify_membership(body, snapshot)
    else:
        allowed_related_refs = (
            snapshot.objective_refs if body.reason_code == "unsupported_composition" else snapshot.semantic_refs
        )
        unknown = set(body.related_refs) - allowed_related_refs
        if unknown:
            raise ValueError(f"unknown or wrong-kind related refs: {sorted(unknown)}")


def _require_ref(value: str, allowed: frozenset[str], kind: str) -> None:
    if value not in allowed:
        raise ValueError(f"unknown or wrong-kind {kind} ref: {value}")


def _validate_arguments(
    arguments: Sequence[SlotArgument],
    allowed_slots: frozenset[str],
    snapshot: SnapshotConstraints,
) -> None:
    for argument in arguments:
        _require_ref(argument.slot_ref, snapshot.slot_refs, "slot")
        if argument.slot_ref not in allowed_slots:
            raise ValueError("argument slot does not belong to its constraint or query")
        _validate_owned_value(
            argument.value,
            snapshot.slot_value_policies.get(argument.slot_ref),
            snapshot,
            f"slot {argument.slot_ref}",
        )


def _validate_selection(
    selection: AllSelection | IdsSelection | QuerySelection,
    snapshot: SnapshotConstraints,
) -> None:
    _require_ref(selection.entity_ref, snapshot.entity_refs, "entity")
    if isinstance(selection, IdsSelection):
        known = snapshot.catalog_ids_by_entity.get(selection.entity_ref, frozenset())
        if not set(selection.ids) <= known:
            raise ValueError(f"unknown catalog IDs for {selection.entity_ref}")
    elif isinstance(selection, QuerySelection):
        _require_ref(selection.query_ref, snapshot.query_refs, "query")
        if snapshot.query_entity_refs.get(selection.query_ref) != selection.entity_ref:
            raise ValueError("query selection has the wrong owner entity")
        _validate_arguments(
            selection.arguments,
            snapshot.query_argument_slots.get(selection.query_ref, frozenset()),
            snapshot,
        )


def _validate_owned_value(
    value: TypedValue,
    policy: ValuePolicy | None,
    snapshot: SnapshotConstraints,
    position: str,
) -> None:
    if policy is None:
        raise ValueError(f"missing closed value policy for {position}")
    if isinstance(value, ListValue):
        container, scalars = "list", value.values
    elif isinstance(value, RangeValue):
        container = "range"
        scalars = [item for item in (value.lower, value.upper) if item is not None]
    else:
        container, scalars = "scalar", [value]
    if container != policy.container:
        raise ValueError(f"wrong value container for {position}")
    for scalar in scalars:
        scalar_type = scalar.type
        if scalar_type != policy.value_type:
            raise ValueError(f"wrong scalar type for {position}")
        if isinstance(scalar, EnumValue):
            if scalar.value_ref not in snapshot.enum_refs or scalar.value_ref not in policy.enum_refs:
                raise ValueError(f"unknown or disallowed enum ref: {scalar.value_ref}")
        elif isinstance(scalar, DurationValue | QuantityValue):
            if scalar.unit_ref not in snapshot.unit_refs or scalar.unit_ref not in policy.unit_refs:
                raise ValueError(f"unknown or disallowed unit ref: {scalar.unit_ref}")
        elif isinstance(scalar, RefValue):
            matches = sum(
                scalar.value_ref in snapshot.catalog_ids_by_entity.get(entity, ())
                for entity in policy.ref_target_entity_refs
            )
            if matches != 1:
                raise ValueError(f"typed ref must resolve exactly once: {scalar.value_ref}")


def _validate_clarify_membership(body: ClarifyBody, snapshot: SnapshotConstraints) -> None:
    candidate_kinds = {
        "ambiguous_capability": snapshot.objective_refs
        | snapshot.constraint_refs
        | snapshot.query_refs
        | snapshot.parameter_refs,
        "ambiguous_scope": snapshot.entity_refs,
        "unit_ambiguity": snapshot.unit_refs,
        "composition_choice_required": snapshot.objective_refs,
        "hardness_ambiguity": snapshot.constraint_refs,
        "form_choice_required": snapshot.constraint_refs,
    }
    allowed_candidates = candidate_kinds.get(body.reason_code, frozenset())
    if set(body.candidate_refs) - allowed_candidates:
        raise ValueError("clarify candidate contains an unknown or wrong-kind ref")
    if set(body.missing_field_refs) - snapshot.field_refs:
        raise ValueError("clarify missing field contains an unknown or wrong-kind ref")
    compatible_units: set[str] = set()
    for required in body.required_inputs:
        policy: ValuePolicy | None = None
        if isinstance(required, SlotInput):
            _require_ref(required.slot_ref, snapshot.slot_refs, "slot")
            policy = snapshot.slot_value_policies.get(required.slot_ref)
        elif isinstance(required, ParameterInput):
            _require_ref(required.parameter_ref, snapshot.parameter_refs, "parameter")
            policy = snapshot.parameter_value_policies.get(required.parameter_ref)
        if body.reason_code == "unit_ambiguity" and isinstance(required, DecisionFieldInput):
            raise ValueError("unit ambiguity may identify only slots or parameters")
        if isinstance(required, SlotInput | ParameterInput) and policy is None:
            raise ValueError("required input has no closed value policy")
        if body.reason_code == "unit_ambiguity":
            if policy is None or policy.value_type not in {"duration", "quantity"}:
                raise ValueError("unit ambiguity requires a unit-bearing value position")
            compatible_units.update(policy.unit_refs)
    if body.reason_code == "unit_ambiguity" and set(body.candidate_refs) != compatible_units:
        raise ValueError("unit ambiguity candidates must exactly cover compatible units")


def parse_decision(content: str | bytes, snapshot: SnapshotConstraints) -> DecisionBody:
    """Parse one JSON value from assistant content and validate it as a DecisionBody against ``snapshot``."""

    return parse_decision_with_evidence(content, snapshot).body


def parse_decision_with_evidence(content: str | bytes, snapshot: SnapshotConstraints) -> DecisionParseResult:
    """Parse one body and retain immutable raw key/number token evidence."""

    raw, evidence = _strict_json(content)
    if not isinstance(raw, dict):
        raise ValueError("DecisionBody must be a JSON object")  # noqa: TRY004 - parse failure
    for value in _walk_json(raw):
        if isinstance(value, str) and any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            raise ValueError("JSON strings must contain Unicode scalar values")
    body = _BODY_ADAPTER.validate_python(raw, strict=True)
    _validate_membership(body, snapshot)
    _validate_safe_integers(body)
    return DecisionParseResult(body, evidence)


def _validate_safe_integers(body: DecisionBody) -> None:
    for value in _walk_json(body.model_dump(mode="json")):
        if type(value) is int and not -SAFE_INT <= value <= SAFE_INT:
            raise ValueError("JSON integer is outside the canonical safe range")


def _walk_json(value: Any):
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json(child)


def normalize_decision(body: DecisionBody) -> DecisionBody:
    """Return the canonical form: datetimes at +09:00 and weighted-sum weights as reduced shares summing to one."""

    data = body.model_dump(mode="json")
    for value in _walk_json(data):
        if isinstance(value, dict) and value.get("type") == "datetime":
            value["value"] = _project_datetime(value["value"])
    if isinstance(body, ExecuteBody) and isinstance(body.objectives, WeightedSum):
        coefficients = [Fraction(term.weight) for term in body.objectives.terms]
        total = sum(coefficients, Fraction())
        for term, share in zip(data["objectives"]["terms"], coefficients, strict=True):
            reduced = share / total
            term["weight"] = f"{reduced.numerator}/{reduced.denominator}"
    return validate_normalized_decision(_BODY_ADAPTER.validate_python(data, strict=True))


def validate_normalized_decision(body: DecisionBody) -> DecisionBody:
    """Return ``body`` re-validated; raise ``ValueError`` unless it is in normalized form.

    Normalized form means weighted-sum shares that are reduced and sum to one, datetimes at
    +09:00 and integers inside the safe range.
    """

    body = _BODY_ADAPTER.validate_python(body.model_dump(mode="json"), strict=True)
    _validate_safe_integers(body)
    if isinstance(body, ExecuteBody) and isinstance(body.objectives, WeightedSum):
        shares = [Fraction(term.weight) for term in body.objectives.terms]
        if sum(shares, Fraction()) != 1 or any(
            term.weight != f"{share.numerator}/{share.denominator}"
            for term, share in zip(body.objectives.terms, shares, strict=True)
        ):
            raise ValueError("normalized objective shares must be reduced and sum exactly to one")
    for value in _walk_json(body.model_dump(mode="json")):
        if (
            isinstance(value, dict)
            and value.get("type") == "datetime"
            and (not value["value"].endswith("+09:00") or _project_datetime(value["value"]) != value["value"])
        ):
            raise ValueError("normalized datetime must use canonical +09:00 spelling")
    return body


def jcs_bytes(value: Any) -> bytes:
    """Serialize ``value`` as canonical JSON bytes (sorted keys, compact separators, UTF-8).

    Only exact JSON types are accepted: no floats, no integers outside the safe range, ASCII
    object keys. A decision body must already be in normalized form.
    """

    if isinstance(value, ExecuteBody | ClarifyBody | UnsupportedBody):
        validate_normalized_decision(value)
    value = _plain(value)
    _validate_exact_json(value)
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _validate_exact_json(value: Any) -> None:
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str or not key.isascii():
                raise ValueError("canonical profile object keys must be ASCII strings")
            _validate_exact_json(child)
    elif type(value) is list:
        for child in value:
            _validate_exact_json(child)
    elif type(value) is str:
        if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            raise ValueError("JSON strings must contain Unicode scalar values")
    elif type(value) is int:
        if not -SAFE_INT <= value <= SAFE_INT:
            raise ValueError("integer is outside the canonical safe range")
    elif value is not None and type(value) is not bool:
        raise ValueError("canonical profile accepts exact JSON types only")


def decision_body_json_schema() -> dict[str, Any]:
    """Return the JSON Schema for ``DecisionBody``, generated from the models in this module."""

    return _BODY_ADAPTER.json_schema()


def normalized_jcs_bytes(body: DecisionBody) -> bytes:
    """Normalize ``body`` (datetimes and objective shares) and serialize it as canonical JSON bytes."""

    return jcs_bytes(normalize_decision(body))


__all__ = [  # noqa: RUF022 - public contract order
    "DecisionBody",
    "ExecuteBody",
    "ClarifyBody",
    "UnsupportedBody",
    "SnapshotConstraints",
    "decision_body_json_schema",
    "jcs_bytes",
    "normalize_decision",
    "normalized_jcs_bytes",
    "parse_decision",
    "parse_decision_with_evidence",
    "validate_normalized_decision",
    "ValuePolicy",
    "DecisionParseResult",
    "JsonEvidence",
    "JsonKeyOccurrence",
    "JsonNumberToken",
]
