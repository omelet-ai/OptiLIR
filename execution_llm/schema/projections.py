"""Portable projection schemas and snapshot construction; no bundled registry data."""

from __future__ import annotations

from decimal import Decimal
from hashlib import sha256
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from execution_llm.schema.context import SemanticManifestRef
from execution_llm.schema.decision import ScalarType, SnapshotConstraints, ValuePolicy, is_canonical_decimal, jcs_bytes

NonEmpty = Annotated[str, Field(strict=True, min_length=1)]
Sha256Hex = Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]
Binary64Hex = Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{16}$")]
Status = Literal["EXACT", "UNSUPPORTED"]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class SourceProvenance(_FrozenModel):
    declared_element: NonEmpty
    source_root: NonEmpty
    source_path: NonEmpty
    immutable_version: NonEmpty
    source_sha256: Sha256Hex
    source_excerpt: NonEmpty
    observed: NonEmpty
    expected: NonEmpty
    notes: NonEmpty


_NUMERIC_TYPES = frozenset({"integer", "decimal", "duration", "quantity"})


def _canonical_refs(refs: tuple[str, ...], what: str) -> None:
    if list(refs) != sorted(refs) or len(refs) != len(set(refs)):
        raise ValueError(f"{what} must be sorted and unique")


def _decimal_facts(*values: str | None) -> None:
    for value in values:
        if value is not None and not is_canonical_decimal(value):
            raise ValueError("declared numeric facts must be canonical decimals")


def _bounds_coherent(minimum: str | None, maximum: str | None, default: str | None) -> None:
    """A declared interval must be non-empty and must contain its declared default."""
    low = None if minimum is None else Decimal(minimum)
    high = None if maximum is None else Decimal(maximum)
    mid = None if default is None else Decimal(default)
    if low is not None and high is not None and low > high:
        raise ValueError("declared minimum exceeds its maximum")
    if mid is not None and ((low is not None and mid < low) or (high is not None and mid > high)):
        raise ValueError("declared default falls outside its declared bounds")


class SemanticObjective(_FrozenModel):
    ref: NonEmpty
    direction: Literal["minimize", "maximize"]
    # Catalog fields the source declares this objective requires (capability -> required-field edge).
    required_field_refs: tuple[NonEmpty, ...] = ()
    # Request-side prepared data is distinct from catalog fields; conflating the two
    # would assert field-presence facts that the catalog cannot prove.
    required_data_refs: tuple[NonEmpty, ...] = ()

    @model_validator(mode="after")
    def canonical_requirements(self) -> SemanticObjective:
        _canonical_refs(self.required_field_refs, "objective required fields")
        _canonical_refs(self.required_data_refs, "objective required data")
        return self


class SemanticField(_FrozenModel):
    ref: NonEmpty
    value_type: Literal["text"] = "text"


class SoftWeightSchema(_FrozenModel):
    """Source-declared soft-weight facts for one constraint.

    A null bound records a fact -- "the pinned source declares no such bound" -- and is never an
    omission; a bound is not filled in without source evidence. Independently of what is declared
    here, an emitted soft weight is a non-negative canonical decimal (enforced on
    ``SoftStrength``), so a null minimum never admits a negative weight.
    """

    # Explicit soft selections without a weight must clarify this declared slot.
    # Its identifier is data, not inferred from a naming convention.
    slot_ref: NonEmpty | None = None
    minimum: NonEmpty | None = None
    maximum: NonEmpty | None = None
    default: NonEmpty | None = None

    @model_validator(mode="after")
    def source_derived_bounds(self) -> SoftWeightSchema:
        _decimal_facts(self.minimum, self.maximum, self.default)
        # Every bound shares the non-negative weight baseline, not just the minimum: a negative maximum
        # declares an interval no valid weight can occupy, and a negative default declares a
        # value SoftStrength always rejects.
        if any(value is not None and value.startswith("-") for value in (self.minimum, self.maximum, self.default)):
            raise ValueError("declared soft-weight facts cannot be negative")
        _bounds_coherent(self.minimum, self.maximum, self.default)
        return self


class SlotDeclaration(_FrozenModel):
    """One canonical argument slot with explicitly declared value facts."""

    ref: NonEmpty
    value_type: ScalarType
    container: Literal["scalar", "list", "range"] = "scalar"
    unit_ref: NonEmpty | None = None
    optional: bool = False
    default: NonEmpty | None = None
    minimum: NonEmpty | None = None
    maximum: NonEmpty | None = None
    empty_set_semantics: Literal["all", "none"] | None = None

    @model_validator(mode="after")
    def declared_facts_are_coherent(self) -> SlotDeclaration:
        _decimal_facts(self.default, self.minimum, self.maximum)
        if self.empty_set_semantics is not None and self.container != "list":
            raise ValueError("empty-set polarity applies only to list-valued slots")
        if (self.unit_ref is not None) != (self.value_type in {"duration", "quantity"}):
            raise ValueError("exactly duration and quantity slots carry a unit ref")
        # Kept in step with ValuePolicy, which this declaration projects into: numeric facts
        # belong only on numeric positions, and enum/ref positions need refs a slot cannot carry.
        if self.value_type not in _NUMERIC_TYPES and any(
            value is not None for value in (self.default, self.minimum, self.maximum)
        ):
            raise ValueError("numeric default or bounds declared on a non-numeric slot")
        if self.value_type in {"enum", "ref"}:
            raise ValueError("enum and ref slots need position-specific refs this declaration cannot carry")
        _bounds_coherent(self.minimum, self.maximum, self.default)
        return self


class SemanticConstraint(_FrozenModel):
    """One declared constraint capability.

    ``scope_policy`` is always ``forbidden``: none of the supported request surfaces exposes a
    per-constraint scope selector, so declaring a scope entity would invent semantics. Targeting
    specific riders or staff goes through argument slots or required data instead.
    """

    ref: NonEmpty
    predicate_kind: Literal["comparison", "opaque"]
    hardness_policy: Literal["user_selectable"] = "user_selectable"
    default_hardness: Literal["hard", "soft"]
    soft_weight_schema: SoftWeightSchema = SoftWeightSchema()
    scope_policy: Literal["forbidden"] = "forbidden"
    argument_slot_refs: tuple[NonEmpty, ...] = ()
    required_field_refs: tuple[NonEmpty, ...] = ()
    required_data_refs: tuple[NonEmpty, ...] = ()
    # Which quantity the constraint aggregates, and over which entity's cardinality. Showing that
    # two constraints contradict each other by reading alone needs both -- "staff per department x
    # shift x day" against "hours per nurse per week" is only a contradiction once the dimensions
    # are known. Null means the registry declares none.
    aggregation: Literal["sum", "count", "max", "min", "peak"] | None = None
    cardinality_entity_ref: NonEmpty | None = None

    @model_validator(mode="after")
    def canonical_requirements(self) -> SemanticConstraint:
        _canonical_refs(self.argument_slot_refs, "constraint argument slots")
        _canonical_refs(self.required_field_refs, "constraint required fields")
        _canonical_refs(self.required_data_refs, "constraint required data")
        return self


class UnitDefinition(_FrozenModel):
    ref: NonEmpty
    dimension: NonEmpty


DeferralReason = Literal[
    "not_in_served_vocabulary",
    "source_unpinnable",
    "no_capability_requires_it",
    "required_primitive_absent",
    "out_of_scope",
]


class DeferralRecord(_FrozenModel):
    """An element deliberately not declared, recorded so that no source element is omitted silently.

    ``source_unpinnable`` is the one reason with no source to cite: its evidence is that no pinnable
    source exists. Every other reason is expected to cite a pinned source (a ``SourceProvenance``
    record) as its deferral evidence.
    """

    element_ref: NonEmpty
    reason: DeferralReason


class SemanticManifestProjection(_FrozenModel):
    server_id: NonEmpty
    manifest_schema_version: Literal["1.0"] = "1.0"
    semantic_version: NonEmpty
    objectives: tuple[SemanticObjective, ...] = ()
    fields: tuple[SemanticField, ...] = ()
    constraints: tuple[SemanticConstraint, ...] = ()
    slots: tuple[SlotDeclaration, ...] = ()
    units: tuple[UnitDefinition, ...] = ()
    deferrals: tuple[DeferralRecord, ...] = ()
    composition_methods: tuple[Literal["weighted_sum"], ...] = ()
    max_weighted_terms: int = Field(default=0, strict=True, ge=0)
    default_weights: tuple[str, ...] = ()

    @model_validator(mode="after")
    def canonical_collections(self) -> SemanticManifestProjection:
        for items in (self.objectives, self.fields, self.constraints, self.slots, self.units):
            _canonical_refs(tuple(item.ref for item in items), "semantic elements")
        _canonical_refs(tuple(item.element_ref for item in self.deferrals), "deferral records")
        # Uniqueness must hold across collections, not only within each. Provenance records
        # (``SourceProvenance.declared_element``) and the declared/deferred check below key
        # elements by bare ref, so two kinds sharing one ref would collapse into a single declared
        # element and one provenance record could satisfy both.
        every_ref = [item.ref for item in (*self.objectives, *self.fields, *self.constraints, *self.slots, *self.units)]
        if len(every_ref) != len(set(every_ref)):
            raise ValueError("a ref may belong to exactly one declared kind")
        return self

    @model_validator(mode="after")
    def closed_cross_references(self) -> SemanticManifestProjection:
        slot_refs = {item.ref for item in self.slots}
        unit_refs = {item.ref for item in self.units}
        field_refs = {item.ref for item in self.fields}
        if self.default_weights and len(self.default_weights) != len(self.objectives):
            raise ValueError("default weights must cover every objective")
        if any(item.unit_ref is not None and item.unit_ref not in unit_refs for item in self.slots):
            raise ValueError("slot unit refs must be declared units")
        for item in self.constraints:
            if set(item.argument_slot_refs) - slot_refs:
                raise ValueError("constraint argument slots must be declared slots")
            weight_slot = item.soft_weight_schema.slot_ref
            if weight_slot is not None and weight_slot not in slot_refs:
                raise ValueError("soft-weight slot refs must be declared slots")
        for capability in (*self.objectives, *self.constraints):
            if set(capability.required_field_refs) - field_refs:
                raise ValueError("capability required fields must be declared fields")
        declared = {item.ref for item in (*self.objectives, *self.fields, *self.constraints, *self.slots, *self.units)}
        if declared & {item.element_ref for item in self.deferrals}:
            raise ValueError("an element cannot be both declared and deferred")
        return self


class CatalogEntity(_FrozenModel):
    entity_ref: NonEmpty
    ids: tuple[NonEmpty, ...] = ()

    @model_validator(mode="after")
    def canonical_ids(self) -> CatalogEntity:
        if list(self.ids) != sorted(self.ids) or len(self.ids) != len(set(self.ids)):
            raise ValueError("catalog ids must be sorted and unique")
        return self


class DataCatalogProjection(_FrozenModel):
    semantic_manifest_ref: SemanticManifestRef
    dataset_ref: NonEmpty
    catalog_schema_version: Literal["1.0"] = "1.0"
    version: NonEmpty | None
    entities: tuple[CatalogEntity, ...] = ()

    @model_validator(mode="after")
    def canonical_entities(self) -> DataCatalogProjection:
        refs = [item.entity_ref for item in self.entities]
        if refs != sorted(refs) or len(refs) != len(set(refs)):
            raise ValueError("catalog entities must be sorted and unique")
        return self


class BindingProjection(_FrozenModel):
    semantic_ref: NonEmpty
    request_status: Status
    result_status: Status
    checker_status: Status
    availability: Literal["executable", "mapping_only"]
    ineligibility_reason: NonEmpty | None = None

    @model_validator(mode="after")
    def coherent(self) -> BindingProjection:
        exact = self.request_status == self.result_status == self.checker_status == "EXACT"
        if exact != (self.availability == "executable"):
            raise ValueError("executable availability requires exactly all-EXACT statuses")
        if (self.availability == "mapping_only") != (self.ineligibility_reason is not None):
            raise ValueError("mapping-only availability requires exactly one ineligibility reason")
        return self


class BindingManifestProjection(_FrozenModel):
    semantic_manifest_ref: SemanticManifestRef
    server_id: NonEmpty
    binding_schema_version: Literal["1.0"] = "1.0"
    binding_version: NonEmpty
    admissible: bool
    rejection_reason: NonEmpty | None = None
    projections: tuple[BindingProjection, ...] = ()
    required_paths: tuple[NonEmpty, ...] = ()

    @model_validator(mode="after")
    def coherent(self) -> BindingManifestProjection:
        refs = [item.semantic_ref for item in self.projections]
        if refs != sorted(refs) or len(refs) != len(set(refs)):
            raise ValueError("binding projections must be sorted and unique")
        if self.admissible == (self.rejection_reason is not None):
            raise ValueError("exactly inadmissible bindings require a rejection reason")
        if self.server_id != self.semantic_manifest_ref.server_id:
            raise ValueError("binding server id must match its semantic manifest ref")
        return self


class CompilePolicy(_FrozenModel):
    domain: Literal["routing", "nurse", "retail"]
    binding_admissible: bool
    binding_rejection: NonEmpty | None = None
    projections: tuple[BindingProjection, ...] = ()
    required_paths: tuple[NonEmpty, ...] = ()


def _slot_value_policy(slot: SlotDeclaration) -> ValuePolicy:
    """Project one declared slot into the closed value policy the runtime validator consumes."""
    return ValuePolicy(
        value_type=slot.value_type,
        container=slot.container,
        unit_refs=frozenset() if slot.unit_ref is None else frozenset({slot.unit_ref}),
        optional=slot.optional,
        default=slot.default,
        minimum=slot.minimum,
        maximum=slot.maximum,
        empty_set_semantics=slot.empty_set_semantics,
    )


def _project_snapshot(
    semantic: SemanticManifestProjection,
    catalog: DataCatalogProjection,
    context_digest: str,
) -> SnapshotConstraints:
    """Carry every declared element of the manifest into the runtime membership snapshot.

    The decision validator checks bodies against this snapshot: a declared constraint missing here
    would be rejected as an unknown ref, and a slot missing here would have no value policy.
    ``required_data_refs`` and the aggregation facts (``aggregation``, ``cardinality_entity_ref``)
    are not carried over; the snapshot has no field for them.
    """
    return SnapshotConstraints(
        objective_refs=(item.ref for item in semantic.objectives),
        constraint_refs=(item.ref for item in semantic.constraints),
        # The catalog is where an entity set is declared, so its refs are the snapshot's entity
        # kind. Without them the id map keys nothing and a ref-valued position could resolve to
        # an entity the runtime never heard of.
        entity_refs=(item.entity_ref for item in catalog.entities),
        field_refs=(item.ref for item in semantic.fields),
        slot_refs=(item.ref for item in semantic.slots),
        unit_refs=(item.ref for item in semantic.units),
        catalog_ids_by_entity={item.entity_ref: item.ids for item in catalog.entities},
        constraint_argument_slots={
            item.ref: item.argument_slot_refs for item in semantic.constraints if item.argument_slot_refs
        },
        capability_required_fields={
            item.ref: item.required_field_refs
            for item in (*semantic.objectives, *semantic.constraints)
            if item.required_field_refs
        },
        slot_value_policies={item.ref: _slot_value_policy(item) for item in semantic.slots},
        allowed_composition_methods=semantic.composition_methods,
        max_weighted_terms=semantic.max_weighted_terms,
        source_generation_context_digest=context_digest,
    )


def artifact_digest(model: BaseModel) -> tuple[bytes, str]:
    data = jcs_bytes(model)
    return data, sha256(data).hexdigest()


def _semantic_ref(manifest: SemanticManifestProjection) -> tuple[bytes, SemanticManifestRef]:
    data, digest = artifact_digest(manifest)
    return data, SemanticManifestRef(
        server_id=manifest.server_id,
        manifest_schema_version=manifest.manifest_schema_version,
        semantic_version=manifest.semantic_version,
        digest=digest,
    )
