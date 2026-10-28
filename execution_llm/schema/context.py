"""Decision boundary for decision schema 3.0 (the version string each decision envelope carries): verify
trusted inputs, then parse and seal one decision."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
from typing import Annotated, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from execution_llm.schema.decision import (
    DecisionBody,
    JsonEvidence,
    SnapshotConstraints,
    jcs_bytes,
    normalize_decision,
    parse_decision_with_evidence,
    validate_normalized_decision,
)

NonEmptyString = Annotated[str, Field(strict=True, min_length=1)]
Sha256Hex = Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]


class _FrozenClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class SemanticManifestRef(_FrozenClosedModel):
    server_id: NonEmptyString
    manifest_schema_version: Literal["1.0"]
    semantic_version: NonEmptyString
    digest: Sha256Hex


class CatalogRef(_FrozenClosedModel):
    dataset_ref: NonEmptyString
    catalog_schema_version: Literal["1.0"]
    version: NonEmptyString | None
    digest: Sha256Hex


class BindingManifestRef(_FrozenClosedModel):
    server_id: NonEmptyString
    binding_schema_version: Literal["1.0"]
    binding_version: NonEmptyString
    digest: Sha256Hex


class GenerationContext(_FrozenClosedModel):
    canonical_schema_version: Literal["3.0"]
    semantic_manifest_ref: SemanticManifestRef
    catalog_ref: CatalogRef


class ComponentRef(_FrozenClosedModel):
    version: NonEmptyString
    digest: Sha256Hex


class ComponentBundle(_FrozenClosedModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, serialize_by_alias=True)

    parser: ComponentRef
    schema_component: ComponentRef = Field(alias="schema")
    validator: ComponentRef
    normalizer: ComponentRef
    renderer: ComponentRef
    comparator: ComponentRef


class CompatibilityTuple(_FrozenClosedModel):
    canonical_schema_version: Literal["3.0"]
    semantic_manifest_ref: SemanticManifestRef
    catalog_ref: CatalogRef
    binding_manifest_ref: BindingManifestRef
    components: ComponentBundle

    @classmethod
    def from_context(
        cls,
        context: GenerationContext,
        binding_manifest_ref: BindingManifestRef,
        components: ComponentBundle,
    ) -> CompatibilityTuple:
        return cls(
            canonical_schema_version=context.canonical_schema_version,
            semantic_manifest_ref=context.semantic_manifest_ref,
            catalog_ref=context.catalog_ref,
            binding_manifest_ref=binding_manifest_ref,
            components=components,
        )


@dataclass(frozen=True, slots=True, init=False)
class CompatibilityRegistry:
    approved: frozenset[CompatibilityTuple]

    def __init__(self, approved: Iterable[CompatibilityTuple]) -> None:
        items = frozenset(approved)
        if any(type(item) is not CompatibilityTuple for item in items):
            raise TypeError("compatibility registry accepts exact CompatibilityTuple values")
        object.__setattr__(self, "approved", items)

    def validate_generation(self, components: ComponentBundle, context: GenerationContext) -> None:
        if not any(
            item.canonical_schema_version == context.canonical_schema_version
            and item.semantic_manifest_ref == context.semantic_manifest_ref
            and item.catalog_ref == context.catalog_ref
            and item.components == components
            for item in self.approved
        ):
            raise ValueError("generation components and context are not an approved compatibility tuple")

    def validate_binding(
        self,
        components: ComponentBundle,
        context: GenerationContext,
        binding_manifest_ref: BindingManifestRef,
    ) -> None:
        if CompatibilityTuple.from_context(context, binding_manifest_ref, components) not in self.approved:
            raise ValueError("binding context is not an approved compatibility tuple")


@dataclass(frozen=True, slots=True)
class PinnedConstraintsBundle:
    """A generation context paired with the snapshot constraints compiled for exactly that context."""

    context: GenerationContext
    constraints: SnapshotConstraints

    def __post_init__(self) -> None:
        if type(self.context) is not GenerationContext or type(self.constraints) is not SnapshotConstraints:
            raise TypeError("pinned bundle requires exact context and constraints values")

    def validate(self, expected_context: GenerationContext) -> None:
        if expected_context != self.context:
            raise ValueError("assistant context does not match the pinned constraints bundle")
        if self.constraints.source_generation_context_digest != generation_context_digest(expected_context):
            raise ValueError("snapshot constraints lack the exact pinned generation context identity")


class AssistantFrame(_FrozenClosedModel):
    role: Literal["assistant"]
    content: str | bytes


class DecisionEnvelope(_FrozenClosedModel):
    canonical_schema_version: Literal["3.0"]
    semantic_manifest_ref: SemanticManifestRef
    catalog_ref: CatalogRef
    body: DecisionBody

    @model_validator(mode="after")
    def require_normalized_body(self) -> DecisionEnvelope:
        validate_normalized_decision(self.body)
        return self


@dataclass(frozen=True, slots=True, init=False)
class DecisionArtifact:
    canonical_envelope_bytes: bytes
    decision_digest: str
    evidence: JsonEvidence

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("DecisionArtifact can only be created by the atomic decision boundary")

    def _validate_seal(self) -> None:
        if type(self.canonical_envelope_bytes) is not bytes:
            raise TypeError("canonical envelope must be immutable bytes")
        envelope = DecisionEnvelope.model_validate_json(self.canonical_envelope_bytes, strict=True)
        if _envelope_jcs_bytes(envelope) != self.canonical_envelope_bytes:
            raise ValueError("envelope bytes are not canonical JCS")
        if self.decision_digest != sha256(self.canonical_envelope_bytes).hexdigest():
            raise ValueError("decision digest does not match the complete envelope")
        if type(self.evidence) is not JsonEvidence:
            raise TypeError("decision artifact requires immutable intake evidence")

    @property
    def envelope(self) -> DecisionEnvelope:
        return DecisionEnvelope.model_validate_json(self.canonical_envelope_bytes, strict=True)

    @property
    def context(self) -> GenerationContext:
        envelope = self.envelope
        return GenerationContext(
            canonical_schema_version=envelope.canonical_schema_version,
            semantic_manifest_ref=envelope.semantic_manifest_ref,
            catalog_ref=envelope.catalog_ref,
        )


def generation_context_digest(context: GenerationContext) -> str:
    return sha256(jcs_bytes(context)).hexdigest()


def accept_assistant_decision(
    *,
    components: ComponentBundle,
    registry: CompatibilityRegistry,
    expected_context: GenerationContext,
    pinned: PinnedConstraintsBundle,
    frame: object,
    previous: DecisionArtifact | None = None,
) -> DecisionArtifact:
    """Validate every trusted input, then atomically parse and seal one complete body."""

    trusted_inputs = (
        (components, ComponentBundle, "components"),
        (registry, CompatibilityRegistry, "registry"),
        (expected_context, GenerationContext, "expected context"),
        (pinned, PinnedConstraintsBundle, "pinned constraints"),
        (frame, AssistantFrame, "provider frame"),
    )
    for value, expected_type, name in trusted_inputs:
        if type(value) is not expected_type:
            raise TypeError(f"{name} must have exact type {expected_type.__name__}")
    if previous is not None and type(previous) is not DecisionArtifact:
        raise TypeError("previous decision must be a verified artifact")

    registry.validate_generation(components, expected_context)
    pinned.validate(expected_context)
    if previous is not None:
        previous._validate_seal()
        if previous.context != expected_context:
            raise ValueError("context changed; a new decision call is required")

    assistant_frame = cast(AssistantFrame, frame)
    parsed = parse_decision_with_evidence(assistant_frame.content, pinned.constraints)
    envelope = DecisionEnvelope(
        canonical_schema_version=expected_context.canonical_schema_version,
        semantic_manifest_ref=expected_context.semantic_manifest_ref,
        catalog_ref=expected_context.catalog_ref,
        body=normalize_decision(parsed.body),
    )
    canonical = _envelope_jcs_bytes(envelope)
    return _seal_decision_artifact(canonical, parsed.evidence)


def _seal_decision_artifact(canonical_envelope_bytes: bytes, evidence: JsonEvidence) -> DecisionArtifact:
    artifact = object.__new__(DecisionArtifact)
    object.__setattr__(artifact, "canonical_envelope_bytes", canonical_envelope_bytes)
    object.__setattr__(artifact, "decision_digest", sha256(canonical_envelope_bytes).hexdigest())
    object.__setattr__(artifact, "evidence", evidence)
    artifact._validate_seal()
    return artifact


def _envelope_jcs_bytes(envelope: DecisionEnvelope) -> bytes:
    validated = DecisionEnvelope.model_validate(envelope.model_dump(mode="json"), strict=True)
    return jcs_bytes(validated)


__all__ = [
    "AssistantFrame",
    "BindingManifestRef",
    "CatalogRef",
    "CompatibilityRegistry",
    "CompatibilityTuple",
    "ComponentBundle",
    "ComponentRef",
    "DecisionArtifact",
    "DecisionEnvelope",
    "GenerationContext",
    "PinnedConstraintsBundle",
    "SemanticManifestRef",
    "accept_assistant_decision",
    "generation_context_digest",
]
