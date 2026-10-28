"""The per-domain grounding inputs and the decision builders the grounding shares.

`sampler_inputs` materializes one domain's pinned snapshot from the supplied registry's projections (a
catalog fixture, when given, supplies the entity members), and `SamplerInputs` carries it with the
domain's `GradingSignature`. The names come from the corpus sampler that built the dataset from the
same inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from execution_llm.grading import GradingSignature
from execution_llm.grounding.catalogs import CorpusCatalog
from execution_llm.grounding.signatures import domain_signature
from execution_llm.schema.context import (
    CatalogRef,
    GenerationContext,
    generation_context_digest,
)
from execution_llm.schema.decision import (
    ClarifyBody,
    ConstraintSelection,
    DataBinding,
    DecisionBody,
    ExecuteBody,
    SnapshotConstraints,
    WeightedSum,
    _validate_membership,
    normalize_decision,
)
from execution_llm.schema.projections import (
    DataCatalogProjection,
    SemanticManifestProjection,
    _project_snapshot,
    _semantic_ref,
    artifact_digest,
)
from execution_llm.schema.registry import domain_projection

Domain = Literal["routing", "nurse", "retail"]


class _Blocked(Exception):
    """The frames cannot be grounded as given; the grounder re-raises it as GroundingError."""


@dataclass(frozen=True)
class SamplerInputs:
    """The per-domain deterministic inputs that grounding reads."""

    domain: Domain
    semantic: SemanticManifestProjection
    snapshot: SnapshotConstraints
    signature: GradingSignature


def sampler_inputs(
    domain: Domain,
    *,
    corpus_catalog: CorpusCatalog | None = None,
    signature: GradingSignature | None = None,
) -> SamplerInputs:
    """Materialize the pinned snapshot for one domain from the registry projections.

    With a ``corpus_catalog``, the snapshot projects over its entity members (the
    catalog supplies the entities the decisions reference), and its field-presence map
    must cover the projected fields exactly. Without one, the registry's own catalog is
    used. ``signature`` defaults to ``domain_signature(domain)``.
    """

    semantic, _, _ = domain_projection(domain)
    _, semantic_ref = _semantic_ref(semantic)
    if corpus_catalog is None:
        _, catalog, _ = domain_projection(domain, semantic_ref)
        assert catalog is not None
    else:
        if corpus_catalog.domain != domain:
            raise ValueError(f"corpus catalog is for {corpus_catalog.domain}, not {domain}")
        declared = set(corpus_catalog.present_field_refs) | set(corpus_catalog.absent_field_refs)
        projected = {item.ref for item in semantic.fields}
        if declared != projected:
            raise ValueError(
                f"catalog field-presence map does not cover the projection exactly"
                f" (missing {sorted(projected - declared)}, extra {sorted(declared - projected)})"
            )
        catalog = DataCatalogProjection(
            semantic_manifest_ref=semantic_ref,
            dataset_ref=corpus_catalog.dataset_ref,
            version=corpus_catalog.version,
            entities=corpus_catalog.entities,
        )
    _, catalog_digest = artifact_digest(catalog)
    context = GenerationContext(
        canonical_schema_version="3.0",
        semantic_manifest_ref=semantic_ref,
        catalog_ref=CatalogRef(
            dataset_ref=catalog.dataset_ref,
            catalog_schema_version=catalog.catalog_schema_version,
            version=catalog.version,
            digest=catalog_digest,
        ),
    )
    snapshot = _project_snapshot(semantic, catalog, generation_context_digest(context))
    return SamplerInputs(
        domain=domain,
        semantic=semantic,
        snapshot=snapshot,
        signature=signature if signature is not None else domain_signature(domain),
    )


def _execute(objectives: WeightedSum, constraints: list[ConstraintSelection] = []) -> ExecuteBody:  # noqa: B006 - never mutated
    return ExecuteBody(
        kind="execute",
        objectives=objectives,
        constraints=constraints,
        parameters=[],
        data_binding=DataBinding(selections=[], overrides=[]),
    )


def _finalize_abstention(body: DecisionBody, inputs: SamplerInputs) -> DecisionBody:
    _validate_membership(body, inputs.snapshot)
    return normalize_decision(body)


def _clarify(code: str, *, candidates: list[str] = [], inputs_: list = [], fields_: list[str] = []) -> ClarifyBody:  # noqa: B006 - never mutated
    return ClarifyBody(
        kind="clarify",
        reason_code=code,  # type: ignore[arg-type]
        candidate_refs=sorted(candidates),
        required_inputs=inputs_,
        missing_field_refs=sorted(fields_),
    )
