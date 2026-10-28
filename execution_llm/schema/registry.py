"""Explicit, data-only registry input. Loading a registry never contacts a solver."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from execution_llm.schema.context import SemanticManifestRef
from execution_llm.schema.projections import (
    BindingManifestProjection,
    DataCatalogProjection,
    SemanticManifestProjection,
    _FrozenModel,
    _semantic_ref,
)

Domain = Literal["routing", "nurse", "retail"]


class RegistryEntry(_FrozenModel):
    semantic: SemanticManifestProjection
    catalog: DataCatalogProjection
    binding: BindingManifestProjection

    @model_validator(mode="after")
    def exact_links(self) -> RegistryEntry:
        _, ref = _semantic_ref(self.semantic)
        if self.catalog.semantic_manifest_ref != ref or self.binding.semantic_manifest_ref != ref:
            raise ValueError("registry artifacts must reference the exact semantic manifest digest")
        objectives = {item.ref for item in self.semantic.objectives}
        if any(item.semantic_ref not in objectives for item in self.binding.projections):
            raise ValueError("binding refs must belong to declared objectives")
        entities = {item.entity_ref for item in self.catalog.entities}
        if any(
            item.cardinality_entity_ref is not None and item.cardinality_entity_ref not in entities
            for item in self.semantic.constraints
        ):
            raise ValueError("cardinality refs must belong to declared catalog entities")
        return self


class Registry(_FrozenModel):
    schema_version: Literal["1"] = "1"
    domains: dict[Domain, RegistryEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def domain_identity(self) -> Registry:
        # The shipped grammar surfaces support these three domains only.
        servers = {"routing": "advanced-routing", "nurse": "nurse-scheduling", "retail": "retail-optimization"}
        for domain, entry in self.domains.items():
            if entry.semantic.server_id != servers[domain]:
                raise ValueError(f"registry domain {domain!r} does not match its semantic server id")
        return self


def load_registry(path: str | Path | None = None) -> Registry:
    """Read the registry JSON file at ``path`` or ``$EXECUTION_LLM_REGISTRY``; there is no bundled fallback."""
    selected = path if path is not None else os.environ.get("EXECUTION_LLM_REGISTRY")
    if not selected:
        raise ValueError("EXECUTION_LLM_REGISTRY is required: set it or pass the path of a registry JSON file")
    return Registry.model_validate_json(Path(selected).read_bytes(), strict=True)


def domain_projection(
    domain: Domain,
    semantic_ref: SemanticManifestRef | None = None,
    *,
    registry_path: str | Path | None = None,
) -> tuple[SemanticManifestProjection, DataCatalogProjection | None, BindingManifestProjection | None]:
    registry = load_registry(registry_path)
    if domain not in registry.domains:
        raise ValueError(f"domain {domain!r} is not declared in the supplied registry")
    entry = registry.domains[domain]
    if semantic_ref is None:
        return entry.semantic, None, None
    if semantic_ref != entry.catalog.semantic_manifest_ref:
        raise ValueError("requested semantic manifest ref does not match the supplied registry")
    return entry.semantic, entry.catalog, entry.binding
