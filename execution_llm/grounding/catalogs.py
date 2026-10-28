"""Authored catalog fixtures for nurse and retail: the catalog those domains' grounding inputs are
projected over (the published gold used them); routing uses the registry's own catalog.

A ``CorpusCatalog`` is **distinct from the registry's catalog projection**, which may declare
entities without member ids; that projection is not modified here. It carries plausible entity
members (ids for ``ids``-selections) and the field-presence map, which must cover the registry's
projected fields exactly. The fixtures declare every projected field present.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from execution_llm.schema.projections import CatalogEntity
from execution_llm.schema.registry import domain_projection

Domain = Literal["routing", "nurse", "retail"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CorpusCatalog(_Model):
    """One catalog fixture: entity members and field presence."""

    domain: Domain
    dataset_ref: str
    version: str
    entities: tuple[CatalogEntity, ...]
    # The field-presence map: every projected field ref lands in exactly one side
    # (sampler_inputs enforces the exact cover against the registry's semantic projection).
    present_field_refs: tuple[str, ...] = ()
    absent_field_refs: tuple[str, ...] = ()

    @model_validator(mode="after")
    def coherent(self) -> CorpusCatalog:
        refs = [entity.entity_ref for entity in self.entities]
        if refs != sorted(refs) or len(refs) != len(set(refs)):
            raise ValueError("catalog entities must be sorted and unique")
        for side in (self.present_field_refs, self.absent_field_refs):
            if list(side) != sorted(side) or len(side) != len(set(side)):
                raise ValueError("field-presence sides must be sorted and unique")
        if set(self.present_field_refs) & set(self.absent_field_refs):
            raise ValueError("a field ref cannot be both present and absent")
        return self


def _all_field_refs(domain: Domain) -> tuple[str, ...]:
    semantic, _, _ = domain_projection(domain)
    return tuple(field.ref for field in semantic.fields)


def nurse_fixture() -> CorpusCatalog:
    """Authored nurse catalog: a plausible ward roster and the three-shift pattern."""

    return CorpusCatalog(
        domain="nurse",
        dataset_ref="corpus-nurse-fixture",
        version="1",
        entities=(
            CatalogEntity(
                entity_ref="entity:shift_definitions",
                ids=("shift-day", "shift-evening", "shift-night"),
            ),
            CatalogEntity(
                entity_ref="entity:staff",
                ids=tuple(f"nurse-{index:02d}" for index in range(1, 13)),
            ),
        ),
        present_field_refs=_all_field_refs("nurse"),
    )


def retail_fixture() -> CorpusCatalog:
    """Authored retail catalog: a small store network over a sku roster."""

    return CorpusCatalog(
        domain="retail",
        dataset_ref="corpus-retail-fixture",
        version="1",
        entities=(
            CatalogEntity(entity_ref="entity:item", ids=tuple(f"sku-{index:02d}" for index in range(1, 13))),
            CatalogEntity(entity_ref="entity:store", ids=("store-01", "store-02", "store-03")),
        ),
        present_field_refs=_all_field_refs("retail"),
    )
