"""The three released domains: one module per domain with its vocabulary and capability forms, and the
domain surfaces built from them (`SURFACES`)."""

from __future__ import annotations

from collections.abc import Mapping

from execution_llm.tier2.domains.nurse import NURSE_SPEC
from execution_llm.tier2.domains.retail import RETAIL_SPEC
from execution_llm.tier2.domains.routing import ROUTING_SPEC
from execution_llm.tier2.surface import DomainSurface

ROUTING_SURFACE = DomainSurface.build(ROUTING_SPEC)
NURSE_SURFACE = DomainSurface.build(NURSE_SPEC)
RETAIL_SURFACE = DomainSurface.build(RETAIL_SPEC)
SURFACES: Mapping[str, DomainSurface] = {"routing": ROUTING_SURFACE, "nurse": NURSE_SURFACE, "retail": RETAIL_SURFACE}
