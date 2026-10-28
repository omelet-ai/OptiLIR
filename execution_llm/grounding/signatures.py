"""Per-domain GradingSignature data: what the decision comparator reads beside the snapshot.

Each signature carries the unit rescale table (``unit:{token}`` refs to their dimension's base unit:
DISTANCE m, DURATION s; routing also MASS kg and VOLUME m3), the COUNT-sorted seats (the ``day``
disambiguation) and the degenerate constraint/objective pairs (``g0_constraint_to_objective``).
Defaults and empty-set polarity are not here: the comparator reads them from the supplied registry.
These are the signatures ``inputs.sampler_inputs`` uses by default.

Notes on the entries:

- A unit row applies only where a registry declares that unit; a unit no row names is carried
  verbatim.
- Nurse maps ``unit:day`` and ``unit:hour`` to themselves: the nurse duration seats keep the unit they
  are read in, and the run-length seat ``slot:max_days`` uses ``day`` in its COUNT sense, which never
  rescales.
- The routing degenerate-pair map lists two pairs (``constraint:distance`` and
  ``constraint:duration``); the Execute form-defect probe in ``grading`` reads them only for a
  registry that declares those constraints.
- No TABLE-sorted slots and no ``entry_parameters`` are listed.
"""

from __future__ import annotations

from execution_llm.grading import GradingSignature

Domain = str

_DISTANCE_DURATION_TABLE: dict[str, tuple[str, int]] = {
    # DISTANCE base m; DURATION base s.
    "unit:m": ("unit:m", 1),
    "unit:km": ("unit:m", 1000),
    "unit:s": ("unit:s", 1),
    "unit:min": ("unit:s", 60),
    "unit:h": ("unit:s", 3600),
    "unit:day": ("unit:s", 86400),
}


def routing_signature() -> GradingSignature:
    return GradingSignature(
        unit_rescale={
            **_DISTANCE_DURATION_TABLE,
            # Per-key load[d] dimensions: MASS base kg, VOLUME base m3.
            "unit:kg": ("unit:kg", 1),
            "unit:m3": ("unit:m3", 1),
        },
        # Degenerate constraint/objective pairs: demoted constraint half -> canonical
        # objective half.
        g0_constraint_to_objective={
            "constraint:distance": "objective:minimize_distance",
            "constraint:duration": "objective:minimize_duration",
        },
    )


def nurse_signature() -> GradingSignature:
    return GradingSignature(
        # day's COUNT sense never rescales and its seat is listed in count_sorted_positions below,
        # so both units map to themselves.
        unit_rescale={
            "unit:day": ("unit:day", 1),
            "unit:hour": ("unit:hour", 1),
        },
        count_sorted_positions=frozenset({"slot:max_days"}),
    )


def retail_signature() -> GradingSignature:
    return GradingSignature(unit_rescale=dict(_DISTANCE_DURATION_TABLE))


def domain_signature(domain: Domain) -> GradingSignature:
    factory = {"routing": routing_signature, "nurse": nurse_signature, "retail": retail_signature}.get(domain)
    if factory is None:
        raise ValueError(f"no grading signature for domain {domain}")
    return factory()
