"""The nurse domain: vocabulary and capability forms, keyed by registry refs. Each nurse capability is
one configuration of a rule family (coverage, assignment, balance, consecutive days, rest gap,
pattern); its form is that family's expression at that configuration.

Units: the nurse duration seats are read in `unit:hour` / `unit:day` (the nurse GradingSignature
rescales neither), so the spoken units map to those refs with factor 1 rather than to seconds, the
DURATION base of the other domains. `$max_run` (slot `slot:max_days`) is COUNT-sorted, so its day
value is never rescaled.
"""

from __future__ import annotations

from execution_llm.tier2.expr import Accessor, Measure, Vocabulary
from execution_llm.tier2.surface import DomainSpec, Extra, Signature, SlotSpec, signature

_M = Measure
NURSE_MEASURES: dict[str, Measure] = {
    "work_hours": _M("measure:work_hours", "DURATION", "assignment", "working hours"),
    "gap_duration": _M("measure:gap_duration", "DURATION", "pair", "rest gap between the two shifts"),
    "staff_count": _M("measure:staff_count", "COUNT", "shift_instance", "number of nurses on the shift"),
    "unmet_demand": _M("measure:unmet_demand", "COUNT", "shift_instance", "staffing shortfall against demand"),
    "overtime_hours": _M("measure:overtime_hours", "DURATION", "worker", "overtime hours beyond the weekly limit"),
    "preference_score": _M("measure:preference_score", "SCALAR", "assignment", "preference satisfaction score"),
    "workload": _M("measure:workload", "COUNT", "worker", "workload (shifts worked)"),
}
_A = Accessor
NURSE_ACCESSORS: dict[str, Accessor] = {
    "interval": _A("attr:interval", "INTERVAL", "the shift's time interval"),
    "works": _A("attr:works", "BOOLEAN", "works that day"),
    "group": _A("attr:group", "KEY", "the cost group"),
    "demand": _A("attr:demand", "COUNT", "the shift's demanded headcount"),
    "preferred": _A("attr:preferred", "BOOLEAN", "matches the nurse's stated availability preference"),
    "eligible": _A("attr:eligible", "BOOLEAN", "the nurse's worker type may take the shift"),
    "forbidden_pattern": _A("attr:forbidden_pattern", "PATTERN", "the forbidden shift pattern"),
    "preferred_pattern": _A("attr:preferred_pattern", "PATTERN", "the preferred shift pattern"),
}

NURSE_VOCABULARY = Vocabulary(
    domain="nurse",
    entity_sets={
        "workers": "entity:worker",
        "shift_instances": "entity:shift_instance",
        "assignments": "entity:assignment",
        "days": "entity:day",
    },
    population_aliases={"nurses": "workers", "staff": "workers", "shifts": "shift_instances"},
    containments={"assignments": "entity:assignment"},
    sequences={"assignment_seq": "attr:assignment_seq", "day_seq": "attr:day_seq", "shift_seq": "attr:shift_seq"},
    relations={},
    measures=NURSE_MEASURES,
    accessors=NURSE_ACCESSORS,
    units={
        "h": ("unit:hour", 1),
        "hour": ("unit:hour", 1),
        "hours": ("unit:hour", 1),
        "day": ("unit:day", 1),
        "days": ("unit:day", 1),
    },
    population_gloss={
        "slot": "selected nurse",
        "entity:worker": "nurse",
        "entity:shift_instance": "shift",
        "entity:assignment": "shift assignment",
        "entity:day": "day",
        "windows": "week",
        "adjacent": "pair of consecutive shifts",
    },
)


def _sig(ref: str, text: str, slots: dict[str, SlotSpec] | None = None, **kw) -> Signature:
    return signature(NURSE_VOCABULARY, ref, text, slots, **kw)


NURSE_SIGNATURES: tuple[Signature, ...] = (
    # objectives: the soft-policy configurations of the COVERAGE, ASSIGNMENT and BALANCE rule families
    _sig(
        "objective:overtime",
        "MINIMIZE( SUM(w in workers: overtime_hours(w)) )",
    ),
    _sig(
        "objective:preference_satisfaction",
        "MAXIMIZE( SUM(a in assignments: preference_score(a)) )",
    ),
    _sig("objective:unmet_demand", "MINIMIZE( SUM(s in shift_instances: unmet_demand(s)) )"),
    _sig("objective:workload_balance", "MINIMIZE( RANGE(w in workers: workload(w)) )"),
    # constraints
    _sig(
        "constraint:max_consecutive_days",
        "EACH(w in workers): RUN_LENGTH(d in day_seq(w): works(w, d)) <= $max_run",
        {"sig:max_run": SlotSpec("slot:max_days", "day", sort="COUNT")},
    ),
    _sig(
        "constraint:max_hours_per_week",
        "EACH(w in workers), EACH(win in windows(WEEK, 1)):"
        " SUM(a in assignments(w) WHERE DURING(interval(a), win): work_hours(a)) <= $max_hours",
        {"sig:max_hours": SlotSpec("slot:max_hours", "h", sort="DURATION")},
        family="hours",
    ),
    _sig(
        "constraint:min_rest_hours",
        "EACH(w in workers), EACH(g in adjacent(assignment_seq(w))): gap_duration(g) >= $min_gap",
        {"sig:min_gap": SlotSpec("slot:min_hours", "h", sort="DURATION")},
        family="hours",
    ),
    _sig(
        "constraint:min_staff_per_shift",
        "EACH(s in shift_instances): COUNT(w in workers WHERE ASSIGNED_TO(w, s)) >= demand(s)",
        deferred_primitive="value_primitive:demand_file",
    ),
    _sig(
        "constraint:forbidden_patterns",
        "EACH(w in workers): NOT MATCHES(shift_seq(w), forbidden_pattern(w))",
        deferred_primitive="value_primitive:patterns_file",
    ),
    _sig(
        "constraint:preference_satisfaction",
        "EACH(a in assignments): preferred(a)",
        deferred_primitive="value_primitive:availability_file",
    ),
    _sig(
        "constraint:worker_type_restrictions",
        "EACH(w in workers), EACH(s in shift_instances WHERE NOT eligible(w, s)): NOT ASSIGNED_TO(w, s)",
        deferred_primitive="value_primitive:worker_types_file",
    ),
)

# Unrealized configurations of declared rule families, and family readings with no registry label:
# expressible in the IR, but no registry ref covers them.
NURSE_EXTRAS: tuple[Extra, ...] = (
    Extra(
        "max_gap",
        "EACH(w in workers), EACH(g in adjacent(assignment_seq(w))): gap_duration(g) <= $max_gap",
        "TEMPORAL_GAP max-gap path: an unrealized configuration of the declared family",
        terminal="realization_absent",
        related_ref="constraint:min_rest_hours",
    ),
    Extra(
        "prefer_pattern",
        "EACH(w in workers): MATCHES(shift_seq(w), preferred_pattern(w))",
        "PATTERN effect=REWARD: an unrealized configuration of the declared family",
        terminal="realization_absent",
        related_ref="constraint:forbidden_patterns",
    ),
    Extra(
        "min_weekly_hours",
        "EACH(w in workers), EACH(win in windows(WEEK, 1)): SUM(a in assignments(w) WHERE DURING(interval(a), win): work_hours(a)) >= $min_hours",
        "COVERAGE minimum, DURATION: no registry ref",
    ),
    Extra(
        "max_staff_per_shift",
        "EACH(s in shift_instances): COUNT(w in workers WHERE ASSIGNED_TO(w, s)) <= $max_staff",
        "COVERAGE maximum per shift: no registry ref",
    ),
)

NURSE_SPEC = DomainSpec(
    vocab=NURSE_VOCABULARY,
    signatures=NURSE_SIGNATURES,
    population_slot_default={"op": "pop", "ref": "entity:worker"},
    extras=NURSE_EXTRAS,
    # a loose rule name: "the hour rule" or "the fatigue rule" can mean either of two constraints
    # (max_hours_per_week, min_rest_hours)
    rule_families={"hours": "a rule limiting working or resting hours (the fatigue rule)"},
    # plain readings: "근무 공평하게" (fair shifts: declared as the range of shift counts),
    # "초과근무 줄여" (less overtime: declared as the sum)
    plain_readings={
        "measure:workload": ("objective:workload_balance",),
        "measure:overtime_hours": ("objective:overtime",),
    },
)
