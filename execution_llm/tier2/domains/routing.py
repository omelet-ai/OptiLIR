"""The routing domain: vocabulary and capability forms, keyed by registry refs.

This file is the per-domain authoring surface a new domain adds: names the
parser resolves, measures with their sorts, and one expression per registry entry with
its slot map (signature slot -> registry slot ref, the unit the registry's bare decimal
is read in). Nothing here is code a new domain changes; it is data.

Slot value units (`$limit DISTANCE m`, `$limit DURATION s`): the
registry decimal is the base-unit number. Penalty mechanics (`exp_factor`, weights) are
outside the grammar and have no form slot; the grounder writes their declared defaults into each
decision, so gold decisions carry them although users never state them.
"""

from __future__ import annotations

from execution_llm.tier2.expr import Accessor, Measure, Vocabulary
from execution_llm.tier2.surface import DomainSpec, Extra, Signature, SlotSpec, signature

_M = Measure
ROUTING_MEASURES: dict[str, Measure] = {
    "travel_distance": _M("measure:travel_distance", "DISTANCE", "route", "travel distance"),
    "duration": _M("measure:duration", "DURATION", "route", "elapsed time from start to finish"),
    "wait_duration": _M("measure:wait_duration", "DURATION", "visit", "waiting time before service"),
    "arrival_time": _M("measure:arrival_time", "TIME", "visit", "arrival time"),
    "departure_time": _M("measure:departure_time", "TIME", "visit", "departure time"),
    "end_time": _M("measure:end_time", "TIME", "route", "finishing time"),
    "load": _M("measure:load", "QUANTITY", "position", "carried load", keyed=True),
    "utilization": _M("measure:utilization", "FRACTION", "round", "capacity utilization ratio", keyed=True),
    "delivery_count": _M("measure:delivery_count", "COUNT", "route", "number of deliveries"),
    "stop_count": _M("measure:stop_count", "COUNT", "route", "number of distinct stops"),
    "round_count": _M("measure:round_count", "COUNT", "route", "number of depot round trips"),
    "sec_radius": _M("measure:sec_radius", "DISTANCE", "route", "radius of the smallest circle enclosing the stops"),
    "fuel_cost": _M("measure:fuel_cost", "COST", "route", "fuel cost", family="cost"),
    "fixed_cost": _M("measure:fixed_cost", "COST", "route", "fixed operating cost of using the vehicle", family="cost"),
    "priority_weighted_arrival": _M(
        "measure:priority_weighted_arrival", "SCALAR", "visit", "arrival time weighted by the delivery's time priority"
    ),
    "route_cost": _M("measure:route_cost", "COST", "route", "route cost (distance, duration and fixed components)"),
    "drop_penalty": _M("measure:drop_penalty", "COST", "visit", "penalty for leaving the delivery unserved"),
}
_A = Accessor
ROUTING_ACCESSORS: dict[str, Accessor] = {
    "rider": _A("attr:rider", "ENTITY", "the route's rider"),
    "first_visit": _A("attr:first_visit", "ENTITY", "the route's first delivery"),
    "round_index": _A("attr:round_index", "INTEGER", "the delivery's round index"),
    "zone": _A("attr:zone", "KEY", "the delivery's zone"),
    "time_priority": _A("attr:time_priority", "INTEGER", "the delivery's time priority"),
    "tw_late": _A("attr:tw_late", "TIME", "the delivery's latest allowed time"),
    "service_interval": _A("attr:service_interval", "INTERVAL", "the delivery's service interval"),
    "prohibited_window": _A("attr:prohibited_window", "INTERVAL", "the delivery's prohibited time window"),
    "required_skill": _A("attr:required_skill", "KEY", "the delivery's required skill"),
    "usage": _A("attr:usage", "OPAQUE", "the rider's usage figures"),
    # per-rider data fields read as bounds: the values are stored on each rider, not in registry slots
    "capacity": _A("attr:capacity", "QUANTITY", "the vehicle's capacity"),
    "individual_radius_limit": _A("attr:individual_radius_limit", "DISTANCE", "the rider's own radius limit"),
    "max_rounds": _A("attr:max_rounds", "COUNT", "the rider's own round-trip limit"),
    "max_stops_per_round": _A("attr:max_stops_per_round", "COUNT", "the rider's own per-round stop limit"),
    "max_stops": _A("attr:max_stops", "COUNT", "the rider's own stop limit"),
    "total_limit": _A("attr:total_limit", "DISTANCE", "the rider's own total distance limit"),
    "trip_limit": _A("attr:trip_limit", "DISTANCE", "the rider's own per-round distance limit"),
    "between_limit": _A("attr:between_limit", "DISTANCE", "the rider's own limit between consecutive stops"),
    "max_wait": _A("attr:max_wait", "DURATION", "the rider's own waiting-time limit"),
    "work_end": _A("attr:work_end", "TIME", "the rider's end of working hours"),
    "round_window_end": _A("attr:round_window_end", "TIME", "the rider's round window end"),
    "base_rounds": _A("attr:base_rounds", "COUNT", "the rider's contracted round trips"),
}

ROUTING_VOCABULARY = Vocabulary(
    domain="routing",
    entity_sets={"routes": "entity:route", "visits": "entity:visit", "assignments": "entity:assignment"},
    population_aliases={"riders": "routes"},  # riders and routes are one to one, so `riders` names the route set
    containments={"rounds": "entity:round", "visits": "entity:visit", "stops": "entity:stop"},
    sequences={"visit_seq": "attr:visit_seq", "stop_seq": "attr:stop_seq"},
    relations={"same_coordinate": "rel:same_coordinate", "nearby": "rel:nearby"},
    measures=ROUTING_MEASURES,
    accessors=ROUTING_ACCESSORS,
    units={
        "m": ("unit:m", 1),
        "km": ("unit:m", 1000),
        "s": ("unit:s", 1),
        "min": ("unit:s", 60),
        "h": ("unit:s", 3600),
    },
    population_gloss={
        "key:measure:load": "in each cargo dimension",
        "slot": "selected rider",
        "entity:route": "route",
        "entity:round": "round trip",
        "entity:visit": "delivery",
        "entity:assignment": "rider-delivery assignment",
        "positions": "point",
        "adjacent": "consecutive stop pair",
        "keys:measure:load": "cargo dimension",
    },
)


def _sig(ref: str, text: str, slots: dict[str, SlotSpec] | None = None, **kw) -> Signature:
    return signature(ROUTING_VOCABULARY, ref, text, slots, **kw)


ROUTING_SIGNATURES: tuple[Signature, ...] = (
    # objectives
    _sig("objective:distance_cost", "MINIMIZE( SUM(r in routes: travel_distance(r)) )"),
    _sig("objective:duration_cost", "MINIMIZE( SUM(r in routes: duration(r)) )"),
    _sig("objective:fixed_cost", "MINIMIZE( SUM(r in routes: fixed_cost(r)) )"),
    _sig("objective:fuel_cost", "MINIMIZE( SUM(r in routes: fuel_cost(r)) )"),
    _sig(
        "objective:first_visit_cost",
        "MINIMIZE( SUM(r in routes: table_score($first_visit_cost_table, rider(r), first_visit(r))) )",
        {"sig:first_visit_cost_table": SlotSpec("data:first_visit_cost.first_visit_costs", None, "table")},
        deferred_primitive="value_primitive:first_visit_costs",
    ),
    _sig(
        "objective:consecutive_visit_tiered_cost",
        "MINIMIZE( SUM(r in routes: SUM(a in adjacent(visit_seq(r)): OPAQUE(tier, travel_distance(a)))) )",
        opaque=True,  # the banded cost lookup is not a grammar operator, so it stays an opaque leaf
    ),
    # constraints
    _sig(
        "constraint:capacity",
        "EACH(r in routes), EACH(t in rounds(r)), EACH(d in keys(load)): MAX(p in positions(t): load[d](p)) <= capacity[d](r)",
    ),
    _sig(
        "constraint:delivery_length",
        "EACH(r in routes): delivery_count(r) <= $max_size",
        {"sig:max_size": SlotSpec("slot:delivery_length.max_size", None, sort="COUNT")},
    ),
    _sig("constraint:delivery_time_priority", "MINIMIZE( SUM(v in visits: priority_weighted_arrival(v)) )"),
    _sig(
        "constraint:enclosing_circle_radius",
        "EACH(r in routes): sec_radius(r) <= $max_radius",
        {"sig:max_radius": SlotSpec("slot:enclosing_circle_radius.max_radius", "m")},
    ),
    _sig(
        "constraint:individual_enclosing_circle_radius",
        "EACH(r in routes): sec_radius(r) <= individual_radius_limit(r)",
    ),
    _sig(
        "constraint:round_enclosing_circle_radius",
        "EACH(r in routes), EACH(t in rounds(r)): sec_radius(t) <= $max_radius",
        {"sig:max_radius": SlotSpec("slot:round_enclosing_circle_radius.max_radius", "m")},
    ),
    _sig(
        "constraint:max_distance",
        "EACH(r in routes): travel_distance(r) <= $limit",
        {"sig:limit": SlotSpec("slot:max_distance.max_distance", "m")},
    ),
    _sig(
        "constraint:max_duration",
        "EACH(r in routes): duration(r) <= $limit",
        {"sig:limit": SlotSpec("slot:max_duration.max_duration", "s")},
    ),
    _sig(
        "constraint:min_capacity",
        "EACH(r in $target_riders), EACH(t in rounds(r)): MAX(d in keys(load): utilization[d](t)) >= $target_ratio",
        {
            "sig:target_riders": SlotSpec("slot:min_capacity.target_riders", None, "population"),
            "sig:target_ratio": SlotSpec("slot:min_capacity.target_ratio", None, sort="FRACTION"),
        },
    ),
    _sig(
        "constraint:minimum_order",
        "EACH(r in $target_riders): delivery_count(r) >= $target_order",
        {
            "sig:target_riders": SlotSpec("slot:minimum_order.target_riders", None, "population"),
            "sig:target_order": SlotSpec("slot:minimum_order.target_order", None, sort="COUNT"),
        },
    ),
    _sig("constraint:minimum_round", "EACH(r in routes): round_count(r) >= base_rounds(r)"),
    _sig(
        "constraint:nearby_delivery_grouping",
        "EACH((a, b) in pairs(visits, nearby($distance_threshold))): TOGETHER(a, b, route)",
        {"sig:distance_threshold": SlotSpec("slot:nearby_delivery_grouping.distance_threshold", "m")},
    ),
    _sig(
        "constraint:same_coordinate_grouping",
        "EACH((a, b) in pairs(visits, same_coordinate)): TOGETHER(a, b, route)",
    ),
    _sig(
        "constraint:preference_region",
        "MINIMIZE( SUM((k, v) in assignments: table_score($preference_cost_table, k, v)) )",
        {"sig:preference_cost_table": SlotSpec("data:preference_region.preference_regions", None, "table")},
        deferred_primitive="value_primitive:preference_regions",
    ),
    _sig(
        "constraint:round_preference_region",
        "MINIMIZE( SUM((k, v) in assignments: table_score($round_preference_cost_table, k, v, round_index(v))) )",
        {"sig:round_preference_cost_table": SlotSpec("data:round_preference_region.preference_regions", None, "table")},
        deferred_primitive="value_primitive:preference_regions",
    ),
    _sig(
        "constraint:prohibited_time_window",
        "EACH(v in visits): NOT OVERLAPS(service_interval(v), prohibited_window(v))",
    ),
    _sig(
        "constraint:rider_delivery_length",
        "AND( EACH(r in routes): round_count(r) <= max_rounds(r),"
        " EACH(r in routes), EACH(t in rounds(r)): stop_count(t) <= max_stops_per_round(r),"
        " EACH(r in routes): stop_count(r) <= max_stops(r) )",
    ),
    _sig(
        "constraint:rider_distance",
        "AND( EACH(r in routes): travel_distance(r) <= total_limit(r),"
        " EACH(r in routes), EACH(t in rounds(r)): travel_distance(t) <= trip_limit(r),"
        " EACH(r in routes), EACH(a in adjacent(visit_seq(r))): travel_distance(a) <= between_limit(r) )",
    ),
    _sig(
        "constraint:rider_route_zone",
        "MINIMIZE( SUM(r in routes: SUM(t in rounds(r): COUNT_DISTINCT(v in visits(t): zone(v)))) )",
    ),
    _sig(
        "constraint:rider_time_window",
        "AND( EACH(r in routes), EACH(v in visits(r)): wait_duration(v) <= max_wait(r),"
        " EACH(r in routes): end_time(r) <= work_end(r),"
        " EACH(r in routes), EACH(t in rounds(r)): end_time(t) <= round_window_end(r) )",
    ),
    _sig("constraint:time_priority_order", "EACH(r in routes): ORDERED_BY(visit_seq(r), time_priority, DESC)"),
    _sig("constraint:time_window", "EACH(v in visits): departure_time(v) <= tw_late(v)"),
    _sig(
        "constraint:fixed_contract_vehicle",
        "MINIMIZE( SUM(k in $target_riders: OPAQUE(fixed_contract_pay, usage(k))) )",
        {"sig:target_riders": SlotSpec("slot:fixed_contract_vehicle.target_riders", None, "population")},
        opaque=True,
    ),
    _sig(
        "constraint:fixed_contract_incentive_vehicle",
        "MINIMIZE( SUM(k in $target_riders: OPAQUE(fixed_contract_incentive_pay, usage(k))) )",
        {"sig:target_riders": SlotSpec("slot:fixed_contract_incentive_vehicle.target_riders", None, "population")},
        opaque=True,
    ),
)

SIGNATURE_BY_REF: dict[str, Signature] = {s.ref: s for s in ROUTING_SIGNATURES}

# Meanings the grammar can express but no registry entry covers: min-max objectives over routes, a
# selective-service objective and a total waiting time objective that no entry declares.
ROUTING_EXTRAS: tuple[Extra, ...] = (
    # Recognized but not realized: minimize the largest per-route value, on several bases. It grounds to
    # realization_absent when the registry declares objective:minmax; otherwise grounding raises
    # GroundingError.
    Extra(
        "minmax_distance",
        "MINIMIZE( MAX(r in routes: travel_distance(r)) )",
        "min-max, distance",
        terminal="realization_absent",
        related_ref="objective:minmax",
    ),
    Extra(
        "minmax_duration",
        "MINIMIZE( MAX(r in routes: duration(r)) )",
        "min-max, duration",
        terminal="realization_absent",
        related_ref="objective:minmax",
    ),
    Extra(
        "minmax_route_cost",
        "MINIMIZE( MAX(r in routes: route_cost(r)) )",
        "min-max, route cost",
        terminal="realization_absent",
        related_ref="objective:minmax",
    ),
    Extra(
        "minmax_fixed_cost",
        "MINIMIZE( MAX(r in routes: fixed_cost(r)) )",
        "min-max, fixed-cost part of the route cost",
        terminal="realization_absent",
        related_ref="objective:minmax",
    ),
    Extra(
        "minmax_fuel_cost",
        "MINIMIZE( MAX(r in routes: fuel_cost(r)) )",
        "min-max, fuel cost (travel distance times a rate)",
        terminal="realization_absent",
        related_ref="objective:minmax",
    ),
    # Selective service (visits left unserved at a penalty): recognized but not realized, like the min-max
    # entries.
    Extra(
        "selective_service",
        "MINIMIZE( SUM(v in visits WHERE NOT ASSIGNED(v): drop_penalty(v)) )",
        "selective service, objective part",
        terminal="realization_absent",
        related_ref="constraint:selective_service",
    ),
    # outside the served vocabulary: no entry, recognized or otherwise
    Extra("total_wait", "MINIMIZE( SUM(v in visits: wait_duration(v)) )", "no declared accumulation of waiting time"),
)

ROUTING_SPEC = DomainSpec(
    vocab=ROUTING_VOCABULARY,
    signatures=ROUTING_SIGNATURES,
    population_slot_default={"op": "pop", "ref": "entity:route"},
    extras=ROUTING_EXTRAS,
    # "적재율 0.8 이상" (load ratio at least 0.8) leaves the per-dimension aggregation open and is clarified
    # with min_capacity as the candidate; "빨리 끝나게" (finish early), i.e. start-to-finish time, has no
    # single declared reading (minmax is not executable), so it is clarified with no candidate
    plain_readings={"measure:utilization": ("constraint:min_capacity",), "measure:duration": ()},
    # A stated vehicle count is a fleet directive (returned beside the decision as `fleet`); a
    # numberless "use all vehicles" means every route serves at least one delivery
    conventions=(
        'Vehicle counts: a stated number of vehicles is COUNT(x0 in routes) <= N (or = N, >= N for "use all N"); '
        '"차량 다 써" without a number is EACH(x0 in routes): delivery_count(x0) >= 1.',
    ),
)
