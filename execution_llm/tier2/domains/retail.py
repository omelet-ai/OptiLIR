"""The retail domain: vocabulary and capability forms, keyed by registry refs. The four objectives are
performance measures over the planning horizon. The four constraints have no value the user sets
(their bounds are per-item data fields), so their IR bounds an accessor and they ground by form alone.
Revenue, total cost, turnover, storage capacity and budget can be written in IR, but no capability
covers them: they ground to unsupported (realization_absent) when the registry declares their refs,
and raise GroundingError otherwise. `maximize_service_level` is left out on purpose.
"""

from __future__ import annotations

from execution_llm.tier2.expr import Accessor, Measure, Vocabulary
from execution_llm.tier2.surface import DomainSpec, Extra, Signature, SlotSpec, signature

_M = Measure
RETAIL_MEASURES: dict[str, Measure] = {
    "profit": _M("measure:profit", "COST", "horizon", "profit (sales revenue minus cost of goods and discards)"),
    "in_stock_rate": _M("measure:in_stock_rate", "FRACTION", "horizon", "share of days the item is in stock"),
    "waste_rate": _M("measure:waste_rate", "FRACTION", "horizon", "share of stock that is discarded"),
    "sales_revenue": _M("measure:sales_revenue", "COST", "horizon", "sales revenue"),
    "total_cost": _M("measure:total_cost", "COST", "horizon", "total cost (holding, ordering, waste)"),
    "turnover": _M("measure:turnover", "SCALAR", "horizon", "inventory turnover"),
    "holding_cost": _M("measure:holding_cost", "COST", "day", "holding cost"),
    "inventory_level": _M("measure:inventory_level", "QUANTITY", "day", "inventory level", keyed=True),
    "order_cost": _M("measure:order_cost", "COST", "order", "order spend"),
    "quantity": _M("measure:quantity", "QUANTITY", "order", "order quantity"),
    "arrival_day": _M("measure:arrival_day", "TIME", "order", "arrival day"),
    "order_day": _M("measure:order_day", "TIME", "order", "order day"),
    "expiry_day": _M("measure:expiry_day", "TIME", "unit", "expiry day"),
}
_A = Accessor
RETAIL_ACCESSORS: dict[str, Accessor] = {
    "day_of_week": _A("attr:day_of_week", "KEY", "the order's weekday"),
    "allowed_order_days": _A("attr:allowed_order_days", "KEY", "the item's allowed order weekdays"),
    "min_order_quantity": _A("attr:min_order_quantity", "QUANTITY", "the item's minimum order quantity"),
    "lead_time": _A("attr:lead_time", "DURATION", "the item's supplier lead time"),
    "product_life": _A("attr:product_life", "DURATION", "the item's shelf life"),
    "in_stock": _A("attr:in_stock", "BOOLEAN", "in stock that day"),
    "item": _A("attr:item", "ENTITY", "the order's item"),
}

RETAIL_VOCABULARY = Vocabulary(
    domain="retail",
    entity_sets={"items": "entity:item", "orders": "entity:order", "days": "entity:day", "stores": "entity:store"},
    population_aliases={"products": "items", "skus": "items"},
    containments={"orders": "entity:order", "units": "entity:unit"},
    sequences={},
    relations={},
    measures=RETAIL_MEASURES,
    accessors=RETAIL_ACCESSORS,
    # the retail GradingSignature reads unit:day as 86400 s (seconds are the DURATION base); the vocabulary follows it
    units={"day": ("unit:s", 86400), "days": ("unit:s", 86400), "units": ("unit:units", 1)},
    population_gloss={
        "key:measure:inventory_level": "of each item",
        "slot": "selected item",
        "entity:item": "item",
        "entity:order": "replenishment order",
        "entity:day": "day",
        "entity:unit": "stock unit",
        "entity:store": "store",
        "days": "planning day",
        "scenarios": "demand scenario",
    },
)


def _sig(ref: str, text: str, slots: dict[str, SlotSpec] | None = None, **kw) -> Signature:
    return signature(RETAIL_VOCABULARY, ref, text, slots, **kw)


RETAIL_SIGNATURES: tuple[Signature, ...] = (
    _sig("objective:maximize_profit", "MAXIMIZE( profit )"),
    _sig(
        "objective:minimize_stockout",
        "MAXIMIZE( in_stock_rate )",
    ),
    _sig("objective:minimize_holding_cost", "MINIMIZE( MEAN(d in days(horizon): holding_cost(d)) )"),
    _sig("objective:minimize_waste", "MINIMIZE( waste_rate )"),
    _sig("constraint:order_days", "EACH(o in orders): day_of_week(o) IN allowed_order_days(o)", family="ordering"),
    _sig("constraint:minimum_order_quantity", "EACH(o in orders): quantity(o) MULTIPLE_OF min_order_quantity(o)"),
    _sig(
        "constraint:lead_time",
        "EACH(o in orders): arrival_day(o) EQ offset(order_day(o), lead_time(o))",
        family="ordering",
    ),
    _sig(
        "constraint:shelf_life",
        "EACH(i in items), EACH(u in units(i)): expiry_day(u) EQ offset(arrival_day(u), product_life(i))",
    ),
)

_RA = "realization_absent"
RETAIL_EXTRAS: tuple[Extra, ...] = (
    # Recognized but not realized: each grounds to realization_absent when the registry declares its ref;
    # otherwise grounding raises GroundingError.
    Extra("maximize_revenue", "MAXIMIZE( sales_revenue )", "recognized", _RA, "objective:maximize_revenue"),
    Extra("minimize_total_cost", "MINIMIZE( total_cost )", "recognized", _RA, "objective:minimize_total_cost"),
    Extra("maximize_turnover", "MAXIMIZE( turnover )", "recognized", _RA, "objective:maximize_turnover"),
    Extra(
        "storage_capacity",
        "EACH(d in days(horizon)): SUM(i in items: inventory_level[i](d)) <= $max_units",
        "recognized",
        _RA,
        "constraint:storage_capacity",
    ),
    Extra(
        "budget_limit",
        "SUM(o in orders: order_cost(o)) <= 5000000",
        "recognized",
        _RA,
        "constraint:budget_limit",
    ),
)

RETAIL_SPEC = DomainSpec(
    vocab=RETAIL_VOCABULARY,
    signatures=RETAIL_SIGNATURES,
    population_slot_default={"op": "pop", "ref": "entity:item"},
    extras=RETAIL_EXTRAS,
    # a loose rule name: "the ordering rules" can mean either of two supplier-order constraints
    # (order_days, lead_time)
    rule_families={"ordering": "a rule on when or how supplier orders are placed (order days, lead time)"},
    # plain reading: "보관비 낮춰" (lower holding cost) is declared as the daily mean over the horizon; the sum
    # is not declared
    plain_readings={"measure:holding_cost": ("objective:minimize_holding_cost",)},
)
