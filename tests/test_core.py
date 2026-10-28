"""Offline tests: the published prompts and grammars, IR round-trips on rows of the released dataset, and
the grounding contracts over an independently authored, deliberately tiny registry."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from execution_llm.schema.decision import ExecuteBody
from execution_llm.schema.projections import (
    BindingManifestProjection,
    DataCatalogProjection,
    SemanticConstraint,
    SemanticManifestProjection,
    SemanticObjective,
    SlotDeclaration,
    UnitDefinition,
    _semantic_ref,
)
from execution_llm.schema.registry import Registry, RegistryEntry, domain_projection, load_registry
from execution_llm.tier2.domains import SURFACES
from execution_llm.tier2.emit import NO_INTENT, emit_problem, ir_digest, parse_problem
from execution_llm.tier2.grammar import ebnf
from execution_llm.tier2.prompt import system_prompt

ROOT = Path(__file__).resolve().parents[1]

# SHA-256 of prompts/<domain>.txt and grammar/<domain>.ebnf as published with the dataset. The adapters
# expect these prompts byte for byte at inference (training used a seeded per-row shuffle of the same
# lines, see the next test); a change here means a different model input.
PUBLISHED = {
    "routing": (
        "d0e05a63aaf448d55e168f6e76c43c717e2b94b821bf6263631a4acbc410085f",
        "f1b510377624a63997b6ac8f446348735a3615eb5cc012c9eb94b77ddb57161d",
    ),
    "nurse": (
        "96a9fb26233a0e234be00281b3a55194e7c9c36346b891bc226044d6db316bac",
        "d3bf570cb32315212330252f6c2f77ee1e6f630311237c5b09ddc5e3d1a4876b",
    ),
    "retail": (
        "db5127ba8db344d54e105bad3787e3def4157b4bf9bd10f8801ef62c679e873b",
        "a399c3130e1c6d55758872a93cbc35a5272fd7a327bcb7738229c439431ea36c",
    ),
}

# Gold IRs from the released dataset that together cover the line format: COMPOSE, UNSPECIFIED,
# UNEXPRESSED, UNDECIDED, the HOLE forms, the [HARD] and [SOFT] marks, NO_INTENT, CLARIFY(intent),
# windows and RUN_LENGTH.
RELEASE_IRS = [
    ("routing", "MINIMIZE( SUM(x0 in routes: duration(x0)) )\nMINIMIZE( SUM(x1 in routes: fuel_cost(x1)) )\nCOMPOSE(product)"),
    ("nurse", "MINIMIZE( UNSPECIFIED(x0 in workers: overtime_hours(x0)) )"),
    ("routing", "MINIMIZE( SUM(x0 in routes: duration(x0)) )\nUNEXPRESSED(literal, constraint)"),
    (
        "routing",
        "MINIMIZE( SUM(x0 in routes: duration(x0)) )\n[SOFT] EACH(x1 in routes): delivery_count(x1) <= UNDECIDED",
    ),
    ("routing", "MINIMIZE( SUM(x0 in routes: fuel_cost(x0)) )\nEACH(x1 in HOLE(scope)): delivery_count(x1) <= 15"),
    ("nurse", "MAXIMIZE( SUM(x0 in assignments: preference_score(x0)) )\n[HARD] HOLE(constraint, hours)"),
    ("routing", "NO_INTENT"),
    ("retail", "CLARIFY(intent)"),
    (
        "routing",
        "MINIMIZE( SUM(x0 in routes: fixed_cost(x0)) )\nEACH(x1 in visits): departure_time(x1) <= tw_late(x1)\n"
        "[HARD] EACH(x2 in routes): travel_distance(x2) <= 8 km\n"
        "[SOFT] EACH(x3 in routes): round_count(x3) >= base_rounds(x3)",
    ),
    ("routing", "MINIMIZE( SUM(x0 in routes: HOLE(measure, cost, x0)) )"),
    ("routing", "MINIMIZE( SUM(x0 in routes: fixed_cost(x0)) )\nEACH(x1 in routes): duration(x1) <= HOLE(value)"),
    ("retail", "MAXIMIZE( profit )\nMAXIMIZE( in_stock_rate )\nCOMPOSE(weighted_sum: 1=2/3, 2=1/3)"),
    (
        "nurse",
        "[SOFT] EACH(x0 in workers): EACH(x1 in windows(WEEK, 1)): SUM(x2 in assignments(x0) WHERE "
        "DURING(interval(x2), x1): work_hours(x2)) <= 44 h\nMINIMIZE( SUM(x3 in workers: overtime_hours(x3)) )",
    ),
    (
        "nurse",
        "MINIMIZE( SUM(x0 in shift_instances: unmet_demand(x0)) )\nMAXIMIZE( SUM(x1 in assignments: preference_score(x1)) )\n"
        "[SOFT] EACH(x2 in workers): RUN_LENGTH(x3 in day_seq(x2): works(x2, x3)) <= 4 day",
    ),
]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.mark.parametrize("domain", sorted(PUBLISHED))
def test_prompt_and_grammar_match_the_published_files(domain):
    surface = SURFACES[domain]
    prompt, grammar = PUBLISHED[domain]
    assert _sha(system_prompt(domain, forms=True)) == prompt
    assert _sha(ebnf(surface.vocab, surface.spec.signatures, surface.spec.plain_readings)) == grammar


def test_training_prompt_is_a_seeded_shuffle_of_the_published_prompt():
    published = system_prompt("routing", forms=True)
    shuffled = system_prompt("routing", seed="v4/combo:routing:258190c210a57871", forms=True)
    assert shuffled == system_prompt("routing", seed="v4/combo:routing:258190c210a57871", forms=True)
    assert shuffled != published and sorted(shuffled.splitlines()) == sorted(published.splitlines())
    # the training input of that train row, byte for byte
    assert _sha(shuffled) == "aaa58534e20ba0f1994a518bed535bb65fb727dbde7ccc1875852e805fe21687"


@pytest.mark.parametrize(("domain", "ir"), RELEASE_IRS)
def test_release_irs_parse_and_reemit_canonically(domain, ir):
    vocab = SURFACES[domain].vocab
    parsed = parse_problem(ir, vocab)
    text = NO_INTENT if parsed.no_intent else emit_problem(
        list(parsed.frames), vocab, parsed.composition, clarify_intent=parsed.clarify_intent, residuals=parsed.residuals
    )
    assert text == ir
    assert ir_digest(parse_problem(text, vocab), vocab) == ir_digest(parsed, vocab)


# Malformed or truncated outputs, each tripping a different failure inside the parser.
MALFORMED = [
    ("routing", "MINIMIZE( SUM(x0 in routes: no_such_measure(x0)) )"),
    ("routing", "MINIMIZE( SUM(x0 in"),
    ("routing", "[HARD]"),
    ("routing", "[HARD] EACH(x0 in visits): TOGETHER(x0, x0)"),
    ("routing", "[HARD] EACH(x0 in pairs(visits, adjacent)): travel_distance(x0) <= 5 km"),
    (
        "nurse",
        "[SOFT] EACH(x0 in workers): EACH(x1 in windows(WEEK, abc)): SUM(x2 in assignments(x0) WHERE "
        "DURING(interval(x2), x1): work_hours(x2)) <= 44 h",
    ),
]


@pytest.mark.parametrize(("domain", "ir"), MALFORMED)
def test_unparsable_output_is_a_value_error(domain, ir):
    with pytest.raises(ValueError):
        parse_problem(ir, SURFACES[domain].vocab)


def test_evaluation_scores_malformed_outputs_per_row(monkeypatch):
    import importlib.util

    monkeypatch.delenv("EXECUTION_LLM_REGISTRY", raising=False)
    spec = importlib.util.spec_from_file_location("evaluate", ROOT / "scripts" / "evaluate.py")
    evaluate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluate)
    domain, ir = RELEASE_IRS[0]
    row = {"id": "t", "domain": domain, "decision_kind": "execute", "stratum": "t", "language": "en", "ir": ir}
    results = evaluate.score([row] * 3, ["MINIMIZE( SUM(x0 in", "[HARD]", ir])
    assert [r["outcome"] for r in results] == ["parse_error", "parse_error", None]
    assert [r["ir_match"] for r in results] == [False, False, True]


def test_composition_weight_that_is_no_fraction_is_a_grounding_error():
    from execution_llm.grounding.grounder import GroundingError, _stated_weights
    from execution_llm.tier2.emit import Composition

    with pytest.raises(GroundingError):
        _stated_weights(["objective:a", "objective:b"], Composition("weighted_sum", weights={1: "1/0"}))


def test_parser_runs_without_site_packages():
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from execution_llm.tier2.emit import parse_problem; "
        "from execution_llm.tier2.domains.routing import ROUTING_VOCABULARY; "
        "assert parse_problem('CLARIFY(intent)', ROUTING_VOCABULARY).clarify_intent"
    )
    subprocess.run([sys.executable, "-I", "-S", "-c", code, str(ROOT)], check=True, timeout=20)


def test_every_module_imports_without_a_registry_or_model_libraries(monkeypatch):
    monkeypatch.delenv("EXECUTION_LLM_REGISTRY", raising=False)
    code = (
        "import importlib, pkgutil, sys; sys.path.insert(0, sys.argv[1]); import execution_llm; "
        "names = [m.name for m in pkgutil.walk_packages(execution_llm.__path__, 'execution_llm.')]; "
        "[importlib.import_module(name) for name in names]; assert len(names) > 20, names; "
        "assert not {'torch', 'transformers', 'anthropic', 'openai'} & set(sys.modules)"
    )
    subprocess.run([sys.executable, "-I", "-c", code, str(ROOT)], check=True, timeout=60)


# What the tiny registry declares beyond its server ids. Nurse declares one unit-bearing slot with a
# default, so the tests can see grading take the default from the registry.
_DECLARED = {
    "routing": {"objectives": (SemanticObjective(ref="objective:distance_cost", direction="minimize"),)},
    "nurse": {
        "objectives": (SemanticObjective(ref="objective:overtime", direction="minimize"),),
        "constraints": (
            SemanticConstraint(
                ref="constraint:max_hours_per_week",
                predicate_kind="comparison",
                default_hardness="soft",
                argument_slot_refs=("slot:max_hours",),
            ),
        ),
        "slots": (
            SlotDeclaration(ref="slot:max_hours", value_type="duration", unit_ref="unit:hour", optional=True, default="45"),
        ),
        "units": (UnitDefinition(ref="unit:hour", dimension="DURATION"),),
    },
    "retail": {},
}


@pytest.fixture
def registry_path(tmp_path, monkeypatch):
    entries = {}
    for domain, server in (
        ("routing", "advanced-routing"),
        ("nurse", "nurse-scheduling"),
        ("retail", "retail-optimization"),
    ):
        semantic = SemanticManifestProjection(
            server_id=server,
            semantic_version="authored-test-1",
            composition_methods=("weighted_sum",),
            max_weighted_terms=1,
            **_DECLARED[domain],
        )
        _, ref = _semantic_ref(semantic)
        entries[domain] = RegistryEntry(
            semantic=semantic,
            catalog=DataCatalogProjection(semantic_manifest_ref=ref, dataset_ref=f"authored-{domain}", version="1"),
            binding=BindingManifestProjection(
                semantic_manifest_ref=ref,
                server_id=server,
                binding_version="authored-test-1",
                admissible=False,
                rejection_reason="test fixture has no solver binding",
            ),
        )
    path = tmp_path / "registry.json"
    path.write_text(Registry(domains=entries).model_dump_json(), encoding="utf-8")
    monkeypatch.setenv("EXECUTION_LLM_REGISTRY", str(path))
    return path


def test_registry_is_explicit_and_exact(registry_path, monkeypatch):
    semantic, catalog, binding = domain_projection("routing")
    assert catalog is binding is None
    _, ref = _semantic_ref(semantic)
    _, projected_catalog, _ = domain_projection("routing", ref)
    assert projected_catalog is not None and projected_catalog.semantic_manifest_ref == ref
    with pytest.raises(ValueError, match="requested semantic"):
        domain_projection("routing", ref.model_copy(update={"digest": "0" * 64}))
    monkeypatch.delenv("EXECUTION_LLM_REGISTRY")
    with pytest.raises(ValueError, match="EXECUTION_LLM_REGISTRY"):
        load_registry()
    assert load_registry(registry_path).domains["routing"].semantic == semantic
    with pytest.raises(FileNotFoundError):
        load_registry(registry_path.parent / "missing.json")


@pytest.mark.parametrize(
    "fault", ["digest", "unknown_domain", "wrong_domain", "extra", "unit", "missing_domain", "malformed"]
)
def test_registry_rejects_bad_data(registry_path, fault):
    data = json.loads(registry_path.read_text())
    entry = data["domains"]["routing"]
    if fault == "digest":
        entry["catalog"]["semantic_manifest_ref"]["digest"] = "0" * 64
    elif fault == "unknown_domain":
        data["domains"]["unknown"] = entry
    elif fault == "wrong_domain":
        data["domains"]["nurse"] = entry
    elif fault == "extra":
        entry["run_python"] = "must never be interpreted"
    elif fault == "unit":
        entry["semantic"]["slots"] = [{"ref": "slot:test", "value_type": "duration", "unit_ref": "unit:undeclared"}]
    elif fault == "missing_domain":
        del data["domains"]["routing"]
    registry_path.write_text("{" if fault == "malformed" else json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        domain_projection("routing")


def test_grounding_contracts_use_only_the_supplied_registry(registry_path):
    from execution_llm.grounding.decide import decide, domain_inputs, rank_only, user_text
    from execution_llm.grounding.grounder import ground_ir
    from execution_llm.grounding.inputs import sampler_inputs

    inputs = sampler_inputs("routing")
    surface = SURFACES["routing"]
    assert ground_ir("CLARIFY(intent)", inputs, surface).kind == "clarify"
    assert ground_ir("NO_INTENT", inputs, surface).kind == "unsupported"
    body = ground_ir("MINIMIZE( SUM(r in routes: travel_distance(r)) )", inputs, surface)
    assert isinstance(body, ExecuteBody)
    assert body.objectives.terms[0].objective_ref == "objective:distance_cost"
    # An explicit unsupported aggregation must not be replaced with the supported total.
    assert ground_ir("MINIMIZE( MEAN(r in routes: travel_distance(r)) )", inputs, surface).kind == "unsupported"

    domain_inputs.cache_clear()
    result = decide("MINIMIZE( SUM(x0 in routes: travel_distance(x0)) )", "routing", "총 이동거리를 줄여줘")
    assert result["decision"]["kind"] == "execute" and result["fleet"] is None
    assert decide("CLARIFY(intent)", "routing")["decision"]["kind"] == "clarify"
    with pytest.raises(ValueError):
        decide("not an IR", "routing")
    assert user_text([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}]) == "a c"
    assert rank_only("거리 먼저, 그다음 시간") and not rank_only("거리 먼저, 시간은 절대 맞바꾸지 마")


def test_a_constraint_the_registry_does_not_declare_is_a_grounding_error(registry_path):
    from execution_llm.grounding.decide import decide, domain_inputs
    from execution_llm.grounding.grounder import GroundingError

    domain_inputs.cache_clear()
    ir = "MINIMIZE( SUM(x0 in routes: travel_distance(x0)) )\nEACH(x1 in routes): sec_radius(x1) <= individual_radius_limit(x1)"
    with pytest.raises(GroundingError, match="does not declare constraint:individual_enclosing_circle_radius"):
        decide(ir, "routing", "x")


def test_grading_fills_in_the_default_the_registry_declares(registry_path):
    from execution_llm.grading import _canonicalize
    from execution_llm.grounding.inputs import sampler_inputs
    from execution_llm.schema.decision import (
        ConstraintSelection,
        DataBinding,
        SoftStrength,
        WeightedSum,
        WeightedTerm,
    )

    inputs = sampler_inputs("nurse")
    body = ExecuteBody(
        kind="execute",
        objectives=WeightedSum(method="weighted_sum", terms=[WeightedTerm(objective_ref="objective:overtime", weight="1/1")]),
        constraints=[
            ConstraintSelection(
                constraint_ref="constraint:max_hours_per_week",
                strength=SoftStrength(kind="soft", weight="1"),
                arguments=[],
                scope=None,
            )
        ],
        parameters=[],
        data_binding=DataBinding(selections=[], overrides=[]),
    )
    [argument] = _canonicalize(body, inputs.snapshot, inputs.signature).constraints[0].arguments
    assert argument.slot_ref == "slot:max_hours"
    assert argument.value.model_dump(mode="json") == {"type": "duration", "value": "45", "unit_ref": "unit:hour"}


def test_an_objective_the_registry_does_not_declare_is_a_grounding_error(registry_path):
    from execution_llm.grounding.decide import decide, domain_inputs
    from execution_llm.grounding.grounder import GroundingError

    domain_inputs.cache_clear()
    with pytest.raises(GroundingError, match="does not declare objective:fuel_cost"):
        decide("MINIMIZE( SUM(x0 in routes: fuel_cost(x0)) )", "routing", "x")


def test_completeness_checks_the_parameters_a_signature_declares():
    from execution_llm.grading import GradingSignature, _check_required_completeness
    from execution_llm.schema.decision import (
        DataBinding,
        SnapshotConstraints,
        ValuePolicy,
        WeightedSum,
        WeightedTerm,
    )

    snapshot = SnapshotConstraints(
        objective_refs=("objective:distance_cost",),
        parameter_refs=("parameter:horizon",),
        parameter_value_policies={"parameter:horizon": ValuePolicy(value_type="integer")},
        catalog_ids_by_entity={},
    )
    signature = GradingSignature(entry_parameters={"objective:distance_cost": {"parameter:horizon"}})
    body = ExecuteBody(
        kind="execute",
        objectives=WeightedSum(
            method="weighted_sum", terms=[WeightedTerm(objective_ref="objective:distance_cost", weight="1/1")]
        ),
        constraints=[],
        parameters=[],
        data_binding=DataBinding(selections=[], overrides=[]),
    )
    with pytest.raises(ValueError, match="required parameter parameter:horizon is not supplied"):
        _check_required_completeness(body, snapshot, signature)


def _evaluate_script():
    import importlib.util

    spec = importlib.util.spec_from_file_location("evaluate", ROOT / "scripts" / "evaluate.py")
    evaluate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluate)
    return evaluate


def test_evaluation_scores_a_gold_the_registry_rejects_per_row(registry_path):
    from execution_llm.grounding.decide import domain_inputs

    domain_inputs.cache_clear()
    ir = "MINIMIZE( SUM(x0 in routes: travel_distance(x0)) )"
    terms = [{"objective_ref": "objective:fuel_cost", "weight": "1/1"}]  # fuel_cost: not in the tiny registry
    gold = {
        "kind": "execute",
        "objectives": {"method": "weighted_sum", "terms": terms},
        "constraints": [],
        "parameters": [],
        "data_binding": {"selections": [], "overrides": []},
    }
    row = {"id": "t", "domain": "routing", "decision_kind": "execute", "stratum": "t", "language": "en"}
    row |= {"ir": ir, "messages": [{"role": "user", "content": "x"}], "gold_decision": gold}
    [result] = _evaluate_script().score([row], [ir])
    assert result["outcome"] == "gold_rejected" and result["ir_match"]


def test_evaluation_stops_before_generation_when_the_registry_does_not_load(tmp_path, monkeypatch):
    from execution_llm.grounding.decide import domain_inputs

    domain_inputs.cache_clear()
    monkeypatch.setenv("EXECUTION_LLM_REGISTRY", str(tmp_path / "missing.json"))
    with pytest.raises(SystemExit, match="EXECUTION_LLM_REGISTRY"):
        _evaluate_script().check_registry([{"domain": "routing"}])
