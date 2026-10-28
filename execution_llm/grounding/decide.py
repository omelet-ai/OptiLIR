"""IR text -> decision, by the rule that made the dataset's `gold_decision` values.

The rule: parse the IR, ground it with `ground_full` (a routing lexicographic line takes the 10:1 rank
weights when the user turns are given and state no explicit "no trade-off" order), canonicalize the decision
and serialize it as JCS. Grounding reads an operator-supplied capability registry; set
`EXECUTION_LLM_REGISTRY` to its JSON path (schema: `execution_llm.schema.registry.Registry`).
"""

from __future__ import annotations

import dataclasses
import json
import re
from functools import cache

from execution_llm.grading import _canonicalize
from execution_llm.grounding.catalogs import nurse_fixture, retail_fixture
from execution_llm.grounding.grounder import ground_full
from execution_llm.grounding.inputs import SamplerInputs, sampler_inputs
from execution_llm.schema.decision import normalized_jcs_bytes
from execution_llm.tier2.domains import SURFACES
from execution_llm.tier2.emit import parse_problem

# An explicit "no trade-off" order (strict lexicographic). Without such a cue a routing lexicographic
# line reads as a rank-only priority and executes as a 10:1 weighted sum.
STRICT_ORDER = re.compile(
    r"섞|맞바꾸|바꾸지\s*않|유지한\s*(?:채|뒤|다음)|그대로\s*(?:둔|일\s*때)|확정|완전히|절대|엄격|같은\s*해"
    r"|안에서만|범위에서만|끝난?\s*(?:다음|이후|뒤)|끝나야|trade-?\s*offs?|tie-?\s*breakers?|only\s+among"
    r"|without\s+giving\s+up|no\s+compromises?|only\s+(?:after|once)|\block(?:ed|s)?\b|strictly|breaks?\s+ties"
    r"|never\s+swap",
    re.I,
)


def rank_only(text: str) -> bool:
    """The user's text states no explicit "no trade-off" order."""
    return STRICT_ORDER.search(text) is None


def user_text(messages: list[dict]) -> str:
    """The user turns of a conversation joined by spaces, as the rule behind `gold_decision` reads them."""
    return " ".join(m["content"] for m in messages if m["role"] == "user")


@cache
def domain_inputs(domain: str) -> SamplerInputs:
    """The grounding inputs of one domain, built from the supplied registry, with the authored nurse and
    retail catalog fixtures."""
    catalog = {"nurse": nurse_fixture, "retail": retail_fixture}.get(domain)
    return sampler_inputs(domain, corpus_catalog=catalog() if catalog else None)  # type: ignore[arg-type]


def decide(ir: str, domain: str, text: str | None = None) -> dict:
    """{"decision", "fleet", "notices"} for one IR. ``text``: the user turns (see `user_text`); without
    it a routing lexicographic line is never read as a rank-only priority and stays unsupported.

    ``fleet`` is a stated vehicle count that the decision cannot carry ({"count", "use_all"} or None): a
    consumer that executes the decision must give the solver only the first ``count`` vehicles, or not
    execute it. Raises ValueError when the IR does not parse, GroundingError (a ValueError) when it
    cannot be grounded."""
    surface, inputs = SURFACES[domain], domain_inputs(domain)
    rank_weights = text is not None and rank_only(text)
    grounding = ground_full(parse_problem(ir, surface.vocab), inputs, surface, rank_weights=rank_weights)
    body = _canonicalize(grounding.decision, inputs.snapshot, inputs.signature)
    return {
        "decision": json.loads(normalized_jcs_bytes(body)),
        "fleet": dataclasses.asdict(grounding.fleet) if grounding.fleet else None,
        "notices": [notice.code for notice in grounding.notices],
    }
