"""The parser model's prompt: the grammar cheat-sheet plus one domain's vocabulary and, with
`forms=True`, its capability forms; registry refs never appear in it. Shared by training,
evaluation and serving: a served adapter gets the published prompt, the same lines as in training,
where each row listed them in its own shuffled order (`seed=<row id>`).
"""

from __future__ import annotations

import hashlib
import random

from execution_llm.tier2.domains import SURFACES
from execution_llm.tier2.emit import emit_problem
from execution_llm.tier2.expr import Vocabulary

GRAMMAR = """You are the semantic parser of an optimization assistant. Read the conversation and write the user's optimization intent as Tier-2 IR: one frame per line, nothing else.

Grammar (populations are named sets; x is a bound variable, written x0, x1, ... in order of first binding across the frames; a measure is applied as name(x)):
  MINIMIZE( term ) / MAXIMIZE( term )              objective frame
  EACH(x in P): body                               constraint: body holds for every member of P; chains nest: EACH(x in P): EACH(y in Q(x)): body
  AGG(x in P: term) with AGG = SUM | MAX | MIN | MEAN | RANGE   aggregation over P
  COUNT(x in P) / COUNT_DISTINCT(x in P: key)      counts
  RUN_LENGTH(x in seq: pred)                        longest run of consecutive members satisfying pred
  P WHERE pred                                     filtered population; predicates: ASSIGNED_TO(a, b), DURING(interval, window), OVERLAPS(a, b), MATCHES(seq, pattern), NOT pred, a IN b
  comparisons: <=  <  >=  >  =  !=   ; values are written with their spoken unit (85 km, 8 h, 30 min, 3 day) or bare for counts and fractions (12, 0.8)
  offset(term, shift), RATIO(a, b), table_score($table, keys...), $name for a data table the user refers to
  positions(x), adjacent(seq), pairs(P, relation), keys(measure), windows(WEEK, 1), days(horizon)
Holes (what the user left unstated - never guess it):
  HOLE(measure, family, x)   the quantity is genuinely unresolved among a declared measure family (see [family: ...]). A single available capability does not resolve user ambiguity. Use a concrete measure when ordinary words or prior dialogue determine it.
  UNSPECIFIED(x in P: m(x))  in place of SUM/MAX/MIN/MEAN/RANGE when the user names the quantity but not how it is aggregated ("공평하게", "초과근무 줄여", "보관비 낮춰", "빨리 끝나게", "적재율 0.8 이상"); only over the plain-reading measures of the vocabulary. An aggregation the user states (total, average, max-min difference, the latest finish) is written explicitly.
  HOLE(value)                a bound is requested but its number is not given (asked back to the user)
  UNDECIDED                  in place of a bound's value when the user says the value is not decided yet ("상한은 두고 싶은데 값은 아직 미정")
  EACH(x in HOLE(scope)): body   a per-unit rule whose unit (per route, per round trip, per nurse...) is not stated
  HOLE(constraint, family)   the user asked for a rule only by a loose name (a "fatigue rule", the "ordering rules") that this vocabulary realizes as several different constraints; family is a rule family name from the vocabulary
Frame marks:
  [HARD] before a constraint the user makes mandatory (must, never, 반드시, 절대, 무조건); [SOFT] before one the user makes preferential (if possible, 가능하면, 되도록); no mark when the user does not say.
Composition: when two or more objectives are stated with weights or an ordering, add one line COMPOSE(weighted_sum: 1=3/5, 2=2/5) (indices = objective frames in order) or COMPOSE(lexicographic: 1 > 2) for "first A, then B, no trade-off"; a method the user names by another word ("multiply them") is written as COMPOSE(product); with no stated method, write no COMPOSE line.
No optimization intent (a status question, an export request, small talk): write exactly NO_INTENT.
Asking you to recommend a numeric bound is a recommendation task, not a missing-value optimization instruction. A standalone request for an unprovided recommendation service is NO_INTENT (out of scope), not HOLE(value). Do not invent a recommended number.
Optimization intent exists but the objective itself is not determined by the conversation (what to optimize is unclear, e.g. "라이더마다 고르게", "재고 비용 줄여"): write exactly CLARIFY(intent), with no other lines. When the objective is determined, keep every determined frame and write only the unresolved element as a hole or UNSPECIFIED. Do not select SUM, MAX or a nearby capability just because it is available: "빨리 끝나게 해줘" without resolving context is MINIMIZE( UNSPECIFIED(x0 in routes: duration(x0)) ); an explicit request to minimize the sum of all route durations is SUM. A previous clarification and the user's answer can determine the meaning.
A user who states only rules and no objective (the objective is chosen elsewhere): write only those constraint frames.
A stated requirement this grammar cannot express is one line UNEXPRESSED(class, side) in place of that frame: class condition (e.g. "비 오면"), relative (e.g. "지금보다 20% 줄여"), literal (a clock time or date, e.g. "오후 6시 전") or other (e.g. a scope such as "주말만"); side objective or constraint. Only for a clear meaning; never force it into a nearby form, and never drop it.
Equal optimal solutions do not make different meanings interchangeable: preserve totals versus averages and counts versus rates, even over a fixed horizon. If the conversation does not determine which meaning is intended, clarify. An explicit unsupported aggregate must stay explicit, not be replaced by a supported one.
Write only what the user meant; do not add defaults, do not add frames the user did not ask for, keep numbers exactly as stated (an approximate number such as "대략 10km" is written as stated). Do not omit: every requirement the user stated appears as a frame or an UNEXPRESSED line."""


def vocab_block(vocab: Vocabulary, rng: random.Random | None, *, forms: bool = False) -> str:
    """The vocabulary listing; ``rng`` shuffles it (training-time augmentation), ``forms`` appends the
    server's capability forms: each declared non-opaque signature written in IR notation."""

    def order(items):
        items = list(items)
        if rng is not None:
            rng.shuffle(items)
        return items

    lines = [f"Domain: {vocab.domain}", "Populations:"]
    gloss = vocab.population_gloss
    for name, ref in order(vocab.entity_sets.items()):
        lines.append(f"  {name}  ({gloss.get(ref, ref.split(':')[-1])})")
    for alias, target in vocab.population_aliases.items():
        lines.append(f"  {alias} = {target}")
    for name, ref in order(vocab.containments.items()):
        lines.append(f"  {name}(x)  members of x ({gloss.get(ref, ref.split(':')[-1])})")
    for name, ref in order(vocab.sequences.items()):
        lines.append(f"  {name}(x)  ordered sequence ({ref.split(':')[-1]})")
    for name, ref in order(vocab.relations.items()):
        lines.append(f"  pairs(P, {name})  pairs of P related by {ref.split(':')[-1]}")
    lines.append("Measures (name(owner): sort - meaning):")
    for name, m in order(vocab.measures.items()):
        fam = f" [family: {m.family}]" if m.family else ""
        keyed = "[key]" if m.keyed else ""
        lines.append(f"  {name}{keyed}({m.owner}): {m.sort} - {m.gloss}{fam}")
    lines.append("Accessors (name(x): sort - meaning):")
    for name, a in order(vocab.accessors.items()):
        lines.append(f"  {name}(x): {a.sort} - {a.gloss}")
    units = sorted({k for k in vocab.units})
    lines.append("Units: " + ", ".join(units))
    spec = SURFACES[vocab.domain].spec
    plain = [name for name, m in vocab.measures.items() if m.ref in spec.plain_readings]
    if plain:
        lines.append("Plain readings (UNSPECIFIED may aggregate only these measures): " + ", ".join(plain))
    for convention in spec.conventions:
        lines.append(convention)
    families = SURFACES[vocab.domain].spec.rule_families
    if families:
        lines.append(
            "Rule families (a loose name users give to several different constraints; HOLE(constraint, name)):"
        )
        for name, gloss in families.items():
            lines.append(f"  {name}: {gloss}")
    if forms:
        lines.append(
            "Capability forms (how this server writes each declared objective and constraint; $name is a value the user states, "
            "a population slot left unstated means the whole population. Use a form only when it exactly matches the "
            "meaning established by the conversation; never force a clear unsupported meaning into a nearby form. "
            "Use holes for unresolved elements, or CLARIFY(intent) if the objective meaning itself is unresolved):"
        )
        for sig in order(s for s in SURFACES[vocab.domain].spec.signatures if not s.opaque):
            lines.append("  " + emit_problem([(sig.tree(), None)], vocab))
    return "\n".join(lines)


def system_prompt(domain: str, *, seed: str | None = None, forms: bool = False) -> str:
    """``seed`` (an item id) shuffles the listing deterministically; None keeps the declared order."""
    vocab = SURFACES[domain].vocab
    rng = random.Random(int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16)) if seed is not None else None
    return GRAMMAR + "\n\n" + vocab_block(vocab, rng, forms=forms)
