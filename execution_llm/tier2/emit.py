"""Round-trip-safe text form of the IR: what the parser model writes.

`emit` renders a normalized tree in the grammar notation `parse` accepts, so that
`canonical_bytes(normalize(parse(emit(t)))) == canonical_bytes(normalize(t))` for every tree
the corpus stores (units are spelled in the vocabulary's spoken units, holes as `HOLE(...)`,
an unstated EACH chain as `EACH(x in HOLE(scope))`). `emit_problem` / `parse_problem` write and
read the line format: one frame per line with an optional `[HARD]` / `[SOFT]` mark, an
optional `COMPOSE(...)` line for the objective composition (kept outside the expression grammar:
a composition is not an IR node), `UNEXPRESSED(class, side)` lines standing in for stated
requirements the grammar cannot express, `NO_INTENT` for an utterance with no optimization intent,
and `CLARIFY(intent)` when the conversation does not determine the objective.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from decimal import Decimal

from execution_llm.tier2.expr import Node, Vocabulary, canonical_bytes, normalize, parse, renumber_from

_CMP = {"le": "<=", "lt": "<", "ge": ">=", "gt": ">", "eq": "=", "ne": "!="}
_AGG = ("sum", "max", "min", "mean", "range", "unspecified")


def _names(vocab: Vocabulary) -> dict[str, str]:
    """ref -> notation name for every named inventory item (first declared name wins)."""
    out: dict[str, str] = {}
    for name, m in vocab.measures.items():
        out.setdefault(m.ref, name)
    for name, a in vocab.accessors.items():
        out.setdefault(a.ref, name)
    for table in (vocab.entity_sets, vocab.containments, vocab.sequences, vocab.relations):
        for name, ref in table.items():
            out.setdefault(ref, name)
    return out


def spoken_value(num: str, unit: str | None, vocab: Vocabulary) -> str:
    """A base-unit value in the largest declared spoken unit that divides it exactly."""
    if unit is None:
        return num
    if not unit.startswith("unit:"):
        return f"{num} {unit}"
    quantity = Decimal(num)
    best: tuple[int, str] | None = None
    for spoken, (base, factor) in vocab.units.items():
        if base != unit:
            continue
        if quantity % factor == 0 and (best is None or factor > best[0]):
            best = (factor, spoken)
    if best is None:
        raise ValueError(f"no spoken unit for {unit} in {vocab.domain}")
    scaled = quantity / best[0]
    text = str(scaled.quantize(Decimal(1))) if scaled == scaled.to_integral_value() else str(scaled.normalize())
    return f"{text} {best[1]}"


def emit(node: Node, vocab: Vocabulary) -> str:
    names = _names(vocab)

    def name_of(ref: str) -> str:
        return names.get(ref, ref.split(":")[-1])

    def binder(n: Node) -> str:
        return n["var"] if "var" in n else f"({', '.join(n['vars'])})"

    def go(n: Node) -> str:
        op = n.get("op")
        if op in ("minimize", "maximize"):
            return f"{op.upper()}( {go(n['term'])} )"
        if op == "each":
            return f"EACH({binder(n)} in {go(n['over'])}): {go(n['body'])}"
        if op == "scope_hole":
            return f"EACH({n['var']} in HOLE(scope)): {go(n['body'])}"
        if op == "frame_hole":
            return f"HOLE(constraint, {n['family']})"
        if op == "and":
            return "AND( " + ", ".join(go(i) for i in n["items"]) + " )"
        if op in _CMP:
            return f"{go(n['left'])} {_CMP[op]} {go(n['right'])}"
        if op in _AGG:
            return f"{op.upper()}({binder(n)} in {go(n['over'])}: {go(n['term'])})"
        if op == "count":
            return f"COUNT({binder(n)} in {go(n['over'])})"
        if op == "count_distinct":
            return f"COUNT_DISTINCT({binder(n)} in {go(n['over'])}: {go(n['key'])})"
        if op == "measure":
            key = f"[{n['key']['ref']}]" if n.get("key") else ""
            of = n["of"]
            if of.get("op") == "entity":  # a population-owned measure is written bare (e.g. retail profit)
                return f"{name_of(n['ref'])}{key}"
            return f"{name_of(n['ref'])}{key}({go(of)})"
        if op == "attr":
            key = f"[{n['key']['ref']}]" if n.get("key") else ""
            return f"{name_of(n['ref'])}{key}({', '.join(go(a) for a in n['args'])})"
        if op == "value":
            return spoken_value(n["num"], n.get("unit"), vocab)
        if op == "undecided":
            return "UNDECIDED"
        if op == "slot":
            return f"${n['ref'].split(':')[-1]}"
        if op == "hole":
            kind = n["kind"]
            if kind == "measure":
                family = n["family"]
                token = name_of(family) if family.startswith("measure:") else family
                if n["of"].get("op") == "entity":
                    return f"HOLE(measure, {token})"
                return f"HOLE(measure, {token}, {go(n['of'])})"
            return f"HOLE({kind})"
        if op == "var":
            return n["ref"]
        if op == "pop":
            return name_of(n["ref"])
        if op == "within":
            return f"{name_of(n['ref'])}({go(n['container'])})"
        if op == "positions":
            return f"positions({go(n['container'])})"
        if op == "adjacent":
            return f"adjacent({go(n['seq'])})"
        if op == "pairs":
            args = f"({', '.join(go(a) for a in n['args'])})" if n.get("args") else ""
            return f"pairs({go(n['base'])}, {name_of(n['rel'])}{args})"
        if op == "keys":
            return f"keys({name_of(n['measure'])})"
        if op == "not":
            return f"NOT {go(n['pred'])}"
        if op == "overlaps":
            return f"OVERLAPS({go(n['a'])}, {go(n['b'])})"
        if op == "together":
            return f"TOGETHER({go(n['a'])}, {go(n['b'])}, {n['container_sort']})"
        if op == "ordered_by":
            return f"ORDERED_BY({go(n['seq'])}, {n['key']['ref']}, {n['dir'].upper()})"
        if op == "in":
            return f"{go(n['left'])} IN {go(n['right'])}"
        if op == "multiple_of":
            return f"{go(n['left'])} MULTIPLE_OF {go(n['right'])}"
        if op == "where":
            return f"{go(n['base'])} WHERE {go(n['pred'])}"
        if op == "windows":
            count = go(n["n"]) if isinstance(n["n"], dict) else n["n"]
            return f"windows({n['unit'].split(':')[-1].upper()}, {count})"
        if op == "days":
            return f"days({n['horizon'].split(':')[-1]})"
        if op == "scenarios":
            return "scenarios"
        if op == "run_length":
            return f"RUN_LENGTH({binder(n)} in {go(n['seq'])}: {go(n['pred'])})"
        if op == "ratio":
            return f"RATIO({go(n['num'])}, {go(n['den'])})"
        if op == "offset":
            return f"offset({go(n['term'])}, {go(n['shift'])})"
        if op == "assigned_to":
            return f"ASSIGNED_TO({go(n['of'])}, {go(n['to'])})"
        if op == "assigned":
            return f"ASSIGNED({go(n['of'])})"
        if op == "has":
            return f"HAS({go(n['of'])}, {go(n['value'])})"
        if op == "during":
            return f"DURING({go(n['interval'])}, {go(n['window'])})"
        if op == "matches":
            return f"MATCHES({go(n['seq'])}, {go(n['pattern'])})"
        if op == "entity":
            return n["ref"].split(":")[-1]
        if op == "table_score":
            return f"table_score({go(n['table'])}, {', '.join(go(k) for k in n['keys'])})"
        if op == "opaque":
            return f"OPAQUE({n['fn_ref'].split(':')[-1]}, {', '.join(go(a) for a in n['args'])})"
        raise ValueError(f"emit: unknown op {op!r}")

    return go(node)


# --- the line format --------------------------------------------------------------------------


@dataclass(frozen=True)
class Composition:
    """`COMPOSE(weighted_sum: 1=3/5, 2=2/5)`, `COMPOSE(lexicographic: 1 > 2)`, `COMPOSE(product)`.
    Indices count the objective frames in order of appearance, from 1."""

    method: str
    weights: dict[int, str] = field(default_factory=dict)  # weighted_sum
    order: tuple[int, ...] = ()  # lexicographic


@dataclass(frozen=True)
class ParsedProblem:
    frames: tuple[tuple[Node, str | None], ...]  # (expression, HARD | SOFT | None)
    composition: Composition | None = None
    no_intent: bool = False
    clarify_intent: bool = False
    # UNEXPRESSED(class, side) lines, each standing in for a stated requirement the grammar cannot
    # express; (class, side) pairs in line order
    residuals: tuple[tuple[str, str], ...] = ()


NO_INTENT = "NO_INTENT"
CLARIFY_INTENT = "CLARIFY(intent)"
RESIDUAL_CLASSES = ("condition", "relative", "literal", "other")
RESIDUAL_SIDES = ("objective", "constraint")
_MARK = re.compile(r"^\[(HARD|SOFT)\]\s*(.*)$")
_COMPOSE = re.compile(r"^COMPOSE\(\s*([a-z_]+)\s*(?::\s*(.*?))?\s*\)$")
_RESIDUAL = re.compile(r"^UNEXPRESSED\(\s*([a-z]+)\s*,\s*([a-z]+)\s*\)$")


def emit_frame(tree: Node, hardness: str | None, vocab: Vocabulary) -> str:
    text = emit(tree, vocab)
    return f"[{hardness.upper()}] {text}" if hardness else text


def emit_composition(composition: Composition) -> str:
    if composition.method == "weighted_sum":
        body = ", ".join(f"{i}={w}" for i, w in sorted(composition.weights.items()))
        return f"COMPOSE(weighted_sum: {body})"
    if composition.method == "lexicographic":
        return f"COMPOSE(lexicographic: {' > '.join(str(i) for i in composition.order)})"
    return f"COMPOSE({composition.method})"


def emit_problem(
    frames: list[tuple[Node, str | None]],
    vocab: Vocabulary,
    composition: Composition | None = None,
    *,
    clarify_intent: bool = False,
    residuals: tuple[tuple[str, str], ...] = (),
) -> str:
    """Frames, then the UNEXPRESSED lines, then the COMPOSE line; with ``clarify_intent`` the text is
    CLARIFY(intent) alone. Binders are written x0, x1, ... numbered on across the frames in order (the
    canonical IR text). The numbering does not affect `ir_digest`, which renumbers each frame from x0."""
    if clarify_intent:
        if frames or composition is not None or residuals:
            raise ValueError("CLARIFY(intent) must stand alone")
        return CLARIFY_INTENT
    lines = []
    next_binder = 0
    for tree, hardness in frames:
        tree, next_binder = renumber_from(tree, next_binder)
        lines.append(emit_frame(tree, hardness, vocab))
    lines += [f"UNEXPRESSED({kind}, {side})" for kind, side in residuals]
    if composition is not None:
        lines.append(emit_composition(composition))
    return "\n".join(lines)


def parse_residual(line: str) -> tuple[str, str]:
    m = _RESIDUAL.match(line.strip())
    if m is None or m.group(1) not in RESIDUAL_CLASSES or m.group(2) not in RESIDUAL_SIDES:
        raise ValueError(f"malformed UNEXPRESSED line: {line!r}")
    return m.group(1), m.group(2)


def parse_composition(line: str) -> Composition:
    m = _COMPOSE.match(line.strip())
    if m is None:
        raise ValueError(f"malformed COMPOSE line: {line!r}")
    method, body = m.group(1), (m.group(2) or "").strip()
    if method == "weighted_sum":
        weights: dict[int, str] = {}
        for part in filter(None, (p.strip() for p in body.split(","))):
            idx, weight = part.split("=", 1)
            weights[int(idx)] = weight.strip()
        if not weights:
            raise ValueError("weighted_sum needs weights")
        return Composition("weighted_sum", weights=weights)
    if method == "lexicographic":
        order = tuple(int(p.strip()) for p in body.split(">") if p.strip())
        if not order:
            raise ValueError("lexicographic needs an order")
        return Composition("lexicographic", order=order)
    return Composition(method)


def parse_problem(text: str, vocab: Vocabulary) -> ParsedProblem:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        raise ValueError("empty problem")
    if lines == [NO_INTENT]:
        return ParsedProblem((), no_intent=True)
    if lines == [CLARIFY_INTENT]:
        return ParsedProblem((), clarify_intent=True)
    frames: list[tuple[Node, str | None]] = []
    residuals: list[tuple[str, str]] = []
    composition: Composition | None = None
    for line in lines:
        if line.startswith("COMPOSE("):
            if composition is not None:
                raise ValueError("more than one COMPOSE line")
            composition = parse_composition(line)
            continue
        if line in (NO_INTENT, CLARIFY_INTENT):
            raise ValueError(f"{line} must stand alone")
        if line.startswith("UNEXPRESSED("):  # no strength mark: the grammar holds no frame to mark
            residuals.append(parse_residual(line))
            continue
        hardness: str | None = None
        m = _MARK.match(line)
        if m:
            hardness, line = m.group(1), m.group(2)
        frames.append((parse(line, vocab), hardness))
    if not frames and not residuals:
        raise ValueError("a problem needs at least one frame")
    return ParsedProblem(tuple(frames), composition, residuals=tuple(residuals))


def ir_digest(parsed: ParsedProblem, vocab: Vocabulary) -> str:
    """SHA-256 identity of a parsed problem, shared by labels and evaluation: the normalized frames, their
    marks and the composition. The byte layout is kept stable so that the digest of an existing IR document
    does not change; the NO_INTENT, CLARIFY(intent) and UNEXPRESSED parts are appended only when present."""
    return hashlib.sha256(
        b"\n".join(canonical_bytes(normalize(t, vocab)) for t, _ in parsed.frames)
        + b"|"
        + json.dumps([h for _, h in parsed.frames]).encode()
        + b"|"
        + (json.dumps(parsed.composition.__dict__, sort_keys=True, default=str).encode() if parsed.composition else b"")
        + (b"|NO_INTENT" if parsed.no_intent else b"")
        + (b"|CLARIFY(intent)" if parsed.clarify_intent else b"")
        + (b"|UNEXPRESSED" + json.dumps(sorted(parsed.residuals)).encode() if parsed.residuals else b"")
    ).hexdigest()
