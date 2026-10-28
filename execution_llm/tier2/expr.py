"""IR expressions: the IR text parsed into a tree of nodes, normalized, serialized to canonical
bytes, and unified against capability forms.

Normalization implements a subset of the IR's normal form. The rule labels used in the
docstrings and section headers below are:

- NF-2: orient a comparison so the quantity is on the left and the bound on the right;
- NF-3: distribute an upper bound on a MAX (or a lower bound on a MIN) into an EACH, except
  over positions, where an EACH of bounds folds back into the extremum;
- NF-7: flatten nested ANDs and sort and deduplicate their items, order the two arguments of
  TOGETHER and OVERLAPS, and order the binders of an EACH chain by dependency;
- EN-2: renumber binders x0, x1, ... in one preorder walk;
- EN-3/EN-4: exact decimals, rescaled to the base unit;
- EN-6: canonical bytes (sorted-key JSON without whitespace, strings in NFC) and the erased
  form used as an ordering key.

Other rewrite rules of the normal form are not implemented.

Three node kinds appear only in a parsed IR, never in a capability form: the `hole` (a deliberately
unstated element: `HOLE(value)`, `HOLE(measure, ...)` or `HOLE(population)`), the `scope_hole`
(an unstated EACH chain, `EACH(x in HOLE(scope)): ...`) and the `frame_hole` (a whole constraint
named only by its family, `HOLE(constraint, FAMILY)`). Grounding unifies the IR tree against every
capability form: a `slot` in the form binds the tree's `value` at that position, and a `hole` in the
tree matches any element of its kind in the form. A `frame_hole` is not unified: the grounder
returns a clarify decision whose candidates are the capabilities of that family.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

Node = dict[str, Any]

AGG_OPS = ("SUM", "MAX", "MIN", "MEAN", "RANGE")
UNSPECIFIED = "UNSPECIFIED"  # an aggregation the user left unstated (node op "unspecified")
UNDECIDED = "UNDECIDED"  # a bound value the user says is not decided yet (node op "undecided")
COMPARATORS = {
    "<=": "le",
    "<": "lt",
    ">=": "ge",
    ">": "gt",
    "=": "eq",
    "!=": "ne",
    "EQ": "eq",
    "IN": "in",
    "MULTIPLE_OF": "multiple_of",
}
FLIP = {"le": "ge", "ge": "le", "lt": "gt", "gt": "lt", "eq": "eq", "ne": "ne"}
PREDICATES = {"TOGETHER", "ORDERED_BY", "OVERLAPS", "ASSIGNED", "ASSIGNED_TO", "DURING", "HAS", "MATCHES"}
POPULATION_CTORS = {"positions", "adjacent", "pairs", "keys"}


@dataclass(frozen=True)
class Measure:
    ref: str
    sort: str  # DISTANCE | DURATION | COST | COUNT | QUANTITY | FRACTION | SCALAR | TIME
    owner: str  # entity the measure is read on, e.g. "route", "visit", "worker", "horizon"
    gloss: str
    keyed: bool = False
    # A measure hole is matched per measure family, not per sort: a family is the set of measures
    # a user names by one loose word ("cost"). A measure with no declared family is a family of
    # one (a hole over it covers only that measure).
    family: str = ""

    @property
    def family_key(self) -> str:
        return self.family or self.ref


@dataclass(frozen=True)
class Accessor:
    ref: str
    sort: str
    gloss: str


@dataclass(frozen=True)
class Vocabulary:
    """The per-domain closed vocabulary the parser resolves names against."""

    domain: str
    entity_sets: Mapping[str, str]  # notation name -> entity ref ("routes" -> "entity:route")
    population_aliases: Mapping[str, str]  # declared identities ("riders" -> "routes", 1:1)
    containments: Mapping[str, str]  # "rounds" -> "entity:round" (within(container, ref))
    sequences: Mapping[str, str]  # "visit_seq" -> "attr:visit_seq"
    relations: Mapping[str, str]  # "nearby" -> "rel:nearby"
    measures: Mapping[str, Measure]
    accessors: Mapping[str, Accessor]
    units: Mapping[str, tuple[str, int]]  # "km" -> ("unit:m", 1000); COUNT kinds map to themselves x1
    population_gloss: Mapping[str, str] = field(default_factory=dict)


# --- tokenizer / parser ----------------------------------------------------------------------

_TOKEN = re.compile(r"\s*(?:(<=|>=|!=|<|>|=|\(|\)|\[|\]|,|:)|(\$?[A-Za-z_][A-Za-z_0-9]*)|(-?\d+(?:\.\d+)?))")


def _tokens(text: str) -> list[str]:
    out: list[str] = []
    pos = 0
    text = text.strip()
    while pos < len(text):
        match = _TOKEN.match(text, pos)
        if match is None or match.end() == pos:
            raise ValueError(f"cannot tokenize at {text[pos : pos + 20]!r}")
        out.append(next(g for g in match.groups() if g is not None))
        pos = match.end()
    return out


class _Parser:
    def __init__(self, text: str, vocab: Vocabulary) -> None:
        self.toks = _tokens(text)
        self.i = 0
        self.vocab = vocab

    def peek(self, k: int = 0) -> str | None:
        return self.toks[self.i + k] if self.i + k < len(self.toks) else None

    def take(self, expected: str | None = None) -> str:
        tok = self.peek()
        if tok is None or (expected is not None and tok != expected):
            raise ValueError(f"expected {expected!r} at token {self.i}, got {tok!r} in {' '.join(self.toks)}")
        self.i += 1
        return tok

    # frame roots -------------------------------------------------------------------------
    def root(self) -> Node:
        tok = self.peek()
        if tok == "HOLE" and self.peek(2) == "constraint":
            # HOLE(constraint, FAMILY): a whole constraint frame the user named only loosely;
            # the grounder offers the family's signatures as candidates
            self.take()
            self.take("(")
            self.take("constraint")
            self.take(",")
            family = self.take()
            self.take(")")
            return {"op": "frame_hole", "family": family}
        if tok in ("MINIMIZE", "MAXIMIZE"):
            self.take()
            self.take("(")
            term = self.quantity()
            self.take(")")
            return {"op": tok.lower(), "term": term}
        return self.constraint()

    def constraint(self) -> Node:
        tok = self.peek()
        if tok == "AND":
            self.take()
            self.take("(")
            items = [self.constraint()]
            while self.peek() == ",":
                self.take()
                items.append(self.constraint())
            self.take(")")
            return {"op": "and", "items": items}
        if tok == "EACH":
            return self.each_chain()
        return self.predicate()

    def each_chain(self) -> Node:
        binders: list[tuple[Node, Node]] = []
        while self.peek() == "EACH":
            self.take()
            self.take("(")
            binder = self.binder()
            self.take("in")
            over = self.population()
            self.take(")")
            binders.append((binder, over))
            if self.peek() == ",":
                self.take()
        self.take(":")
        body = self.constraint()
        for binder, over in reversed(binders):
            if over.get("op") == "hole" and over.get("kind") == "scope":
                # EACH(x in HOLE(scope)): body - the whole EACH chain is unstated; x stands for
                # the innermost binder of the signature it is unified with (see unify)
                if len(binders) != 1 or "var" not in binder:
                    raise ValueError("a scope hole must be the only binder of its chain")
                return {"op": "scope_hole", "var": binder["var"], "body": body}
            body = {"op": "each", **binder, "over": over, "body": body}
        return body

    def binder(self) -> Node:
        if self.peek() == "(":
            self.take()
            a = self.take()
            self.take(",")
            b = self.take()
            self.take(")")
            return {"vars": [a, b]}
        return {"var": self.take()}

    # predicates and comparisons ----------------------------------------------------------
    def predicate(self) -> Node:
        tok = self.peek()
        if tok == "NOT":
            self.take()
            return {"op": "not", "pred": self.predicate()}
        if tok in PREDICATES:
            self.take()
            self.take("(")
            args = [self.term()]
            while self.peek() == ",":
                self.take()
                args.append(self.term())
            self.take(")")
            return self._predicate(tok, args)
        left = self.term()
        cmp = self.peek()
        if cmp not in COMPARATORS:
            if left.get("op") == "attr":  # a BOOLEAN accessor in predicate position (works(w, d))
                return left
            raise ValueError(f"expected a comparator, got {cmp!r}")
        self.take()
        right = self.term()
        return {"op": COMPARATORS[cmp], "left": left, "right": right}

    def _predicate(self, name: str, args: list[Node]) -> Node:
        if name == "TOGETHER":
            return {"op": "together", "a": args[0], "b": args[1], "container_sort": args[2]["ref"]}
        if name == "ORDERED_BY":
            return {
                "op": "ordered_by",
                "seq": args[0],
                "key": {"op": "fn", "ref": args[1]["ref"]},
                "dir": args[2]["ref"].lower(),
            }
        if name == "OVERLAPS":
            return {"op": "overlaps", "a": args[0], "b": args[1]}
        if name == "ASSIGNED":
            return {"op": "assigned", "of": args[0]}
        if name == "ASSIGNED_TO":
            return {"op": "assigned_to", "of": args[0], "to": args[1]}
        if name == "DURING":
            return {"op": "during", "interval": args[0], "window": args[1]}
        if name == "HAS":
            return {"op": "has", "of": args[0], "value": args[1]}
        if name == "MATCHES":
            return {"op": "matches", "seq": args[0], "pattern": args[1]}
        raise ValueError(name)

    # terms -------------------------------------------------------------------------------
    def term(self) -> Node:
        tok = self.peek()
        if tok is None:
            raise ValueError("the IR ends where a term is expected")
        if tok.startswith("$"):
            return self.slot()
        if re.match(r"-?\d", tok):
            return self.value()
        if tok == UNDECIDED:
            # a bound the user says has no value yet; the grounder drops the frame and adds a notice
            self.take()
            return {"op": "undecided"}
        if tok in AGG_OPS or tok in (
            UNSPECIFIED,
            "COUNT",
            "COUNT_DISTINCT",
            "table_score",
            "OPAQUE",
            "HOLE",
            "RUN_LENGTH",
            "RATIO",
            "offset",
        ):
            return self.quantity()
        return self.applied()

    def quantity(self) -> Node:
        tok = self.take()
        if tok in AGG_OPS or tok == UNSPECIFIED:
            # UNSPECIFIED: the quantity is named but its aggregation is not; the grounder asks, offering
            # the domain's declared plain reading (if any) as the candidate. The parser never picks an aggregation
            self.take("(")
            binder = self.binder()
            self.take("in")
            over = self.population()
            self.take(":")
            term = self.term()
            self.take(")")
            return {"op": tok.lower(), **binder, "over": over, "term": term}
        if tok == "RUN_LENGTH":
            self.take("(")
            binder = self.binder()
            self.take("in")
            seq = self.applied()
            self.take(":")
            pred = self.predicate()
            self.take(")")
            return {"op": "run_length", **binder, "seq": seq, "pred": pred}
        if tok == "RATIO":
            self.take("(")
            num = self.term()
            self.take(",")
            den = self.term()
            self.take(")")
            return {"op": "ratio", "num": num, "den": den}
        if tok == "offset":
            self.take("(")
            term = self.term()
            self.take(",")
            shift = self.term()
            self.take(")")
            return {"op": "offset", "term": term, "shift": shift}
        if tok == "COUNT":
            self.take("(")
            binder = self.binder()
            self.take("in")
            over = self.population()
            self.take(")")
            return {"op": "count", **binder, "over": over}
        if tok == "COUNT_DISTINCT":
            self.take("(")
            binder = self.binder()
            self.take("in")
            over = self.population()
            self.take(":")
            key = self.applied()
            self.take(")")
            return {"op": "count_distinct", **binder, "over": over, "key": key}
        if tok == "table_score":
            self.take("(")
            table = self.slot()
            keys = []
            while self.peek() == ",":
                self.take()
                keys.append(self.applied())
            self.take(")")
            return {"op": "table_score", "table": table, "keys": keys}
        if tok == "OPAQUE":
            self.take("(")
            fn = self.take()
            args = []
            while self.peek() == ",":
                self.take()
                args.append(self.term())
            self.take(")")
            return {"op": "opaque", "fn_ref": f"fn:{fn}", "args": args}
        if tok == "HOLE":
            # HOLE(measure, FAMILY, x) / HOLE(value): holes that appear only in a parsed IR
            self.take("(")
            kind = self.take()
            node: Node = {"op": "hole", "kind": kind}
            if kind == "measure":
                self.take(",")
                family = self.take()
                # a family token is either a declared family name ("cost") or a measure's own
                # notation name (a family of one: its key is the measure ref)
                measure = self.vocab.measures.get(family)
                node["family"] = family if measure is None or measure.family else measure.ref
                if self.peek() == ",":
                    self.take(",")
                    node["of"] = self.applied()
                else:  # HOLE(measure, profit): no owner argument, read on the planning horizon
                    node["of"] = {"op": "entity", "ref": "entity:horizon"}
            self.take(")")
            return node
        self.i -= 1
        return self.applied()

    def slot(self) -> Node:
        name = self.take()
        node: Node = {"op": "slot", "ref": f"sig:{name[1:]}"}
        if self.peek() == "[":
            self.take()
            node["key"] = {"op": "var", "ref": self.take()}
            self.take("]")
        return node

    def value(self) -> Node:
        num = self.take()
        unit = None
        nxt = self.peek()
        if nxt is not None and nxt in self.vocab.units:
            unit = self.take()
        return {"op": "value", "num": num, "unit": unit}

    def applied(self) -> Node:
        """measure(e) / measure[k](e) / attr(e...) / bare variable or entity name."""
        name = self.take()
        key: Node | None = None
        if self.peek() == "[":
            self.take()
            key = {"op": "var", "ref": self.take()}
            self.take("]")
        if self.peek() != "(":
            if name in self.vocab.measures:  # a bare measure name, read on the planning horizon
                return {
                    "op": "measure",
                    "ref": self.vocab.measures[name].ref,
                    "of": {"op": "entity", "ref": "entity:horizon"},
                }
            return {"op": "var", "ref": name}
        self.take("(")
        args = [self.term()]
        while self.peek() == ",":
            self.take()
            args.append(self.term())
        self.take(")")
        if name in self.vocab.measures:
            node: Node = {"op": "measure", "ref": self.vocab.measures[name].ref, "of": args[0]}
            if key is not None:
                node["key"] = key
            return node
        if name in self.vocab.accessors or name in self.vocab.sequences:
            ref = self.vocab.accessors[name].ref if name in self.vocab.accessors else self.vocab.sequences[name]
            attr: Node = {"op": "attr", "ref": ref, "args": args}
            if key is not None:
                attr["key"] = key
            return attr
        raise ValueError(f"unknown applied symbol {name!r}")

    # populations -------------------------------------------------------------------------
    def population(self) -> Node:
        base = self._population_base()
        if self.peek() == "WHERE":
            self.take()
            return {"op": "where", "base": base, "pred": self.predicate()}
        return base

    def _population_base(self) -> Node:
        tok = self.peek()
        if tok is None:
            raise ValueError("the IR ends where a population is expected")
        if tok.startswith("$"):
            return self.slot()
        if tok == "scenarios":
            self.take()
            return {"op": "scenarios"}
        if tok == "HOLE":
            self.take()
            self.take("(")
            kind = self.take()  # population | scope (an unstated EACH chain, see each_chain)
            if kind not in ("population", "scope"):
                raise ValueError(f"unknown population hole kind {kind!r}")
            self.take(")")
            return {"op": "hole", "kind": kind}
        name = self.take()
        name = self.vocab.population_aliases.get(name, name)
        if self.peek() != "(":
            if name not in self.vocab.entity_sets:
                raise ValueError(f"unknown entity set {name!r}")
            return {"op": "pop", "ref": self.vocab.entity_sets[name]}
        self.take("(")
        if name == "pairs":
            base = self.population()
            self.take(",")
            rel = self.take()
            node: Node = {"op": "pairs", "base": base, "rel": self.vocab.relations[rel]}
            if self.peek() == "(":
                self.take()
                node["args"] = [self.term()]
                while self.peek() == ",":
                    self.take()
                    node["args"].append(self.term())
                self.take(")")
            self.take(")")
            return node
        if name == "keys":
            measure = self.take()
            self.take(")")
            return {"op": "keys", "measure": self.vocab.measures[measure].ref}
        if name == "windows":
            unit = self.take()
            self.take(",")
            n: Node | str = self.slot() if (self.peek() or "").startswith("$") else canonical_decimal(self.take())
            self.take(")")
            return {"op": "windows", "unit": f"unit:{unit.lower()}", "n": n}
        if name == "days":
            horizon = self.take()
            self.take(")")
            return {"op": "days", "horizon": f"entity:{horizon}"}
        if name == "positions":
            container = self.term()
            self.take(")")
            return {"op": "positions", "container": container}
        if name == "adjacent":
            seq = self.applied()
            self.take(")")
            return {"op": "adjacent", "seq": seq}
        if name in self.vocab.containments:
            container = self.term()
            self.take(")")
            return {"op": "within", "container": container, "ref": self.vocab.containments[name]}
        raise ValueError(f"unknown population constructor {name!r}")


def parse(text: str, vocab: Vocabulary, *, hardness: str | None = None) -> Node:
    """Parse one frame's expression. With ``hardness`` the result is wrapped as a frame.

    Raises ValueError for any text that is not a valid frame, including text that trips an internal
    lookup (an unknown relation or measure, a missing argument, a non-numeric window size)."""

    parser = _Parser(text, vocab)
    try:
        expression = parser.root()
    except (AssertionError, KeyError, IndexError, ArithmeticError) as error:  # ArithmeticError: decimal errors
        raise ValueError(f"malformed IR frame {text!r}: {type(error).__name__} {error}") from error
    if parser.peek() is not None:
        raise ValueError(f"trailing tokens: {parser.toks[parser.i :]}")
    if hardness is None:
        return expression
    return {"op": "frame", "expression": expression, "hardness": hardness}


# --- tree utilities -------------------------------------------------------------------------


def _children(node: Node) -> list[tuple[str, Node]]:
    out: list[tuple[str, Node]] = []
    for key, value in node.items():
        if isinstance(value, dict) and "op" in value:
            out.append((key, value))
        elif isinstance(value, list):
            out += [(key, v) for v in value if isinstance(v, dict) and "op" in v]
    return out


def walk(node: Node):
    yield node
    for _, child in _children(node):
        yield from walk(child)


def _binder_names(node: Node) -> list[str]:
    if "vars" in node:
        return list(node["vars"])
    if "var" in node:
        return [node["var"]]
    return []


def _mentions(node: Node, names: set[str]) -> bool:
    return any(n.get("op") == "var" and n["ref"] in names for n in walk(node))


# --- EN-3 / EN-4: exact decimals in base units --------------------------------------------


def canonical_decimal(num: str | Decimal) -> str:
    d = Decimal(str(num))
    text = format(d.normalize(), "f")
    if text in ("-0", "-0.0"):
        text = "0"
    return text


def rescale_value(node: Node, vocab: Vocabulary) -> Node:
    unit = node.get("unit")
    if unit is None or unit.startswith("unit:"):  # dimensionless, or already in the base unit
        return {"op": "value", "num": canonical_decimal(node["num"]), "unit": unit}
    base, factor = vocab.units[unit]
    return {"op": "value", "num": canonical_decimal(Decimal(node["num"]) * factor), "unit": base}


# --- erased form and canonical bytes (EN-2, EN-6) ------------------------------------------


def _jcs(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _nfc(obj: Any) -> Any:
    if isinstance(obj, str):
        return unicodedata.normalize("NFC", obj)
    if isinstance(obj, list):
        return [_nfc(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _nfc(v) for k, v in obj.items()}
    return obj


def erased(node: Node, stack: tuple[tuple[str, ...], ...] = ()) -> Any:
    """EN-6's ordering key: binder names dropped, variable references as [distance, component]."""

    if node.get("op") == "var":
        for distance, seat in enumerate(reversed(stack)):
            if node["ref"] in seat:
                return [distance, seat.index(node["ref"])]
        return ["free", node["ref"]]
    names = tuple(_binder_names(node))
    inner = (*stack, names) if names else stack
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in ("var", "vars"):
            continue
        if isinstance(value, dict) and "op" in value:
            out[key] = erased(value, inner)
        elif isinstance(value, list):
            out[key] = [erased(v, inner) if isinstance(v, dict) and "op" in v else v for v in value]
        else:
            out[key] = value
    return out


def erased_key(node: Node) -> str:
    return _jcs(erased(node))


def renumber(node: Node) -> Node:
    """EN-2: binders renumbered x0, x1, ... in one preorder walk (a vars seat takes two)."""
    return renumber_from(node, 0)[0]


def renumber_from(node: Node, start: int) -> tuple[Node, int]:
    """`renumber` starting at x{start}; returns the tree and the next free index, so the frames of one
    problem continue a single numbering, as the canonical IR text does."""

    counter = [start]

    def go(n: Node, env: dict[str, str]) -> Node:
        if n.get("op") == "var":
            return {"op": "var", "ref": env.get(n["ref"], n["ref"])}
        env = dict(env)
        out: Node = {}
        names = _binder_names(n)
        fresh = []
        for name in names:
            fresh.append(f"x{counter[0]}")
            counter[0] += 1
        # every seat of a binder node is inside its scope (a WHERE filter mentions the binder),
        # matching erased()'s stack discipline
        for key, value in n.items():
            if key == "var":
                out["var"] = fresh[0]
            elif key == "vars":
                out["vars"] = fresh
            else:
                inner = dict(env, **dict(zip(names, fresh, strict=True))) if names else env
                if isinstance(value, dict) and "op" in value:
                    out[key] = go(value, inner)
                elif isinstance(value, list):
                    out[key] = [go(v, inner) if isinstance(v, dict) and "op" in v else v for v in value]
                else:
                    out[key] = value
        return out

    renumbered = go(node, {})
    return renumbered, counter[0]


def canonical_bytes(node: Node) -> bytes:
    return _jcs(_nfc(renumber(node))).encode()


# --- the rewrite system (subset) --------------------------------------------------------------

_MEASURE_ROOTED = ("measure", "offset")
_QUANTITY = (
    "sum",
    "max",
    "min",
    "mean",
    "range",
    "unspecified",
    "count",
    "count_distinct",
    "ratio",
    "quantile",
    "run_length",
    "table_score",
    "opaque",
    "var",
    "hole",
)


def _rank(node: Node) -> int:
    op = node.get("op")
    if op in _MEASURE_ROOTED:
        return 0
    if op in _QUANTITY:
        return 1
    return 2  # bare accessor


def _is_bound(node: Node) -> bool:
    return node.get("op") in ("value", "slot", "undecided") or (
        node.get("op") == "hole" and node.get("kind") == "value"
    )


def _orient(node: Node) -> Node:
    """NF-2: term left, bound right; two terms ordered by rank then erased form."""

    op = node["op"]
    if op not in FLIP:
        return node
    left, right = node["left"], node["right"]
    if _is_bound(left) and not _is_bound(right):
        return {"op": FLIP[op], "left": right, "right": left}
    if _is_bound(right) or _is_bound(left):
        return node
    key_l, key_r = (_rank(left), erased_key(left)), (_rank(right), erased_key(right))
    if key_r < key_l:
        return {"op": FLIP[op], "left": right, "right": left}
    return node


def _nf3(node: Node) -> Node:
    """NF-3: le(max(x in P: t), B) -> each(x in P): le(t, B) except over positions, where the
    distributed spelling folds into the extremum."""

    op = node.get("op")
    if op in ("le", "lt") and node["left"].get("op") == "max" or op in ("ge", "gt") and node["left"].get("op") == "min":
        agg = node["left"]
        if agg["over"].get("op") != "positions":
            return {
                "op": "each",
                **{k: agg[k] for k in ("var", "vars") if k in agg},
                "over": agg["over"],
                "body": {"op": op, "left": agg["term"], "right": node["right"]},
            }
    if op == "each" and node["over"].get("op") == "positions":
        body = node["body"]
        if body.get("op") in ("le", "lt", "ge", "gt") and _is_bound(body["right"]):
            agg_op = "max" if body["op"] in ("le", "lt") else "min"
            binder = {k: node[k] for k in ("var", "vars") if k in node}
            return {
                "op": body["op"],
                "left": {"op": agg_op, **binder, "over": node["over"], "term": body["left"]},
                "right": body["right"],
            }
    return node


def _nf7(node: Node) -> Node:
    op = node.get("op")
    if op == "and":
        flat: list[Node] = []
        for item in node["items"]:
            flat += item["items"] if item.get("op") == "and" else [item]
        seen: dict[str, Node] = {}
        for item in flat:
            seen.setdefault(erased_key(item), item)
        items = [seen[k] for k in sorted(seen)]
        return items[0] if len(items) == 1 else {"op": "and", "items": items}
    if op in ("together", "overlaps"):
        a, b = node["a"], node["b"]
        if erased_key(b) < erased_key(a):
            return {**node, "a": b, "b": a}
    if op == "each":
        return _reorder_chain(node)
    return node


def _reorder_chain(node: Node) -> Node:
    """NF-7's binder-chain order: Kahn order over the dependency DAG of a maximal EACH chain,
    ties broken by the erased population. The tie-break keys are computed once, under the binder
    numbering as written, rather than after each Kahn step; on the routing signatures both give
    the same order."""

    chain: list[Node] = []
    cursor = node
    while cursor.get("op") == "each":
        chain.append(cursor)
        cursor = cursor["body"]
    if len(chain) < 2:
        return node
    body = cursor
    names = [set(_binder_names(b)) for b in chain]
    deps = [
        {j for j in range(len(chain)) if j != i and _mentions(chain[i]["over"], names[j])} for i in range(len(chain))
    ]
    order: list[int] = []
    remaining = set(range(len(chain)))
    while remaining:
        ready = sorted((erased_key(chain[i]["over"]), i) for i in remaining if deps[i] <= set(order))
        if not ready:
            raise ValueError("cyclic binder chain")
        order.append(ready[0][1])
        remaining.discard(ready[0][1])
    if order == list(range(len(chain))):
        return node
    rebuilt = body
    for i in reversed(order):
        rebuilt = {**{k: v for k, v in chain[i].items() if k != "body"}, "body": rebuilt}
    return rebuilt


def _rewrite(node: Node, vocab: Vocabulary) -> Node:
    if node.get("op") == "value":
        return rescale_value(node, vocab)
    out: Node = {}
    for key, value in node.items():
        if isinstance(value, dict) and "op" in value:
            out[key] = _rewrite(value, vocab)
        elif isinstance(value, list):
            out[key] = [_rewrite(v, vocab) if isinstance(v, dict) and "op" in v else v for v in value]
        else:
            out[key] = value
    return _nf7(_nf3(_orient(out)))


def normalize(node: Node, vocab: Vocabulary) -> Node:
    """Apply the rewrites until the tree stops changing, then renumber binders (EN-2). Raises
    ValueError if no fixpoint is reached within 8 passes."""

    current = node
    for _ in range(8):
        rewritten = _rewrite(current, vocab)
        if _jcs(rewritten) == _jcs(current):
            return renumber(current)
        current = rewritten
    raise ValueError("normalization did not reach a fixpoint")


# --- unification: user tree against a signature -----------------------------------------------


@dataclass
class Match:
    bindings: dict[str, Node] = field(default_factory=dict)  # signature slot ref -> value node
    missing: list[str] = field(default_factory=list)  # signature slots met by a value hole
    holes: list[Node] = field(default_factory=list)  # tree holes satisfied by the signature
    var_map: dict[str, str] = field(default_factory=dict)

    def merged(self, other: Match) -> Match | None:
        for ref, value in other.bindings.items():
            if ref in self.bindings and _jcs(self.bindings[ref]) != _jcs(value):
                return None
        return Match(
            {**self.bindings, **other.bindings},
            self.missing + other.missing,
            self.holes + other.holes,
            {**self.var_map, **other.var_map},
        )


def _strip_each(node: Node) -> tuple[list[Node], Node]:
    chain = []
    while node.get("op") == "each":
        chain.append(node)
        node = node["body"]
    return chain, node


def unify(signature: Node, tree: Node, vocab: Vocabulary, match: Match | None = None) -> Match | None:
    """Structural unification of a normalized user tree with a normalized signature.
    Signature slots bind tree values; tree holes accept signature elements of their kind."""

    m = match if match is not None else Match()
    s_op, t_op = signature.get("op"), tree.get("op")
    if t_op == "hole":
        kind = tree["kind"]
        if kind == "value" and s_op == "slot":
            m.missing.append(signature["ref"])
            return m
        if kind == "measure" and s_op == "measure":
            measure = next(mm for mm in vocab.measures.values() if mm.ref == signature["ref"])
            if measure.family_key != tree["family"]:
                return None
            m.holes.append(tree)
            return unify(signature["of"], tree["of"], vocab, m)
        if kind == "population" and s_op in ("pop", "within", "positions", "adjacent", "pairs", "keys", "slot"):
            m.holes.append(tree)
            return m
        return None
    if t_op == "scope_hole":
        chain, body = _strip_each(signature)
        if not chain:
            return None
        inner = tree["body"]
        # the hole's own binder stands for the innermost signature binder
        m.var_map[tree["var"]] = chain[-1].get("var", "")
        m.holes.append(tree)
        return unify(body, inner, vocab, m)
    if s_op == "slot" and t_op == "value":
        m.bindings[signature["ref"]] = tree
        return m
    if s_op == "slot" and t_op in ("pop", "within", "where", "windows", "days", "scenarios"):
        m.bindings[signature["ref"]] = tree  # population-typed slot bound to a whole population
        return m
    if s_op == "var" and t_op == "var":
        bound = m.var_map.get(tree["ref"])
        if bound is None:
            m.var_map[tree["ref"]] = signature["ref"]
            return m
        return m if bound == signature["ref"] else None
    if s_op != t_op:
        return None
    if set(signature) != set(tree):
        return None
    for key in sorted(signature):
        s_val, t_val = signature[key], tree[key]
        if isinstance(s_val, dict) and "op" in s_val:
            if not (isinstance(t_val, dict) and "op" in t_val):
                return None
            if unify(s_val, t_val, vocab, m) is None:
                return None
        elif isinstance(s_val, list):
            if not isinstance(t_val, list) or len(s_val) != len(t_val):
                return None
            for s_item, t_item in zip(s_val, t_val, strict=True):
                if isinstance(s_item, dict) and "op" in s_item:
                    if unify(s_item, t_item, vocab, m) is None:
                        return None
                elif s_item != t_item:
                    return None
        elif key in ("var", "vars"):
            names_s, names_t = ([s_val] if key == "var" else list(s_val)), ([t_val] if key == "var" else list(t_val))
            for ns, nt in zip(names_s, names_t, strict=True):
                m.var_map[nt] = ns
        elif s_val != t_val:
            return None
    return m


# --- tree rewriting and query helpers ----------------------------------------------------------


def substitute(node: Node, replace: Mapping[str, Node]) -> Node:
    """Replace signature slot nodes by ref with the given nodes. The result is not normalized:
    bind first, then normalize."""

    if node.get("op") == "slot" and node["ref"] in replace:
        return dict(replace[node["ref"]])
    out: Node = {}
    for key, value in node.items():
        if isinstance(value, dict) and "op" in value:
            out[key] = substitute(value, replace)
        elif isinstance(value, list):
            out[key] = [substitute(v, replace) if isinstance(v, dict) and "op" in v else v for v in value]
        else:
            out[key] = value
    return out


def replace_where(node: Node, pred, make) -> Node:
    """Rebuild the tree, replacing every node satisfying ``pred`` by ``make(node)``."""

    if pred(node):
        return make(node)
    out: Node = {}
    for key, value in node.items():
        if isinstance(value, dict) and "op" in value:
            out[key] = replace_where(value, pred, make)
        elif isinstance(value, list):
            out[key] = [replace_where(v, pred, make) if isinstance(v, dict) and "op" in v else v for v in value]
        else:
            out[key] = value
    return out


def slots_in(node: Node) -> list[str]:
    return [n["ref"] for n in walk(node) if n.get("op") == "slot"]


def values_in(node: Node) -> list[Node]:
    return [n for n in walk(node) if n.get("op") == "value"]


def has_holes(node: Node) -> bool:
    return any(n.get("op") in ("hole", "scope_hole", "frame_hole") for n in walk(node))


def frames_of(problem: Sequence[Node]) -> Node:
    return {"op": "problem", "frames": list(problem)}
