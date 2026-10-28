"""EBNF for the IR line format, derived from `emit` and `_Parser`.

The grammar follows `emit_problem` over one domain's vocabulary, narrowed by the closures described
in `ebnf` and with every repetition bounded. It is meant as the decoder constraint of the parser
model (xgrammar / vLLM structured outputs), so a small model's output stays within the line format's
syntax. It does not repeat every check of `parse_problem`: a few admitted strings are still rejected
(a keyword such as offset as the last argument of TOGETHER, a binder pair over HOLE(scope)), and
semantic validity stays with the grounder.

The syntax is xgrammar's EBNF (GBNF-style): `root ::= ...`, quoted literals, character
classes, `? * +` and `|`.

Rule names are not IR keywords: `residual` is UNEXPRESSED, `plain` is UNSPECIFIED, `rule_hole` is
HOLE(constraint, family) (op `frame_hole` in the tree), and HOLE(scope) is op `scope_hole`. The
literal "\\x00" matches nothing; it fills a list that is empty in this domain.
"""

from __future__ import annotations

from collections.abc import Iterable

from execution_llm.tier2.expr import Vocabulary
from execution_llm.tier2.surface import Signature

_KEYWORD_FIXED = """
root ::= "NO_INTENT" | "CLARIFY(intent)" | line ("\\n" line){0,5} ("\\n" compose)?
line ::= frame | residual
residual ::= "UNEXPRESSED(" ("condition" | "relative" | "literal" | "other") ", " ("objective" | "constraint") ")"
frame ::= ("[HARD] " | "[SOFT] ")? (objective | constraint | rule_hole)
objective ::= ("MINIMIZE( " | "MAXIMIZE( ") qterm " )"
constraint ::= "AND( " constraint (", " constraint){1,5} " )" | each | predicate
each ::= "EACH(" binder " in HOLE(scope)): " constraint | "EACH(" binder " in " population "): " constraint
binder ::= bvar | "(" bvar ", " bvar ")"
bvar ::= "x" [0-9]{1,2}
predicate ::= "NOT " predicate
  | "OVERLAPS(" term ", " term ")"
  | "TOGETHER(" term ", " term ", " var ")"
  | "ORDERED_BY(" term ", " attr_name ", " ("ASC" | "DESC") ")"
  | "ASSIGNED_TO(" term ", " term ")"
  | "ASSIGNED(" term ")"
  | "HAS(" term ", " term ")"
  | "DURING(" term ", " term ")"
  | "MATCHES(" term ", " term ")"
  | qterm (" <= " | " < " | " >= " | " > " | " = " | " != " | " IN " | " MULTIPLE_OF ") term
  | bool_attr
term ::= slot | value | "UNDECIDED" | qterm
qterm ::= agg | plain
  | "RUN_LENGTH(" binder " in " applied ": " predicate ")"
  | "RATIO(" term ", " term ")"
  | "offset(" term ", " term ")"
  | "COUNT(" binder " in " population ")"
  | "COUNT_DISTINCT(" binder " in " population ": " applied ")"
  | "table_score(" slot (", " applied){0,3} ")"
  | "HOLE(value)"
  | "HOLE(measure, " family (", " applied)? ")"
  | applied
agg ::= ("SUM(" | "MAX(" | "MIN(" | "MEAN(" | "RANGE(") binder " in " population ": " term ")"
population ::= pop_base (" WHERE " predicate)?
slot ::= "$" slot_name ("[" bvar "]")?
value ::= number (" " unit)?
number ::= "-"? [0-9]{1,9} ("." [0-9]{1,6})?
key ::= "[" bvar "]"
var ::= [a-z] [a-z0-9_]{0,15}
compose ::= "COMPOSE(weighted_sum: " witem (", " witem){1,5} ")" | "COMPOSE(lexicographic: " index (" > " index){1,5} ")" | "COMPOSE(product)"
witem ::= index "=" weight
index ::= [1-9]
weight ::= [0-9]{1,4} ("/" [1-9] [0-9]{0,3} | "." [0-9]{1,4})?
"""


def _alt(names: list[str]) -> str:
    """Longest first so a shared prefix never shadows the longer literal."""
    quoted = [f'"{n}"' for n in sorted(set(names), key=lambda s: (-len(s), s))]
    return " | ".join(quoted) if quoted else '"\\x00"'  # an unmatchable literal for an empty table


def ebnf(vocab: Vocabulary, signatures: Iterable[Signature] = (), plain_readings: Iterable[str] = ()) -> str:
    """``signatures``: the domain's declared signatures; they close the ``$name`` slot tokens and the
    rule families of ``HOLE(constraint, family)``. With none given any short lowercase slot name is
    admitted and no frame hole is. ``plain_readings``: the measure refs of the domain's plain-reading
    table, the only measures ``UNSPECIFIED(...)`` may aggregate (with none, ``UNSPECIFIED`` is not admitted).

    The grammar is narrower than the parser: binders are x0..x99, there is no ``HOLE(population)`` (an
    unstated subset means the whole population), and a measure hole names a declared family, never a
    single measure. A comparison's left side is a quantity, never a bare value or slot: the normal form
    orients bounds to the right, and a line that opens with a number is usually a model starting a
    numbered explanation ("1. ..."), which constrained decoding would otherwise force into a value such
    as "1.100000 ..."."""
    signatures = list(signatures)
    measures = list(vocab.measures)
    slots = sorted({k.split(":")[-1] for sig in signatures for k in sig.slots})
    families_r = sorted({sig.family for sig in signatures if sig.family})
    attrs = list(vocab.accessors) + list(vocab.sequences)
    families = sorted({m.family for m in vocab.measures.values() if m.family})
    plain_refs = set(plain_readings)
    plain = [name for name, m in vocab.measures.items() if m.ref in plain_refs]
    pops = list(vocab.entity_sets) + list(vocab.population_aliases)
    ctors = [
        "slot",
        '"scenarios"',
        "entity_set",
        '"keys(" measure_name ")"',
        '"windows(" ("WEEK" | "DAY" | "MONTH") ", " (slot | number) ")"',
        '"days(" var ")"',
        '"positions(" term ")"',
        '"adjacent(" applied ")"',
    ]
    if vocab.relations:
        ctors.append('"pairs(" population ", " rel_name ("(" term (", " term){0,3} ")")? ")"')
    if vocab.containments:
        ctors.append('containment "(" term ")"')
    # applied: measure(of) / measure[k](of) / attr(args) / a bare population-owned measure / a variable.
    # Known gap: `var` also admits the lowercase keywords offset and table_score; as the last argument of
    # TOGETHER the parser reads either as the start of a quantity and rejects the frame. A model does not
    # write a keyword there in practice, so the grammar carries no exclusion for them.
    # Every repetition is bounded (see the fixed rules): under constrained decoding a model with no
    # legal continuation would otherwise spiral inside an identifier or a number until max_tokens.
    applied = ['measure_name key? "(" term ")"', "measure_name", "bvar"]
    if attrs:
        applied.insert(1, 'attr_name key? "(" term (", " term){0,3} ")"')
    tables = [
        f"measure_name ::= {_alt(measures)}",
        (
            f'plain ::= "UNSPECIFIED(" binder " in " population ": " ({_alt(plain)}) key? "(" term ")" ")"'
            if plain
            else 'plain ::= "\\x00"'
        ),
        f"attr_name ::= {_alt(attrs)}",
        f"family ::= {_alt(families)}",
        f"entity_set ::= {_alt(pops)}",
        f"containment ::= {_alt(list(vocab.containments))}",
        f"rel_name ::= {_alt(list(vocab.relations))}",
        f"unit ::= {_alt(list(vocab.units))}",
        f"slot_name ::= {_alt(slots)}" if slots else "slot_name ::= [a-z] [a-z0-9_]{0,31}",
        f'rule_hole ::= "HOLE(constraint, " ({_alt(families_r)}) ")"' if families_r else 'rule_hole ::= "\\x00"',
        "pop_base ::= " + " | ".join(ctors),
        "applied ::= " + " | ".join(applied),
        'bool_attr ::= attr_name "(" term (", " term){0,3} ")"' if attrs else 'bool_attr ::= "\\x00"',
    ]
    return _KEYWORD_FIXED.strip() + "\n" + "\n".join(tables) + "\n"
