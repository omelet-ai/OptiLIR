# OptiLIR

Code for parsing optimization requests into OptiLIR (Optimization-oriented Language Intermediate
Representation; see [Terms](#terms)) and grounding the IR into decisions. It goes with a dataset and two
LoRA parser adapters on Hugging Face:

- dataset: [`omelet-ai/OptiLIR-Instruct-v1`](https://huggingface.co/datasets/omelet-ai/OptiLIR-Instruct-v1)
- adapters: [`omelet-ai/OptiLIR-Qwen3-8B`](https://huggingface.co/omelet-ai/OptiLIR-Qwen3-8B) (Qwen3-8B) and
  [`omelet-ai/OptiLIR-Gemma-4-12B`](https://huggingface.co/omelet-ai/OptiLIR-Gemma-4-12B) (Gemma 4 12B)

A user asks for an optimization in Korean, English or both, for example "minimize the total travel
distance, and if possible keep every route under 80 km". A small parser model writes what the user
meant as OptiLIR, one frame per line:

```text
MINIMIZE( SUM(x0 in routes: travel_distance(x0)) )
[SOFT] EACH(x1 in routes): travel_distance(x1) <= 80 km
```

A deterministic grounder then maps the IR onto the objectives and constraints that an optimization
server declares and returns one decision: `execute` (with refs, strengths, weights and values),
`clarify` (ask the user) or `unsupported` (decline); both carry a reason code. The model only writes
what the user meant; whether the request can run is decided by the grounder. Three domains are
covered: vehicle routing (`routing`), nurse scheduling (`nurse`) and retail inventory (`retail`).

This repository has:

| Path | What it is |
|---|---|
| `execution_llm/tier2/` | the IR parser and emitter, the IR normal form and line format (`parse_problem`, `emit_problem`, `ir_digest`), the decoder grammar (`grammar.py`) and the parser prompt (`prompt.py`) |
| `execution_llm/tier2/domains/` | the three domain vocabularies with their capability forms, one module per domain, and `SURFACES` |
| `execution_llm/grounding/` | the grounder (`grounder.py`, `ground_full`); it matches each frame against the domain's capability forms (`cover.py`) with the inputs it builds from the supplied registry (`inputs.py`), the nurse and retail catalog fixtures (`catalogs.py`) and the comparator's per-domain GradingSignature data (`signatures.py`) |
| `execution_llm/grounding/decide.py` | IR text to decision by the rule that made the dataset's `gold_decision` |
| `execution_llm/schema/` | the decision contract (`decision.py`) and the capability registry schema (`registry.py`) |
| `execution_llm/grading.py` | decision comparator and outcome classes used by the evaluation |
| `scripts/train_lora.py` | the LoRA recipe of the released adapters |
| `scripts/evaluate.py` | generation (optionally grammar-constrained) and scoring on a dataset split |
| `tests/` | offline tests; they pin the published prompts and grammars byte for byte |

## Terms

- **OptiLIR** (Optimization-oriented Language Intermediate Representation): the line-based language the
  parser model writes, used both for the capability forms in the prompt and for the model's output. The
  system prompt and the code (`execution_llm.tier2`) use its earlier name, Tier-2 IR; tier 1 was a flat
  record per capability (measure, aggregation, direction or comparator, unit) and is not part of this
  release. Below, "IR" means OptiLIR.
- **Frame**: one objective line (`MINIMIZE(...)`, `MAXIMIZE(...)`) or one constraint line (`EACH(...)`,
  `AND(...)`, a bare comparison such as a vehicle count, or `HOLE(constraint, family)`). `COMPOSE`,
  `UNEXPRESSED`, `NO_INTENT` and `CLARIFY(intent)` lines are not frames; the prompt's "one frame per line"
  means one item per line. The form a frame matches decides which capability it grounds to, not its first
  word: in routing, four capabilities the server declares as constraints are written as `MINIMIZE` frames.
- **Mark**: `[HARD]` (mandatory) or `[SOFT]` (preferential) at the start of a frame, written only when the
  user says so; otherwise the server's default applies. The code calls it `hardness`; decisions call it
  `strength`.
- **COMPOSE line**: how several objectives combine (`weighted_sum`, `lexicographic`, `product`). Without
  one, the server's default applies.
- **Hole**: a marker for something the user left unstated, so that the parser never guesses: `HOLE(value)`,
  `HOLE(measure, family, x)`, `HOLE(scope)`, `HOLE(constraint, family)`, and `UNSPECIFIED(...)` (a quantity
  named without its aggregation). A family is a group of measures or rules listed in the prompt. A hole
  leads to `clarify`. The prompt's "Holes" heading also lists `UNDECIDED`, which behaves differently.
- **UNDECIDED**: a bound the user wants but has not set a value for yet. The system does not ask: the bound
  is left out, the rest is grounded as usual, and a notice says so.
- **UNEXPRESSED(class, side)**: a clear requirement that the IR cannot express (class: what kind, such as a
  literal clock time or date). The request is `unsupported`, unless another line needs a clarification,
  which is asked first.
- **NO_INTENT** / **CLARIFY(intent)**: the whole output when there is no optimization request / when it
  does not determine what to optimize.
- **Vocabulary**: the domain names the prompt lists: populations (sets such as `routes`), measures
  (quantities such as `travel_distance`), accessors (per-item data such as `capacity`) and units. "Sort"
  there means value type. In code, a domain's vocabulary and capability forms together are its surface
  (`execution_llm.tier2.domains.SURFACES`).
- **Capability**: an objective or constraint the server declares. Its **capability form** is how it is
  written in IR, as listed in the prompt (class `Signature` in the code). Its **ref** is its id in
  decisions, such as `objective:distance_cost`. Refs never appear in the IR, and their names can differ from
  the IR's names.
- **Slot**: a value a capability takes, such as the 80 km of a distance limit; written `$name` in a capability
  form. The registry declares each slot's type, unit, default and bounds (its value policy).
- **Deferred**: a declared capability whose input data the server does not supply; a request for it is
  declined with `required_primitive_absent`.
- **Server**, **registry**: the optimization service of a domain, and the JSON file that lists what it
  declares. The registry behind `gold_decision` is published with the dataset as `registry.json`;
  decisions follow whichever registry you supply.
- **Grounder**: the deterministic code (`execution_llm.grounding`) that turns an IR into one decision
  against a registry.
- **Decision**: `execute` (run with these refs and values), `clarify` (ask the user) or `unsupported`
  (decline). Clarify and unsupported decisions carry a reason code; together they are called abstentions.
- **Fleet**: a stated vehicle count is returned beside the decision (`fleet`), not in it.
- **Gold**: a row's reference label: the gold IR (`ir`) and the gold decision (`gold_decision`: what
  `decide` returns for the gold IR and the row's user turns).
- **Stratum**: the recipe that produced a row. It records where the row came from, not its outcome.
- **Held-out capability**: upper bounds on route duration. Its form is in the prompt, but no training row
  uses it; the held-out test subset does (928 rows; 822 of them name `constraint:max_duration` in
  `gold_decision`).
- **Capability forms in the prompt** (`forms=True`): `system_prompt(domain, forms=True)` includes the
  capability forms, and the adapters were trained with it (each training row with its own shuffled order,
  `seed=<row id>`). The default, `forms=False`, leaves the forms out.
- **Parse rate**, **IR match**, **decision match**: the output is accepted by the IR parser; it equals the
  gold IR once both are in normal form; its grounded decision equals the gold decision.
- **Normal form**: values in base units (80 km = 80000 m); the quantity on the left of a comparison; an
  upper bound on a MAX (or a lower bound on a MIN) written as the same bound under EACH; AND items flattened, sorted and deduplicated;
  TOGETHER/OVERLAPS arguments and EACH chains in a fixed order; binders renumbered. For an IR match, frames
  must come in the same order with the same marks; COMPOSE, NO_INTENT and CLARIFY(intent) lines must agree;
  UNEXPRESSED lines are compared regardless of order.

## Install

Python 3.11 or newer.

```bash
git clone https://github.com/omelet-ai/OptiLIR
cd OptiLIR
python -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
pytest -q
```

The distribution is named `optilir`; the import name is `execution_llm`. The package itself needs
only Pydantic. Extras: `eval` (PyTorch, Transformers, Accelerate, PEFT, datasets, xgrammar) for
`scripts/evaluate.py`, `train` (PyTorch, Transformers, Accelerate, PEFT, TRL, datasets) for
`scripts/train_lora.py`. Generation (`scripts/evaluate.py` without `--predictions`) and training need a
CUDA GPU (the released adapters were trained and evaluated on 80 GB NVIDIA H100 and A100 GPUs);
scoring with `--predictions` runs on CPU.

## Prompts and grammars

The adapters expect the domain's system prompt byte for byte, capability forms included (`forms=True`,
see [Terms](#terms)). The dataset publishes it as `prompts/<domain>.txt`, and this package renders the
same text:

```python
from execution_llm.tier2.domains import SURFACES
from execution_llm.tier2.grammar import ebnf
from execution_llm.tier2.prompt import system_prompt

prompt = system_prompt("routing", forms=True)  # == prompts/routing.txt
s = SURFACES["routing"]
grammar = ebnf(s.vocab, s.spec.signatures, s.spec.plain_readings)  # == grammar/routing.ebnf
```

`ebnf` renders `grammar/<domain>.ebnf`, the EBNF used for grammar-constrained decoding, only with all
three arguments as above; given the vocabulary alone it returns a narrower grammar. `tests/test_core.py`
checks the SHA-256 of all six files.

During training the vocabulary listing in the prompt was shuffled per row, seeded by the row id:
`system_prompt(domain, seed=row["id"], forms=True)`. At inference, use the published prompt.

## Parse the model's output

```python
from execution_llm.tier2.domains import SURFACES
from execution_llm.tier2.emit import parse_problem

ir = "MINIMIZE( SUM(x0 in routes: travel_distance(x0)) )\n[SOFT] EACH(x1 in routes): travel_distance(x1) <= 80 km"
parsed = parse_problem(ir, SURFACES["routing"].vocab)  # ValueError if the text does not parse
```

The parser checks a name against the vocabulary where it is applied, as in `travel_distance(x0)`, or
where it names a population. A bare unknown name still parses, as a variable, and grounds to
`unsupported` (`semantic_not_declared`); the decoder grammar admits no unknown name.

## Ground the IR into a decision

Grounding reads a capability registry: what the server declares for each domain (objectives,
constraints, slots, units, catalogs, default weights). The package bundles and downloads none: it reads
the JSON file whose path is in `EXECUTION_LLM_REGISTRY` (not read from `.env`). The dataset repository
carries `registry.json`, the registry that produced its `gold_decision` values, with
`binding.projections` and `binding.required_paths` left empty in every domain (neither grounding nor
scoring uses them). With it, `decide` reproduces each row's `gold_decision` from its gold IR and user
turns, and `scripts/evaluate.py` reports the `decision_match` of the model cards:

```bash
export EXECUTION_LLM_REGISTRY="$(python -c 'from huggingface_hub import hf_hub_download
print(hf_hub_download("omelet-ai/OptiLIR-Instruct-v1", "registry.json", repo_type="dataset"))')"
```

Another registry works too; decisions then follow its declarations. The schema is
`execution_llm.schema.registry.Registry`: `schema_version` `"1"` and `domains`, keyed by
`routing`, `nurse` or `retail`; each entry has three projections (`execution_llm/schema/projections.py`):
`semantic` (the declared objectives, constraints, slots and units), `catalog` (the data entities and
their ids) and `binding` (whether and how the server executes each declared element). The semantic `server_id` must be `advanced-routing`,
`nurse-scheduling` or `retail-optimization` respectively. Catalog and binding must reference the exact
digest of the semantic projection. Missing configuration, unknown domains, undeclared units and
inconsistent references fail when the registry is loaded. `tests/test_core.py` builds a tiny example
registry. Defaults (a declared value, or an empty set meaning all members) are read from the registry's
slot declarations.

The sample below runs with `registry.json`. Another registry must declare every objective and
constraint the IR maps to, with a value policy (the slot's type, unit, default and bounds) for each of
their slots (here `constraint:max_distance`); the tiny test registry is not enough. Grounding raises
`GroundingError` (a `ValueError`) when the registry lacks one. Some well-formed IRs raise it too: two
frames with the same mark on one constraint, or a meaning the code recognizes but the registry does not
declare (such as a routing min-max objective under `registry.json`). Callers must handle it, as
`scripts/evaluate.py` does (`ground_error`).

```python
from execution_llm.grounding.decide import decide, user_text

# ir from the previous example
messages = [{"role": "user", "content": "총 이동거리를 줄여줘. 가능하면 경로마다 80km 이하로."}]
result = decide(ir, "routing", user_text(messages))
result["decision"]  # the decision as JSON, in the form of the dataset's gold_decision
result["fleet"]     # {"count": N, "use_all": bool} for a stated vehicle count, or None
result["notices"]   # codes of default readings the grounder applied: rank_weighted,
                    # stockout_objective, unit_misfit, undecided_bound
```

`decide` follows the rule used for `gold_decision`: a routing priority stated without numbers
("distance first, then time") executes as a 10:1 weighted sum unless the user states an explicit
"no trade-off" order, which is read from the user turns; called without them, `decide` leaves such a
priority unsupported. A non-null `fleet` means the user stated a vehicle count: whoever executes the
decision must give the solver only that many vehicles, or not run it.

An `execute` decision is a structured representation of the request, not a solver call; this
repository contains no solver client.

## Evaluate an adapter

```bash
pip install -e '.[eval]'
# for decision_match, set EXECUTION_LLM_REGISTRY first (see "Ground the IR into a decision")

# Qwen3-8B adapter
python scripts/evaluate.py --dataset omelet-ai/OptiLIR-Instruct-v1 --split test \
  --adapter omelet-ai/OptiLIR-Qwen3-8B --base Qwen/Qwen3-8B --grammar --shuffle

# Gemma 4 12B adapter
python scripts/evaluate.py --dataset omelet-ai/OptiLIR-Instruct-v1 --split test \
  --adapter omelet-ai/OptiLIR-Gemma-4-12B --base google/gemma-4-12B-it \
  --base-class Gemma4UnifiedForConditionalGeneration --attn eager --grammar --shuffle
```

These are the settings behind the metrics on the model cards: greedy decoding constrained by the
domain grammar, and `--shuffle`, the training-time prompt rendering (without it the prompt is the
published one, as at inference). Gemma 4 needs `--attn eager`: under SDPA attention it decodes
left-padded batches wrongly (seen with Transformers 5.16), and the Gemma card's metrics were computed
with eager attention.

Tested versions: both adapters were trained, and their card outputs generated, with PyTorch 2.9.1
(CUDA 12.8 build), Transformers 5.16.1, PEFT 0.20.0, TRL 1.12.0, Accelerate 1.14.0, Datasets 5.0.1
and xgrammar 0.2.3 (Tokenizers 0.23.1 for Gemma 4, 0.23.3 for Qwen3-8B). The extras set lower bounds
only, so a fresh install gets newer, untested versions (current PyTorch wheels default to CUDA 13,
which needs a recent NVIDIA driver).

`--predictions FILE` scores outputs produced elsewhere (for example with vLLM, as
shown on the model cards) instead of generating; each line is `{"id": ..., "output": ...}`. Scoring
alone needs only `pip install -e . datasets`.

Per row the script reports `parse` (the IR parser in this package accepts the output; `parse_rate` in
`metrics.json`, Parse rate on the cards) and `ir_match` (the output equals the gold IR once both are in
normal form, frame order and marks included; see Normal form under [Terms](#terms)). These two need only
the dataset. With `EXECUTION_LLM_REGISTRY` set it also reports `decision_match` (the grounded decision
equals `gold_decision`) and an outcome class: `success` (the decision equals gold); `mis_execution`
(executes, but gold does not execute or executes differently); `unnecessary_clarify` (clarify where gold
executes); `refusal` (unsupported where gold executes); `abstention_arm_confusion` (clarify where gold is
unsupported, or the reverse); `abstention_payload_defect` (the right kind, but a different reason code,
candidates, required inputs or related refs); `execute_form_defect` (equal to gold only once a soft
constraint is moved onto its paired objective; it does not occur with the registry behind
`gold_decision`); `rejected` (the decision fails the schema check, names a ref, slot or value the
registry lacks, or leaves a required slot empty); or `parse_error` / `ground_error` (an output that
cannot be parsed or grounded) / `gold_rejected` (the supplied registry rejects the row's
`gold_decision`). Such a row never stops the run; a registry that does not load stops it before
generation. With the dataset's `registry.json` these match the model cards; with another registry they
follow its declarations. The raw generations are written to `eval_<split>/outputs.jsonl` before scoring
(rescore them with `--predictions`); the scored rows go to `predictions.jsonl` and the totals to
`metrics.json`.

## Train an adapter

```bash
pip install -e '.[train]'

# Qwen3-8B on 2 GPUs (2 per device x ACCUM 4 x 2 GPUs = 16, the released adapter's effective batch)
DATASET=omelet-ai/OptiLIR-Instruct-v1 OUT=runs/qwen3-8b \
  torchrun --nproc_per_node=2 scripts/train_lora.py

# Gemma 4 12B on 3 GPUs (2 per device x ACCUM 3 x 3 GPUs = 18, as the released adapter)
DATASET=omelet-ai/OptiLIR-Instruct-v1 OUT=runs/gemma-4-12b ACCUM=3 \
  BASE_MODEL=google/gemma-4-12B-it BASE_CLASS=Gemma4UnifiedForConditionalGeneration LOSS_TYPE=nll \
  TARGET_MODULES='^.*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$' \
  torchrun --nproc_per_node=3 scripts/train_lora.py
```

LoRA rank 32, alpha 64, dropout 0.05 on the attention and MLP projections; 3 epochs, learning rate
1e-4 with a cosine schedule and 10 warmup steps, bf16, maximum length 4096 tokens, seed 41; the loss is
computed on the IR tokens only. The adapter is saved to `OUT/final`. See the docstring of
`scripts/train_lora.py` (or `python scripts/train_lora.py --help`) for every setting, and "Tested
versions" above for the library versions.

## Scope and limitations

- Three domains with fixed vocabularies. Other domains need their own vocabulary, capability forms and
  prompts (`execution_llm/tier2/domains/routing.py` shows the shape) and a parser trained on them.
- Decisions depend on the registry: `gold_decision` and the cards' `decision_match` hold for the
  dataset's `registry.json`, and change with another registry's declarations.
- The grounder reads only the IR (and the user's text for the routing priority rule). A wrong IR
  grounds to a wrong decision; grammar-constrained decoding fixes syntax only.
- A few names come from the project's internal design records and are kept as they are: the
  rewrite-rule labels NF-2, NF-3, NF-7, EN-2, EN-3, EN-4 and EN-6 in `tier2/expr.py` (defined in its
  module docstring), the comparator's checks O1 to O3 in `grading.py` (schema; refs and required
  slots; byte equality) and `SamplerInputs` in `grounding/inputs.py`, named after the corpus sampler
  that built the dataset.

## License

The code in this repository is under the Apache License 2.0; see `LICENSE`. The dataset is under
CC BY 4.0 and the adapters are under the Apache License 2.0. An adapter runs on its base model, whose
own license also applies.
