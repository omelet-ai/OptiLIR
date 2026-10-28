"""Evaluate a parser adapter on one split of the IR dataset: generate the IR, then score it.

  python scripts/evaluate.py --dataset <org>/<dataset> --split test --adapter <org>/<adapter> --grammar
  python scripts/evaluate.py --dataset <org>/<dataset> --split test --predictions preds.jsonl  # score only

Generation is greedy with thinking disabled, as in training. --grammar constrains decoding with the
domain's EBNF through xgrammar. --shuffle renders each prompt as in training (vocabulary listing
shuffled per row, seeded by the row id); without it the prompt is `prompts/<domain>.txt` as published.
--predictions scores outputs made elsewhere (e.g. with vLLM): JSONL rows {"id": ..., "output": ...}.
For Gemma 4 pass --attn eager: under SDPA attention it decodes left-padded batches wrongly.

Per row: parse (the IR parser in this package accepts the output) and ir_match (equal to the gold IR
once both are in normal form: values in base units, so 80 km = 80000 m; the quantity on the left of a
comparison; an upper bound on a MAX (or a lower bound on a MIN) written as the same bound under EACH;
AND items flattened, sorted and deduplicated; TOGETHER/OVERLAPS arguments and EACH chains in a fixed
order; binders renumbered. Frames must come in the same order with the same marks; the COMPOSE,
NO_INTENT and CLARIFY(intent) lines must agree; UNEXPRESSED lines are compared regardless of order).
When EXECUTION_LLM_REGISTRY names a capability registry, also decision_match (the grounded decision
equals `gold_decision`) and the outcome class; a registry that does not load stops the run before
generation. A row whose output cannot be parsed or grounded is scored as parse_error or ground_error,
and one whose `gold_decision` the registry rejects as gold_rejected. Writes the raw generations to
OUT/outputs.jsonl (the --predictions format) before scoring, then OUT/predictions.jsonl and OUT/metrics.json.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from execution_llm.tier2.domains import SURFACES
from execution_llm.tier2.emit import ir_digest, parse_problem
from execution_llm.tier2.grammar import ebnf
from execution_llm.tier2.prompt import system_prompt


def generate(rows, args) -> list[str]:
    import torch
    import transformers
    from peft import PeftModel

    tok = transformers.AutoTokenizer.from_pretrained(args.base, padding_side="left")
    model_class = getattr(transformers, args.base_class) if args.base_class else transformers.AutoModelForCausalLM
    model = model_class.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map="auto", attn_implementation=args.attn
    )
    model = PeftModel.from_pretrained(model, args.adapter).eval()
    compiled = {}
    if args.grammar:
        import xgrammar as xgr

        config = model.config
        vocab_size = getattr(config, "vocab_size", None) or config.get_text_config().vocab_size  # multimodal configs
        compiler = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tok, vocab_size=vocab_size))
        compiled = {
            d: compiler.compile_grammar(ebnf(s.vocab, s.spec.signatures, s.spec.plain_readings))
            for d, s in SURFACES.items()
        }
    outs: list[str] = []
    for b in range(0, len(rows), args.batch):
        chunk = rows[b : b + args.batch]
        extra = {}
        if args.grammar:  # one matcher per sequence, fresh per generate call
            from transformers import LogitsProcessorList
            from xgrammar.contrib.hf import LogitsProcessor

            extra["logits_processor"] = LogitsProcessorList([LogitsProcessor([compiled[r["domain"]] for r in chunk])])
        prompts = [
            tok.apply_chat_template(
                [{"role": "system", "content": system_prompt(r["domain"], seed=r["id"] if args.shuffle else None, forms=True)},
                 *r["messages"]],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for r in chunk
        ]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                **extra,
            )
        outs += [tok.decode(seq[enc.input_ids.shape[1] :], skip_special_tokens=True) for seq in gen]
        print(f"  generated {len(outs)}/{len(rows)}", flush=True)
    return outs


def jcs(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def check_registry(rows) -> None:
    """With EXECUTION_LLM_REGISTRY set, build each domain's grounding inputs once, before generation and
    scoring, so a wrong path or a malformed registry stops the run instead of failing every row."""
    if not os.environ.get("EXECUTION_LLM_REGISTRY"):
        return
    from execution_llm.grounding.decide import domain_inputs

    for domain in sorted({row["domain"] for row in rows}):
        try:
            domain_inputs(domain)
        except (OSError, ValueError) as error:
            raise SystemExit(f"EXECUTION_LLM_REGISTRY: {type(error).__name__}: {error}") from error


def score(rows, outputs) -> list[dict]:
    grounded = bool(os.environ.get("EXECUTION_LLM_REGISTRY"))
    if grounded:
        from pydantic import TypeAdapter

        from execution_llm.grading import grade_turn
        from execution_llm.grounding.decide import decide, domain_inputs, user_text
        from execution_llm.schema.decision import DecisionBody

        bodies = TypeAdapter(DecisionBody)
    results = []
    for row, raw in zip(rows, outputs, strict=True):
        domain, vocab = row["domain"], SURFACES[row["domain"]].vocab
        text = raw.strip()
        if "</think>" in text:
            text = text.split("</think>", 1)[1].strip()
        rec = {
            "id": row["id"],
            "domain": domain,
            "decision_kind": row["decision_kind"],
            "stratum": row["stratum"],
            "language": row["language"],
            "output": raw,
            "gold_ir": row["ir"],
            "parse": False,
            "ir_match": False,
            "decision_match": False if grounded else None,
            "outcome": None,
        }
        results.append(rec)
        # Any failure on one row is its parse_error, ground_error or gold_rejected, never the end of the run.
        try:
            parsed = parse_problem(text, vocab)
            rec["parse"] = True
            rec["ir_match"] = ir_digest(parsed, vocab) == ir_digest(parse_problem(row["ir"], vocab), vocab)
        except Exception as error:
            rec["outcome"], rec["error"] = "parse_error", f"{type(error).__name__}: {error}"[:200]
            continue
        if not grounded:
            continue
        try:
            decision = decide(text, domain, user_text(row["messages"]))["decision"]
        except Exception as error:  # GroundingError (a ValueError), or anything else this output trips
            rec["outcome"], rec["error"] = "ground_error", f"{type(error).__name__}: {error}"[:200]
            continue
        rec["decision"] = decision
        rec["decision_match"] = jcs(decision) == jcs(row["gold_decision"])
        inputs = domain_inputs(domain)
        try:
            gold = bodies.validate_python(row["gold_decision"])
            grade = grade_turn(jcs(decision).encode(), gold, inputs.snapshot, inputs.signature, domain=domain)
            rec["outcome"] = grade.outcome
        except ValueError as error:  # the supplied registry rejects this row's gold_decision
            rec["outcome"], rec["error"] = "gold_rejected", f"{type(error).__name__}: {error}"[:200]
    return results


def summarize(results) -> dict:
    def block(items):
        n = len(items)
        rate = lambda key: round(sum(bool(x[key]) for x in items) / n, 4) if n else None  # noqa: E731
        out = {"n": n, "parse_rate": rate("parse"), "ir_match": rate("ir_match")}
        if items and items[0]["decision_match"] is not None:
            out["decision_match"] = rate("decision_match")
        out["outcomes"] = dict(Counter(x["outcome"] for x in items if x["outcome"]))
        return out

    summary = {"all": block(results)}
    for key in ("domain", "decision_kind", "stratum", "language"):
        groups = defaultdict(list)
        for x in results:
            groups[x[key]].append(x)
        summary[f"by_{key}"] = {k: block(v) for k, v in sorted(groups.items())}
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="hub id or local folder of the dataset")
    ap.add_argument("--split", default="test")
    ap.add_argument("--adapter", help="hub id or local folder of the LoRA adapter")
    ap.add_argument("--base", default="Qwen/Qwen3-8B", help="base model of the adapter")
    ap.add_argument("--base-class", help="transformers class for the base (Gemma 4: Gemma4UnifiedForConditionalGeneration)")
    ap.add_argument("--attn", help="attention implementation of the base (Gemma 4: eager)")
    ap.add_argument("--predictions", type=Path, help="score these outputs instead of generating")
    ap.add_argument("--grammar", action="store_true", help="grammar-constrained decoding (xgrammar)")
    ap.add_argument("--shuffle", action="store_true", help="training-time prompt rendering (seeded by row id)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--limit", type=int, default=0, help="first N rows only")
    ap.add_argument("--out", type=Path, help="output folder (default eval_<split>)")
    args = ap.parse_args()
    if not (args.adapter or args.predictions):
        ap.error("give --adapter (generate) or --predictions (score only)")

    from datasets import load_dataset

    rows = list(load_dataset(args.dataset, split=args.split))
    if args.limit:
        rows = rows[: args.limit]
    check_registry(rows)
    if args.predictions:
        by_id = {}
        for line in args.predictions.read_text(encoding="utf-8").splitlines():
            if line.strip():
                pred = json.loads(line)
                by_id[pred["id"]] = pred["output"]
        missing = [r["id"] for r in rows if r["id"] not in by_id]
        if missing:
            raise SystemExit(f"{len(missing)} rows have no prediction, e.g. {missing[0]}")
        outputs = [by_id[r["id"]] for r in rows]
    else:
        outputs = generate(rows, args)
    out = args.out or Path(f"eval_{args.split}")
    out.mkdir(parents=True, exist_ok=True)
    if not args.predictions:  # kept even if scoring fails: rescore with --predictions OUT/outputs.jsonl
        with (out / "outputs.jsonl").open("w", encoding="utf-8") as f:
            for row, output in zip(rows, outputs, strict=True):
                f.write(json.dumps({"id": row["id"], "output": output}, ensure_ascii=False) + "\n")
    results = score(rows, outputs)
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    metrics = summarize(results)
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"split": args.split, **metrics["all"]}, ensure_ascii=False))
    for domain, block in metrics["by_domain"].items():
        print("  ", domain, json.dumps(block, ensure_ascii=False))


if __name__ == "__main__":
    main()
