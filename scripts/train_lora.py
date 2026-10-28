"""LoRA SFT of a parser adapter on the IR dataset, with the loss on the IR tokens only.

Each example is the domain's system prompt, the conversation (`messages`), then the gold `ir` as the
completion. With SHUFFLE=1 (default, as the released adapters were trained) the prompt's vocabulary
listing is shuffled per row, seeded by the row id; with SHUFFLE=0 it is `prompts/<domain>.txt` as published.

  DATASET=<org>/<dataset> OUT=runs/qwen3-8b torchrun --nproc_per_node=2 scripts/train_lora.py

Environment (defaults are the settings of the released Qwen3-8B adapter, laid out for 2 GPUs; effective
batch 16 = 2 per device x ACCUM x GPUs):
  DATASET         hub id or local folder of the dataset (splits train and validation)
  OUT             output folder; the adapter is saved to OUT/final
  BASE_MODEL      default Qwen/Qwen3-8B
  BASE_CLASS      a transformers class name to load the base with, for multimodal checkpoints
                  (Gemma 4: Gemma4UnifiedForConditionalGeneration)
  TARGET_MODULES  comma-separated module-name suffixes, or a PEFT regex starting with "^"
  LOSS_TYPE       TRL loss type; "nll" for Gemma 4, whose final logit softcapping the default
                  chunked loss skips
  EPOCHS=3 ACCUM=4 MAX_LEN=4096 EVAL_STEPS=40 SHUFFLE=1 REPORT_TO=none
  SMOKE           any non-empty value runs a quick check: 32 training and 8 validation rows,
                  1 epoch, no gradient accumulation, evaluation every 5 steps
"""

import os
import sys
from pathlib import Path

if {"-h", "--help"} & set(sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

from execution_llm.tier2.prompt import system_prompt

DATASET = os.environ["DATASET"]
OUT = Path(os.environ.get("OUT", "runs/adapter"))
MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3-8B")
BASE_CLASS = os.environ.get("BASE_CLASS")
TARGET_MODULES = os.environ.get("TARGET_MODULES", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
TARGET_MODULES = TARGET_MODULES if TARGET_MODULES.startswith("^") else TARGET_MODULES.split(",")
MAX_LEN = int(os.environ.get("MAX_LEN", "4096"))
EPOCHS = float(os.environ.get("EPOCHS", "3"))
ACCUM = int(os.environ.get("ACCUM", "4"))
EVAL_STEPS = int(os.environ.get("EVAL_STEPS", "40"))  # validation loss for monitoring; it never picks the adapter
SHUFFLE = os.environ.get("SHUFFLE", "1") == "1"
SMOKE = bool(os.environ.get("SMOKE"))

tokenizer = AutoTokenizer.from_pretrained(MODEL)


def to_prompt_completion(row):
    system = system_prompt(row["domain"], seed=row["id"] if SHUFFLE else None, forms=True)
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, *row["messages"]],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return {"prompt": prompt, "completion": row["ir"]}


def main():
    data = load_dataset(DATASET)
    data = {split: data[split].map(to_prompt_completion, remove_columns=data[split].column_names)
            for split in ("train", "validation")}
    if SMOKE:
        data = {"train": data["train"].select(range(32)), "validation": data["validation"].select(range(8))}
    peft_cfg = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    cfg = SFTConfig(
        output_dir=str(OUT),
        run_name=OUT.name,
        model_init_kwargs={"dtype": "bfloat16", "attn_implementation": "sdpa"},
        max_length=MAX_LEN,
        num_train_epochs=1 if SMOKE else EPOCHS,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1 if SMOKE else ACCUM,
        gradient_checkpointing=True,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        bf16=True,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=5 if SMOKE else EVAL_STEPS,
        per_device_eval_batch_size=2,
        save_strategy="steps",
        save_steps=40,
        save_total_limit=2,
        report_to=os.environ.get("REPORT_TO", "none"),
        completion_only_loss=True,
        seed=41,
        **({"loss_type": os.environ["LOSS_TYPE"]} if os.environ.get("LOSS_TYPE") else {}),
    )
    model = MODEL
    extra = {}
    if BASE_CLASS:
        import torch
        import transformers

        model = getattr(transformers, BASE_CLASS).from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa")
        extra["processing_class"] = tokenizer  # the tokenizer, not the multimodal processor
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        peft_config=peft_cfg,
        **extra,
    )
    over = sum(1 for ex in trainer.train_dataset if len(ex["input_ids"]) >= MAX_LEN)
    if over:
        raise SystemExit(f"{over} rows at or over MAX_LEN={MAX_LEN}; raise MAX_LEN")
    trainer.train()
    trainer.save_model(str(OUT / "final"))
    tokenizer.save_pretrained(str(OUT / "final"))
    print(f"adapter saved to {OUT / 'final'}")


if __name__ == "__main__":
    main()
