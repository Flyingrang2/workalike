"""QLoRA on Muse Glimmer from pre-tokenized rows (sft/*.tok.jsonl from render_sft.py).

    python scripts/train_lora.py --base <Muse-Glimmer-30B dir> --data data/h1/sft/a-train.tok.jsonl --out adapters/a
    python scripts/train_lora.py ... --out adapters/smoke --max-steps 1 --accum 1    # first smoke run

The rows already carry labels (loss only on the model's own turns), so this is a
plain Trainer with a padding collator; no chat template is applied here. Loading
follows the Glimmer post-training guide: 4-bit NF4 base with bf16 compute and bf16
quant storage. The checkpoint's own class (MuseGlimmerForConditionalGeneration) is
loaded so adapter names match what servers read; adapters go on the language
model's attention and MLP projections, never the vision tower. The matched module
names, library versions, arguments, trained tokens and steps are saved next to the adapter.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
from pathlib import Path

# On a 48 GB card the run sits near the limit; fragmentation alone can end it with an out-of-memory error.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

PAD_ID = 200018  # <|finetune_right_pad|>, pad_token_id in generation_config.json
LANGUAGE_PROJECTIONS = (
    r"^(?!.*(vision|visual|image|patch|mm_|projector)).*\."
    r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
)


class Rows(torch.utils.data.Dataset):
    def __init__(self, path, max_len):
        with open(path) as f:
            self.rows = [json.loads(line) for line in f if line.strip()]
        too_long = [len(r["input_ids"]) for r in self.rows if len(r["input_ids"]) > max_len]
        if too_long:
            raise ValueError(f"{len(too_long)} rows exceed --max-len {max_len} (longest {max(too_long)})")
        if any(all(l == -100 for l in r["labels"]) for r in self.rows):
            raise ValueError("a row has no trained tokens")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def collate(batch):
    n = max(len(r["input_ids"]) for r in batch)
    pad = lambda seq, v: seq + [v] * (n - len(seq))
    return {
        "input_ids": torch.tensor([pad(r["input_ids"], PAD_ID) for r in batch]),
        "attention_mask": torch.tensor([pad(r["attention_mask"], 0) for r in batch]),
        "labels": torch.tensor([pad(r["labels"], -100) for r in batch]),
    }


def load_base(path, causal: bool):
    gen = json.loads((Path(path) / "generation_config.json").read_text())
    assert gen.get("pad_token_id") == PAD_ID, f"pad_token_id is {gen.get('pad_token_id')}, expected {PAD_ID}"
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
    )
    cls = AutoModelForCausalLM if causal else AutoModelForImageTextToText
    model = cls.from_pretrained(path, quantization_config=bnb, dtype=torch.bfloat16, device_map="auto")
    placed = set(getattr(model, "hf_device_map", {}).values())
    if placed & {"cpu", "disk"}:
        raise SystemExit(f"part of the model was offloaded ({placed}); use a larger GPU instead of silent offload")
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--max-steps", type=int, default=-1, help="for smoke runs")
    ap.add_argument("--limit", type=int, help="train on the first N rows only (overfit checks, data-size curves)")
    ap.add_argument(
        "--seed", type=int, default=42, help="Trainer seed (42 is the Trainer default the first experiments used)"
    )
    ap.add_argument("--target", default=LANGUAGE_PROJECTIONS)
    ap.add_argument("--causal", action="store_true", help="load with AutoModelForCausalLM instead")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = load_base(args.base, args.causal)
    targets = [n for n, _ in model.named_modules() if re.match(args.target, n)]
    print(f"{type(model).__name__}: LoRA on {len(targets)} modules, e.g. {targets[:2]}", flush=True)
    if not targets:
        raise SystemExit("no modules matched --target")
    (out / "lora_modules.txt").write_text("\n".join(targets) + "\n")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, target_modules=args.target, task_type="CAUSAL_LM"
        ),
    )
    model.print_trainable_parameters()

    data = Rows(args.data, args.max_len)
    if args.limit:
        data.rows = data.rows[: args.limit]
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out),
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_steps=0.05,
            bf16=True,
            logging_steps=1,
            save_strategy="epoch",
            save_total_limit=1,
            save_only_model=True,
            report_to=[],
            gradient_checkpointing=True,
            remove_unused_columns=False,
            seed=args.seed,
            data_seed=args.seed,
        ),
        train_dataset=data,
        data_collator=collate,
    )
    trainer.train()
    model.save_pretrained(out)
    versions = {
        p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft", "bitsandbytes", "accelerate")
    }
    trained = sum(sum(l != -100 for l in r["labels"]) for r in data.rows)
    (out / "train_args.json").write_text(
        json.dumps(
            {
                **vars(args),
                "versions": versions,
                "rows": len(data),
                "trained_tokens_per_epoch": trained,
                "optimizer_steps": trainer.state.global_step,
                "log": trainer.state.log_history,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
