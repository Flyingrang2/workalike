"""Load the base model and a saved adapter in a fresh process and compare first moves with and without it.

    python scripts/check_adapter.py --base <Muse-Glimmer-30B dir> --adapter adapters/a \
        --tasks data/h1/tasks/dev.jsonl --n 3 [--bf16]

For each ticket it renders the prompt the agent loop builds and generates the first
assistant turn twice, with the adapter on and off. An A adapter should start with
" to=get_cost" or " to=check_stock", a B adapter with " to=send_reply"; if both
outputs are identical, the adapter is not being applied. Use dev tickets here, not
the eval splits. --bf16 loads the base unquantized (needs about 64 GB of VRAM), to
separate a precision effect from a training effect. This is a reload check, not the
evaluation; the evaluation goes through a server and island.run_agent.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workalike.desk import TOOL_SPECS  # noqa: E402
from workalike.serialize import head_messages  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()

    quant = None if args.bf16 else BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_storage=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(args.base)
    base = AutoModelForImageTextToText.from_pretrained(args.base, quantization_config=quant, dtype=torch.bfloat16,
                                                       device_map="auto")
    model = PeftModel.from_pretrained(base, args.adapter)
    lora = [n for n, _ in model.named_modules() if n.endswith("lora_A")]
    print(f"adapter applied to {len(lora)} modules", flush=True)
    model.eval()
    with open(args.tasks) as f:
        tasks = [json.loads(line) for line in f][: args.n]

    def first_move(ids):
        out = model.generate(**ids, max_new_tokens=args.max_new_tokens, do_sample=False, eos_token_id=[200001, 200008])
        return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=False)

    for task in tasks:
        prompt = tok.apply_chat_template(head_messages(task["view"]), tools=TOOL_SPECS, tokenize=False,
                                         add_generation_prompt=True)
        ids = tok(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
        tuned = first_move(ids)
        with model.disable_adapter():
            plain = first_move(ids)
        print(f"--- {task['task_id']} ({task['family']})\nadapter: {tuned[:400]}\nbase:    {plain[:400]}\n"
              f"same: {tuned == plain}\n", flush=True)


if __name__ == "__main__":
    main()
