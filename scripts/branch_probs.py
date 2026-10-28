"""Branch probabilities under the training stack: where worker C made a conditional choice in a training row, how
likely is C's next step against the routine one, for base Glimmer and the adapters, loaded in NF4 as in training.

    python scripts/branch_probs.py --base <model dir> --rows data/v2/sft/c-s1-train.jsonl \
        --tasks data/v2/tasks/train.jsonl --adapter fA60=adapters/v2-fA60 --adapter fB60=adapters/v2-fB60 --out x.json

Positions, each with C's step (A) and the routine alternative (B):
  ask           C writes to the customer straight after check_stock (missing date, item code to confirm, short
                stock with the preference unknown); B is get_cost on the item C just checked
  second_quote  C makes a second quote (mixed and all-substitute) before replying; B is C's own reply with the
                first quote attached, sent at once
For each model the continuation of A and of B is scored after the same prompt (the chat template's generation
prompt, as at inference). Reported: the log-probability of each at the first token where they differ (the choice
itself), whether A wins there, and the summed log-probability of each whole step. If A wins here but the FP8
server takes B (island.next_action), serving loses what training put in; if B wins here, it was not learned.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workalike.render import load_tokenizer, render  # noqa: E402
from workalike.sft import read_jsonl  # noqa: E402

spec = importlib.util.spec_from_file_location("train_lora", ROOT / "scripts" / "train_lora.py")
train_lora = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_lora)


def ticket_of(row) -> str:
    return row["messages"][1]["content"].split("Ticket ")[1].split(" ")[0]


def call_of(message) -> dict | None:
    calls = message.get("tool_calls") or []
    return calls[0]["function"] if calls else None


def tool_msg(name, arguments):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"id": "x1", "type": "function", "function": {"name": name, "arguments": arguments}}]}


def positions(row, family: str):
    msgs = row["messages"]
    calls = []  # (index, function) of earlier assistant calls
    for k, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        f = call_of(m)
        if f is None:
            continue
        prev = calls[-1][1] if calls else None
        sent_quote = any(c["name"] == "send_reply" and c["arguments"].get("quote_id") for _, c in calls)
        if (f["name"] == "send_reply" and not f["arguments"].get("quote_id") and not sent_quote
                and prev and prev["name"] == "check_stock"):
            yield {"kind": "ask", "k": k, "family": family, "A": m,
                   "B": tool_msg("get_cost", {"sku": prev["arguments"]["sku"]})}
        if f["name"] == "create_quote" and not sent_quote and any(c["name"] == "create_quote" for _, c in calls):
            nxt = next((n for n in msgs[k + 1:] if n["role"] == "assistant" and call_of(n)), None)
            if nxt is not None and call_of(nxt)["name"] == "send_reply":
                # C's own reply (it attaches the first quote), sent without making the second one
                yield {"kind": "second_quote", "k": k, "family": family, "A": m, "B": json.loads(json.dumps(nxt))}
        calls.append((k, f))


def continuation(tok, messages, tools, k, step) -> tuple[list[int], list[int]]:
    prompt = render(tok, messages[:k], tools, add_generation_prompt=True)
    full = render(tok, messages[:k] + [step], tools)
    if not full.startswith(prompt):
        raise ValueError("the step does not extend the generation prompt")
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    f = tok(full, add_special_tokens=False)["input_ids"]
    if f[: len(p)] != p:
        raise ValueError("tokenization of the prompt changes inside the full text")
    return p, f[len(p):]


@torch.no_grad()
def scores(model, prompt: list[int], cont: list[int]) -> list[float]:
    ids = torch.tensor([prompt + cont], device=model.device)
    logits = model(input_ids=ids).logits[0, len(prompt) - 1: len(prompt) + len(cont) - 1].float()
    lp = logits.log_softmax(-1)
    return [lp[i, t].item() for i, t in enumerate(cont)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("--rows", required=True, help="training rows (sft jsonl with messages and tools)")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--adapter", action="append", default=[], help="name=dir, repeatable")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    families = {t["view"]["ticket_id"]: t["family"] for t in read_jsonl(args.tasks)}
    tok = load_tokenizer(args.base)
    rows = read_jsonl(args.rows)
    todo = [(r, pos) for r in rows for pos in positions(r, families[ticket_of(r)])]
    model = train_lora.load_base(args.base, causal=False)
    model.eval()
    adapters = [a.split("=", 1) for a in args.adapter]
    if adapters:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapters[0][1], adapter_name=adapters[0][0])
        for name, path in adapters[1:]:
            model.load_adapter(path, adapter_name=name)
    results = []
    for name in ["base"] + [a for a, _ in adapters]:
        for row, pos in todo:
            pa, ca = continuation(tok, row["messages"], row["tools"], pos["k"], pos["A"])
            pb, cb = continuation(tok, row["messages"], row["tools"], pos["k"], pos["B"])
            if name == "base" and adapters:
                with model.disable_adapter():
                    la, lb = scores(model, pa, ca), scores(model, pb, cb)
            else:
                if adapters:
                    model.set_adapter(name)
                la, lb = scores(model, pa, ca), scores(model, pb, cb)
            d = next((i for i, (x, y) in enumerate(zip(ca, cb)) if x != y), min(len(ca), len(cb)) - 1)
            results.append({"model": name, "kind": pos["kind"], "family": pos["family"], "ticket": ticket_of(row),
                            "A_token": tok.decode([ca[d]]), "B_token": tok.decode([cb[d]]),
                            "lp_A_at_choice": round(la[d], 3), "lp_B_at_choice": round(lb[d], 3),
                            "A_wins": la[d] > lb[d], "lp_A_step": round(sum(la), 2), "lp_B_step": round(sum(lb), 2)})
            print(json.dumps(results[-1]), flush=True)
    summary = {}
    for r in results:
        key = f"{r['model']} {r['kind']}"
        s = summary.setdefault(key, {"n": 0, "A_wins": 0, "by_family": {}})
        s["n"] += 1
        s["A_wins"] += r["A_wins"]
        fam = s["by_family"].setdefault(r["family"], [0, 0])
        fam[0] += r["A_wins"]
        fam[1] += 1
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "results": results}, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
