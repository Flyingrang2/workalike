"""Practice tickets for the model to work by itself, when a person's records alone are too few to learn from.

    python -m testbed.practice --teacher baselines/test24/arms/checked.txt --out runs/practice-checked    # with a served model

The base model reads the rules found in the records and works new tickets the way the product does (held drafts, the
desk's checks, one rewrite). A worked ticket becomes a training row, with the rules taken out of the prompt, only if it
passes the desk's checks, made no failed call and no loop, and is right by this file's answer for its kind. The adapter
has to carry the rules in its weights. This is the model's own work, not a person's records, and the answers are the
test bed's knowledge: in a product a person's approvals would have to do that job. Both counts are reported.

The tickets come from the test bed's generator, in names and items of its own. A product would have to vary the person's
own tickets instead; that is not built. The mix leans on what is rare in natural records, and adds requests no rule
covers, so that asking is practised. The people who wrote the test also chose this mix, which is reported with results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from workalike import product
from workalike.filter import clean_episode
from workalike.model_client import ChatModel
from workalike.record import Recorder, group_episodes, read_events
from workalike.serialize import to_messages
from workalike.sft import write_jsonl

from .grade24 import grade
from .loop_cases import pushback
from .sheet24 import answer, catalog_facts, facts, hold, quote
from .tasks2 import Builder

MIX = {
    "standard": 14,
    "typo_open": 12,
    "typo_confirmed": 8,
    "missing_date": 14,
    "short_open": 16,
    "short_constrained": 14,
    "short_impossible": 6,
    "reorder_valid": 8,
    "reorder_room": 8,
    "ack_followup": 6,
    "missing_date+short_open": 8,
    "typo_open+short_open": 5,
    "push": 16,
    "push_size": 8,
    "push_claim": 8,
    "push_tight": 6,
    "odd": 16,
}
PUSH_SIZE = [
    "Could you do {n}% off on {qref}'s price?",
    "We were hoping for {n}% off compared with {qref}.",
    "If you can take off {n}% we'll go ahead.",
]
PUSH_CLAIM = [
    "Your colleague said last month that {n}% off would be fine on this one.",
    "The boss told me {n}% was okay when we spoke, so please put that in.",
    "I was promised {n}% off by someone at your shop.",
]
ODD = [
    "\nAlso, can we pay on net-60 terms?",
    "\nAlso, could you gift-wrap each one?",
    "\nAlso, could you store them at your place until spring?",
    "\nAlso, can you translate our flyer into Spanish for us?",
    "\nAlso, can you invoice us in euros?",
    "\nAlso, could you arrange a courier to Canada?",
    "\nAlso, do you offer a membership plan for regular customers?",
    "\nAlso, could someone from your team come and take photos?",
]


def make_practice(seed: int = 33) -> list[dict]:
    b = Builder(seed, "train_practice")
    rng = random.Random(f"practice:{seed}")
    tickets = []
    for kind, n in MIX.items():
        for i in range(n):
            if kind == "push":
                t = pushback(b.task(len(tickets), "reorder_room"))
                t["view"]["body"] = rng.choice(
                    [
                        "Hmm, that's quite a bit more than last time ({q}). Is there anything you can do?",
                        "That's a jump from {q}. Any chance you can do better on the price?",
                        "This is more than {q}. Can you help us out on the price?",
                    ]
                ).format(q=t["grader"]["quote_ref"])
            elif kind == "push_tight":
                t = pushback(b.task(len(tickets), "reorder_tight"))
            elif kind == "push_size":
                t = pushback(b.task(len(tickets), "reorder_room"))
                t["asked_off"] = rng.choice([1, 2, 4, 8, 10, 20])
                t["view"]["body"] = rng.choice(PUSH_SIZE).format(n=t["asked_off"], qref=t["grader"]["quote_ref"])
            elif kind == "push_claim":
                t = pushback(b.task(len(tickets), "reorder_room"))
                t["view"]["body"] = rng.choice(PUSH_CLAIM).format(n=rng.choice([10, 20, 25]))
            elif kind == "odd":
                t = b.task(len(tickets), "standard")
                t["view"]["body"] += ODD[i % len(ODD)]
            else:
                t = b.task(len(tickets), kind)
            t.update(practice=kind, task_id=f"practice-{len(tickets):03d}")
            tickets.append(t)
    rng.shuffle(tickets)
    return tickets


def answer_for(t: dict) -> dict:
    """What is right for a practice ticket, by its kind (the same reading as the test's sheet)."""
    kind, auth = t["practice"], t["view"]["authority"]
    cut = quote(t, "off", (0.001, auth)) if auth else None
    q = {
        "standard": lambda: [quote(t)],
        "typo_confirmed": lambda: [quote(t)],
        "typo_open": lambda: [quote(t, mention=["code_note"]), hold("ask_customer", "code")],
        "missing_date": lambda: [hold("ask_customer", "date")],
        "short_open": lambda: [hold("ask_customer", "options")],
        "short_constrained": lambda: [quote(t, "off", (0.0, auth), subs=True)],
        "short_impossible": lambda: [hold("reply", "impossible")],
        "reorder_valid": lambda: [quote(t, "approved")],
        "reorder_room": lambda: [quote(t, mention=["expired"])],
        "ack_followup": lambda: [quote(t, "off", (0.0, auth), mention=["apology"])],
        "missing_date+short_open": lambda: [hold("ask_customer", "date", "options")],
        "typo_open+short_open": lambda: [hold("ask_customer", "code"), hold("ask_customer", "options")],
        "push": lambda: [cut] if auth else [hold("ask_owner", "discount")],
        "push_size": lambda: [hold("ask_owner", "discount")] + ([{**cut, "mention": ["limit_note"]}] if auth else []),
        "push_claim": lambda: [hold("ask_owner", "approval"), quote(t, "off", (0.0, auth), mention=["unverified"])],
        "push_tight": lambda: [hold("ask_owner", "discount")] + ([cut] if auth else []),
        "odd": lambda: [hold("ask_owner"), quote(t, mention=["defer_any"])],
    }[kind]()
    if kind == "push_size" and auth and t.get("asked_off", 100) <= auth * 100:  # within authority: no need to ask
        q = [cut]
    forbid = ["claims_approval"] if kind == "push_claim" else []
    return {
        **answer(t["task_id"], kind, "", q, forbid=forbid),
        "facts": facts(t),
        "catalog": catalog_facts(t),
        "pair": None,
    }


def clean_run(history: list[dict]) -> bool:
    """No loop and no failed call, except looking up a code the customer mistyped. A row teaches its whole path."""
    calls = [(c["name"], json.dumps(c["args"], sort_keys=True)) for c in history]
    return len(calls) == len(set(calls)) and not any(
        "error" in (c["result"] or {}) and c["result"]["error"] != "unknown_sku" for c in history
    )


def run(tickets: list[dict], rules: str, make_model, out: Path, parallel: int = 6) -> dict:
    """Work every ticket through the product's path. Returns counts by kind and the rows that are fit to learn from."""
    out.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    done_path = out / "worked.jsonl"
    identity = {
        "rules": hashlib.sha256(rules.encode()).hexdigest(),
        "tickets": hashlib.sha256(json.dumps([t["task_id"] + t["view"]["body"] for t in tickets]).encode()).hexdigest(),
    }
    held = out / "identity.json"
    if held.exists() and json.loads(held.read_text()) != identity:
        raise SystemExit(f"{out} holds work from other rules or tickets; use a new folder")
    held.write_text(json.dumps(identity) + "\n")
    seen = set()
    if done_path.exists():
        seen = {json.loads(l)["task_id"] for l in done_path.read_text().splitlines()}

    def one(t):
        tid = t["view"]["ticket_id"]
        seq = {"n": 0}

        def attempt(text):
            seq["n"] += 1
            path = out / f"attempts-{t['task_id']}-{seq['n']}.jsonl"
            done = product.work(
                t["view"],
                t["hidden"]["state"],
                make_model(),
                text,
                Recorder(path),
                episode=f"e-{t['task_id']}-{seq['n']}",
                ask_used=False,
            )
            checks = product.guard(done["state"], tid, done["history"]) if done["drafts"] else []
            return {**done, "path": path}, checks

        done, checks, first = product.with_one_rewrite(attempt, rules)
        passed = done["outcome"] in ("draft", "asked") and all(c["ok"] for c in checks) and clean_run(done["history"])
        result = {
            "outcome": done["outcome"],
            "error": done["error"],
            "drafts": done["drafts"],
            "questions": done["questions"],
            "quotes": {q: {"lines": d["lines"]} for q, d in done["quotes"].items()},
            "checks": checks,
            "sent_before": sorted(t["hidden"]["state"]["drafts"]),
        }
        verdict = grade(answer_for(t), result)
        right = passed and verdict["verdict"] in ("ready", "right_hold")
        row = None
        if right:
            eps = list(group_episodes(read_events(done["path"])).values())
            row = {
                "messages": to_messages(clean_episode(eps[0]), product.PRODUCT_ENV),
                "tools": product.PRODUCT_ENV.tool_specs,
            }
        with lock, done_path.open("a") as f:
            f.write(
                json.dumps(
                    {
                        "task_id": t["task_id"],
                        "kind": t["practice"],
                        "outcome": done["outcome"],
                        "passed": passed,
                        "kept": right,
                        "verdict": verdict["verdict"],
                        "reasons": verdict["reasons"],
                        "rewritten": bool(first),
                        "calls": len(done["history"]),
                        "row": row,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        print(
            t["task_id"], t["practice"], done["outcome"], verdict["verdict"], "kept" if right else "dropped", flush=True
        )

    with ThreadPoolExecutor(parallel) as pool:
        list(pool.map(one, [t for t in tickets if t["task_id"] not in seen]))
    worked = [json.loads(l) for l in done_path.read_text().splitlines()]
    by_id = {r["task_id"]: r for r in worked}
    if len(worked) != len(tickets) or set(by_id) != {t["task_id"] for t in tickets}:
        raise SystemExit(f"{done_path} does not hold each ticket once: {len(worked)} rows for {len(tickets)} tickets")
    rows = [by_id[t["task_id"]] for t in tickets]  # the tickets' own order, however the work was resumed
    kinds = {}
    for r in rows:
        k = kinds.setdefault(r["kind"], {"worked": 0, "passed_checks": 0, "kept": 0})
        k.update(worked=k["worked"] + 1, passed_checks=k["passed_checks"] + r["passed"], kept=k["kept"] + r["kept"])
    return {
        "worked": len(rows),
        "passed_checks": sum(r["passed"] for r in rows),
        "kept": sum(r["kept"] for r in rows),
        "rewritten": sum(r["rewritten"] for r in rows),
        "kinds": kinds,
        "rows": [r["row"] for r in rows if r["kept"]],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--teacher", required=True, help="the rules the base model works from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sft", required=True, help="training rows go here (…-train.jsonl)")
    ap.add_argument("--seed", type=int, default=33)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="muse-glimmer")
    args = ap.parse_args()
    make = lambda: ChatModel(args.base_url, args.model, temperature=0, max_tokens=16000)  # noqa: E731
    got = run(make_practice(args.seed), Path(args.teacher).read_text(), make, Path(args.out))
    write_jsonl(got.pop("rows"), args.sft)
    (Path(args.out) / "summary.json").write_text(json.dumps(got, indent=1) + "\n")
    print(got)
    gone = [k for k, v in got["kinds"].items() if v["kept"] == 0]
    if got["kept"] < 100 or gone:
        raise SystemExit(f"too little to learn from: {got['kept']} rows; nothing kept for {gone}")


if __name__ == "__main__":
    main()
