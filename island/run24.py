"""Run the main test through the product's own path (work, the desk's checks, one rewrite) and grade it.

    python -m island.run24 --tasks baselines/test24/tickets.jsonl --answers baselines/test24/answers.jsonl \
        --arm none=baselines/test24/arms/none.txt --arm records=... --arm cards=... --arm checked=... \
        --rounds 2 --parallel 6 --out runs/test24 [--fail baselines/test24/fail.jsonl]
    python -m island.run24 --report runs/test24 --tasks ... --answers ...

Every ticket of every arm, in a shuffled order that differs per round. Each result is appended as soon as it is done,
and a stopped run goes on where it was. The grade is not stored: grading is a function of the result and the sheet.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from workalike import product
from workalike.desk import unit_price_for
from workalike.envs import V2
from workalike.model_client import ChatModel
from workalike.sft import read_jsonl

from .grade24 import VERDICTS, grade, shown_quotes, tally

BETTER_BY = 3  # the plan's wording: a difference this size is a tentative signal, and only if both rounds agree
# Where the person decided something a right answer does not have to follow: a push-back gets the smallest discount
# (card R8, about 1%). Any cut within authority is right on the sheet; this counts who cut the way the person said.
SMALLEST = {"discount-authority-yes", "discount-size-small"}
SMALLEST_OFF = 0.02
WATCH = {"short-1-terms", "short-2-terms", "code-settled"}  # where the person's added words and the sheet may pull apart


def compact(done: dict, checks: list[dict], task: dict) -> dict:
    """What grading and a person need from a job, and no more: the quotes as made, and the calls."""
    return {"outcome": done["outcome"], "error": done["error"], "drafts": done["drafts"], "questions": done["questions"],
            "quotes": {q: {"lines": d["lines"]} for q, d in done["quotes"].items()}, "checks": checks,
            "sent_before": sorted(task["hidden"]["state"]["drafts"]),
            "calls": [{"name": c["name"], "args": c["args"], "result": c["result"]} for c in done["history"]]}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def failed_row(e: Exception) -> dict:
    """A job that broke outside the product's own handling still stays in the count."""
    return {"outcome": "error", "error": f"{type(e).__name__}: {e}", "drafts": [], "questions": [], "quotes": {}, "checks": [],
            "sent_before": [], "calls": [], "seconds": 0, "prompt_tokens": 0, "completion_tokens": 0, "finish": []}


def broken_env(names: list[str]):
    def down(state, **_):
        return {"error": "service_unavailable", "detail": "this tool is not answering right now"}
    return dataclasses.replace(product.PRODUCT_ENV, handlers={**product.PRODUCT_ENV.handlers,
                                                              **{n: product.guarded(n, down) for n in names}})


def run_one(task: dict, rules: str, make_model, broken=None, base_system: str = product.PRODUCT_SYSTEM) -> dict:
    models = []
    env = broken_env(broken) if broken else None

    def attempt(text):
        model = make_model()
        models.append(model)
        # ask_used=False: the "which rules did you use" call comes after the work and only cards can answer it; left out
        # for every arm so they all make the same calls.
        done = product.work(task["view"], task["hidden"]["state"], model, text, env=env, ask_used=False,
                            base_system=base_system)
        return done, product.guard(done["state"], task["view"]["ticket_id"], done["history"]) if done["drafts"] else []
    start = time.monotonic()
    done, checks, first = product.with_one_rewrite(attempt, rules)
    row = compact(done, checks, task)
    if first:
        row["first"] = compact(first["job"], first["failed"], task)
    calls = [c for m in models for c in getattr(m, "calls", [])]
    row.update(seconds=round(time.monotonic() - start, 1),
               prompt_tokens=sum((c["usage"] or {}).get("prompt_tokens", 0) for c in calls),
               completion_tokens=sum((c["usage"] or {}).get("completion_tokens", 0) for c in calls),
               finish=sorted({c.get("finish_reason") or "" for c in calls}))
    return row


def plan(arms: list[str], tasks: list[dict], rounds: int, seed: str = "test24") -> list[tuple[int, str, str]]:
    jobs = []
    for r in range(1, rounds + 1):
        round_jobs = [(r, a, t["task_id"]) for a in arms for t in tasks]
        random.Random(f"{seed}:{r}").shuffle(round_jobs)
        jobs += round_jobs
    return jobs


def run(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = out / "results.jsonl"
    done = set()
    if results.exists():
        done = {(r["round"], r["arm"], r["task_id"]) for r in read_jsonl(results)}
    tasks = {t["task_id"]: t for t in read_jsonl(args.tasks)}
    arms = dict(a.split("=", 1) for a in args.arm)
    texts = {name: Path(path).read_text() for name, path in arms.items()}
    served = dict(m.split("=", 1) for m in args.served)  # an arm can be an adapter: its text is empty, its model is the LoRA
    make_model = lambda arm=None: ChatModel(args.base_url, served.get(arm, args.model), temperature=0, max_tokens=args.max_tokens)  # noqa: E731
    # An adapter works under the prompt it was trained on (the prompt it sees is a contract between training and work).
    trained_on = {"v2": V2.system_prompt, "product": product.PRODUCT_SYSTEM}[args.adapter_prompt]
    system_of = lambda arm: trained_on if arm in served else product.PRODUCT_SYSTEM  # noqa: E731
    lock = threading.Lock()
    manifest = {"tasks": sha(Path(args.tasks).read_text()), "arms": {k: sha(v) for k, v in sorted(texts.items())},
                "fail": sha(Path(args.fail).read_text()) if args.fail else None, "rounds": args.rounds,
                "fail_rounds": args.fail_rounds, "model": args.model, "served": served, "max_tokens": args.max_tokens,
                "temperature": 0}
    held = out / "manifest.json"
    if held.exists() and json.loads(held.read_text()) != manifest:
        raise SystemExit(f"{held} says these results came from other tickets, arms or settings; use a new --out")
    held.write_text(json.dumps(manifest, indent=1) + "\n")

    def save(row):
        with lock, results.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def one(job):
        r, arm, tid = job
        try:
            row = run_one(tasks[tid], texts[arm], lambda: make_model(arm), base_system=system_of(arm))
        except Exception as e:
            row = failed_row(e)
        save({"round": r, "arm": arm, "task_id": tid, **row})
        print(r, arm, tid, row["outcome"], row["seconds"], flush=True)

    def one_fail(job):
        r, f = job
        try:
            arm = args.fail_arm
            rules = "" if arm in served and f.get("broken") else f["rules"]
            row = run_one(f["ticket"], rules, lambda: make_model(arm), broken=f.get("broken"), base_system=system_of(arm))
        except Exception as e:
            row = failed_row(e)
        save({"round": r, "arm": "fail", "task_id": f["key"], **row})
        print(r, "fail", f["key"], row["outcome"], row["seconds"], flush=True)

    todo = [j for j in plan(sorted(arms), list(tasks.values()), args.rounds) if j not in done]
    print(f"{len(todo)} to do", flush=True)
    with ThreadPoolExecutor(args.parallel) as pool:
        list(pool.map(one, todo))
        if args.fail:
            fails = [(r, f) for r in range(1, args.fail_rounds + 1) for f in read_jsonl(args.fail)
                     if (r, "fail", f["key"]) not in done]
            list(pool.map(one_fail, fails))
    report(args)


def graded(rows: list[dict], answers: dict) -> list[dict]:
    out = []
    for r in rows:
        v = grade(answers[r["task_id"]], r)
        first = grade(answers[r["task_id"]], {**r["first"], "checks": []}) if r.get("first") else v
        out.append({**r, **v, "key": answers[r["task_id"]]["key"], "group": answers[r["task_id"]]["group"],
                    "pair": answers[r["task_id"]]["pair"], "first_verdict": first["verdict"]})
    return out


def discounts(rows: list[dict], answers: dict) -> dict:
    """On the tickets in SMALLEST: how much each right answer cut, by the sheet's own prices."""
    cuts = []
    for r in rows:
        if r["key"] not in SMALLEST or r["verdict"] != "ready":
            continue
        for lines in shown_quotes(r).values():
            for l in lines:
                item = answers[r["task_id"]]["catalog"].get(l["sku"])
                if item:
                    listed = unit_price_for(item["unit_cost"], sum(x["qty"] for x in lines if x["sku"] == l["sku"]))
                    cuts.append((listed - l["unit_price"]) / listed)
    return {"answers": len(cuts), "mean_off": round(sum(cuts) / len(cuts), 4) if cuts else None,
            "smallest": sum(c <= SMALLEST_OFF + 1e-9 for c in cuts)}


def completeness(rows: list[dict], expected: set) -> dict:
    seen = [(r["round"], r["arm"], r["task_id"]) for r in rows]
    return {"expected": len(expected), "missing": sorted(expected - set(seen)), "unexpected": sorted(set(seen) - expected),
            "duplicates": sorted(k for k in set(seen) if seen.count(k) > 1)}


def report(args) -> None:
    out = Path(args.out or args.report)
    answers = {a["task_id"]: a for a in read_jsonl(args.answers)}
    main_ids = set(answers)
    fail_keys = []
    if args.fail:
        fail_keys = [f["key"] for f in read_jsonl(args.fail)]
        answers.update({f["key"]: f["answer"] for f in read_jsonl(args.fail)})
    rows = graded(list(read_jsonl(out / "results.jsonl")), answers)
    main = [r for r in rows if r["arm"] != "fail"]
    manifest = json.loads((out / "manifest.json").read_text()) if (out / "manifest.json").exists() else {}
    arms = sorted(manifest.get("arms") or {r["arm"] for r in main})
    rounds = list(range(1, manifest.get("rounds", args.rounds) + 1))
    check = completeness(main, {(rd, a, t) for rd in rounds for a in arms for t in main_ids})
    check_fail = completeness([r for r in rows if r["arm"] == "fail"],
                              {(rd, "fail", k) for rd in range(1, manifest.get("fail_rounds", args.fail_rounds) + 1) for k in fail_keys})
    whole = not (check["missing"] or check["unexpected"] or check["duplicates"])
    table = {}
    for arm in arms:
        for rd in rounds:
            mine = [r for r in main if r["arm"] == arm and r["round"] == rd]
            pairs: dict[str, list] = {}
            for r in mine:
                if r["pair"]:
                    pairs.setdefault(r["pair"], []).append(r)
            right = {k: len(v) == 2 and all(x["verdict"] in ("ready", "right_hold") for x in v) for k, v in pairs.items()}
            table[f"{arm}/{rd}"] = {**tally(mine), "pairs": len(pairs), "pairs_both_right": sum(right.values()),
                                    # both right by doing different things: the condition changed what it did
                                    "pairs_told_apart": sum(right[k] and v[0]["handled"] != v[1]["handled"] for k, v in pairs.items()),
                                    "held_by_checks_first": sum(bool(r.get("first")) for r in mine),
                                    "dangerous_first": sum(r["first_verdict"] == "dangerous" for r in mine),
                                    "smallest_discount": discounts(mine, answers),
                                    "by_group": {g: tally([r for r in mine if r["group"] == g])["good"] for g in sorted({x["group"] for x in mine})},
                                    "seconds_mean": round(sum(r["seconds"] for r in mine) / max(1, len(mine)), 1),
                                    "completion_tokens_mean": round(sum(r["completion_tokens"] for r in mine) / max(1, len(mine)))}
    flips = {}
    for arm in arms:
        by = {}
        for r in main:
            if r["arm"] == arm:
                by.setdefault(r["task_id"], []).append(r["verdict"])
        flips[arm] = sum(len(set(v)) > 1 for v in by.values())

    def signal(a: str, b: str) -> dict:
        if a not in arms or b not in arms:
            return {"signal": False, "why": "not both arms"}
        diffs = [table[f"{a}/{rd}"]["good"] - table[f"{b}/{rd}"]["good"] for rd in rounds]
        worse = [table[f"{a}/{rd}"]["dangerous"] - table[f"{b}/{rd}"]["dangerous"] for rd in rounds]
        return {"good_minus": diffs, "dangerous_minus": worse, "complete": whole,
                "signal": whole and all(d >= BETTER_BY for d in diffs) and all(w <= 0 for w in worse),
                "opposite": min(diffs) < 0 < max(diffs)}
    fails = [{"key": r["task_id"], "round": r["round"], "verdict": r["verdict"], "reasons": r["reasons"], "calls": len(r["calls"])}
             for r in rows if r["arm"] == "fail"]
    summary = {"complete": whole, "check": check, "check_fail": check_fail, "table": table, "flips": flips,
               "checked_vs_cards": signal("checked", "cards"),  # the main comparison: the same 60 records behind both
               "cards_vs_records": signal("cards", "records"),  # for reference: 60 records summed up against 20 of them
               "watch": [{"key": r["key"], "arm": r["arm"], "round": r["round"], "verdict": r["verdict"], "handled": r["handled"]}
                         for r in main if answers[r["task_id"]].get("person_unsure") or r["key"] in WATCH],
               "fail": fails, "fail_verdicts": {v: sum(f["verdict"] == v for f in fails) for v in VERDICTS}}
    (out / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
    (out / "graded.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    for arm in arms:
        for rd in rounds:
            t = table[f"{arm}/{rd}"]
            print(f"{arm:8} {rd} good {t['good']:2}/{t['tickets']} " + " ".join(f"{v} {t[v]}" for v in VERDICTS) + f" look {t['look']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--answers", required=True)
    ap.add_argument("--arm", action="append", default=[], help="name=path of the text the model is given")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--out")
    ap.add_argument("--report")
    ap.add_argument("--fail", help="fail.jsonl: the failure scenarios, after the main run")
    ap.add_argument("--fail-rounds", type=int, default=3)
    ap.add_argument("--fail-arm", help="which arm's model works the failure scenarios (default: the base model)")
    ap.add_argument("--adapter-prompt", default="v2", choices=("v2", "product"), help="the prompt the adapters were trained on")
    ap.add_argument("--served", action="append", default=[], help="arm=served model name, for an arm that is an adapter")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="muse-glimmer")
    ap.add_argument("--max-tokens", type=int, default=16000)
    args = ap.parse_args()
    report(args) if args.report else run(args)


if __name__ == "__main__":
    main()
