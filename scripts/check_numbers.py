#!/usr/bin/env python3
"""Recompute the tables and main numbers from the raw results in baselines/results and print them.

    python3 scripts/check_numbers.py            # the numbers, from the raw rows
    python3 scripts/check_numbers.py --check    # and stop if the README or the pages say something else for those numbers

Standard library only, no GPU. p95 is nearest-rank: the value at position ceil(0.95 * n) of the sorted times.
"""

import argparse
import collections
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "baselines/results"
ARMS = ("none", "checked", "p-checked")  # out of the box, reading the written rules, LoRA with no rules in its prompt
NO_DATE = ("date-1-missing", "date-2-missing", "date-and-short")
BEYOND = ("claimed-approval", "discount-authority-no", "discount-size-big")
RIGHT = ("ready", "right_hold")


def rows(path):
    return [json.loads(line) for line in open(R / path, encoding="utf-8") if line.strip()]


def p95(values):
    s = sorted(values)
    return s[math.ceil(0.95 * len(s)) - 1]


def discount(row):
    """Percent off the shop's earlier list quote, read from the quotes the run left on the desk."""
    prices = [line["unit_price"] for q in row["quotes"].values() for line in q["lines"]]
    return round((1 - prices[-1] / prices[0]) * 100, 2) if len(prices) > 1 else None


def main_test():
    g = rows("test24b/graded.jsonl")
    first = rows("test24b/first-drafts.jsonl")
    by = {a: [r for r in g if r["arm"] == a] for a in ARMS}
    out = {"runs per arm": [len(by[a]) for a in ARMS]}
    out["right"] = [sum(r["verdict"] in RIGHT for r in by[a]) for a in ARMS]
    out["dangerous"] = [sum(r["verdict"] == "dangerous" for r in by[a]) for a in ARMS]
    out["held first"] = [sum(r["arm"] == a for r in first) for a in ARMS]
    out["still held"] = [sum(r["verdict"] == "failed" for r in by[a]) for a in ARMS]
    out["first draft dangerous"] = [sum(r["first_verdict"] == "dangerous" for r in by[a]) for a in ARMS]
    out["mean"] = [round(statistics.mean(r["seconds"] for r in by[a])) for a in ARMS]
    out["median"] = [round(statistics.median(r["seconds"] for r in by[a])) for a in ARMS]
    out["p95"] = [round(p95(r["seconds"] for r in by[a])) for a in ARMS]
    out["generated"] = [round(statistics.mean(r["completion_tokens"] for r in by[a])) for a in ARMS]
    out["all tokens"] = [
        round(statistics.mean(r["completion_tokens"] + r["prompt_tokens"] for r in by[a])) for a in ARMS
    ]
    for key, name in (
        ("discount-authority-yes", "first ticket"),
        ("discount-under-floor", "second ticket"),
        ("discount-size-small", "size ticket"),
    ):
        out[name] = [[discount(r) for r in sorted(by[a], key=lambda x: x["round"]) if r["key"] == key] for a in ARMS]
    out["asked"] = [sum(r["key"] in NO_DATE and r["verdict"] == "right_hold" for r in by[a]) for a in ARMS]
    out["date runs"] = sum(r["key"] in NO_DATE for r in by["p-checked"])
    earlier = rows("test24/graded.jsonl")
    out["first sheet, written rules, right by round"] = [
        sum(r["arm"] == "checked" and r["round"] == n and r["verdict"] in RIGHT for r in earlier) for n in (1, 2)
    ]
    out["asked owner"] = [sum(r["key"] in BEYOND and r["outcome"] == "asked" for r in by[a]) for a in ARMS]
    pair = {(r["round"], r["task_id"]): r for r in by["checked"]}
    faster = [r["seconds"] < pair[(r["round"], r["task_id"])]["seconds"] for r in by["p-checked"]]
    both = [
        (r["seconds"], pair[(r["round"], r["task_id"])]["seconds"])
        for r in by["p-checked"]
        if r["verdict"] in RIGHT and pair[(r["round"], r["task_id"])]["verdict"] in RIGHT
    ]
    nobody = [
        r
        for r in by["p-checked"]
        if r["key"] not in BEYOND + ("discount-authority-yes", "discount-under-floor", "discount-size-small")
    ]
    quoted = [r for r in nobody if [q for q in r["quotes"] if q not in r["sent_before"]]]
    out["nobody asked, runs, no quote, kept approved price"] = [
        len(nobody),
        len(nobody) - len(quoted),
        sum(r["key"] == "reorder-valid" for r in quoted),
    ]
    named = [
        r for r in first if r["arm"] == "p-checked" and any("floor" in d["message"].lower() for d in r["first_drafts"])
    ]
    out["LoRA first drafts naming the floor"] = len(named)
    out["faster"] = [sum(faster), len(faster)]
    out["both right"] = [
        len(both),
        round(statistics.mean(b[0] for b in both)),
        round(statistics.mean(b[1] for b in both)),
    ]
    out["verdict changed by the rewrite"] = sum(r["verdict"] != r["first_verdict"] for r in g)
    transitions = collections.Counter(
        (r["arm"], r["first_verdict"], r["verdict"]) for r in g if r["first_verdict"] != r["verdict"]
    )
    out["transitions"] = {f"{a}: {x} to {y}": n for (a, x, y), n in sorted(transitions.items())}
    blind = json.load(open(R / "test24b/blind-read.json"))
    key = json.load(open(R / "test24b/blind-key.json"))
    read = blind if isinstance(blind, list) else blind.get("items", blind.get("verdicts", blind))
    if isinstance(read, dict):
        read = [{"id": k, **v} if isinstance(v, dict) else {"id": k, "verdict": v} for k, v in read.items()]
    same = sum((r.get("verdict") or r.get("read")) == key[r["id"]]["auto"] for r in read)
    out["blind"] = [same, len(read)]
    return out


def throughput():
    wall = dict(line.split() for line in open(R / "throughput/tp.log") if "WALL" in line)
    lora, rules = int(wall["LORA_WALL"]), int(wall["RULES_WALL"])
    right = [sum(r["verdict"] in RIGHT for r in rows(f"throughput/{d}/graded.jsonl")) for d in ("tp-lora", "tp-rules")]
    per_hour = [24 * 3600 / lora, 24 * 3600 / rules]
    right_hour = [right[0] * 3600 / lora, right[1] * 3600 / rules]
    return {
        "wall seconds": [lora, rules],
        "right of 24": right,
        "per hour": [round(x) for x in per_hour],
        "right per hour": [round(x) for x in right_hour],
        "times": round(per_hour[0] / per_hour[1], 1),
        "times, right ones": round(right_hour[0] / right_hour[1], 1),
        "cost": [round(3.69 / x, 3) for x in per_hour],
    }


def practice():
    s = json.load(open(R / "test24b/practice-summary.json"))
    train = [r for r in rows("practice/train-loss.jsonl") if "train_runtime" in r][0]
    return {
        "worked": s["worked"],
        "passed": s["passed_checks"],
        "kept": s["kept"],
        "train seconds": train["train_runtime"],
        "passed after a rewrite": sum(1 for w in rows("practice/worked.jsonl") if w["passed"] and w["rewritten"]),
        "cost at $3.69 an hour, 92 + 40 minutes": round((92 + 40) / 60 * 3.69, 2),
    }


def people():
    cards = json.load(open(ROOT / "baselines/v2/c60-cards.person.json"))
    answers = json.load(open(ROOT / "baselines/v2/c60-cards.user-answers.json"))["answers"]
    return {
        "rules found": len(cards["cards"]),
        "rules marked ok": sum(c["choice"] == "ok" for c in cards["cards"].values()),
        "answers": len(answers),
        "records read": len({r["ep"] for r in rows("rules/records.jsonl")}),
    }


def learning_round():
    s = json.load(open(R / "learning-round/status.json"))
    check = s["steps"][-1]["check"]
    train = [x for x in s["steps"] if "train_seconds" in x][0]["train_seconds"]
    rows_in = [x for x in s["steps"] if "rows" in x][0]["rows"]
    return {
        "records": s["learned_from"],
        "rows": rows_in,
        "train seconds": train,
        "right of 4": check["good"],
        "dangerous": check["dangerous"],
        "turned on": s["active"] is not None,
    }


def rule_change():
    jobs = rows("product-run/jobs.jsonl")
    a = [j for j in jobs if j["case"] == 1 and j["status"] == "done"]
    c = [j for j in jobs if j["case"] == 4 and j["status"] == "done"]
    return {
        "before": [a[0]["adapter"], a[0]["seconds"], a[0]["tokens"]],
        "after": [a[1]["adapter"], a[1]["seconds"], a[1]["tokens"]],
        "second pair, seconds": [c[0]["seconds"], c[1]["seconds"]],
        "approved": sum(j["decision"] == "approved" for j in jobs),
        "sent": len(rows("product-run/sent.jsonl")),
    }


def demo():
    runs = rows("demo-final/results.jsonl")
    done = [e for r in runs for e in r["events"] if e.get("type") == "done"]
    secs = sorted(e["seconds"] for e in done)

    def followed(r):
        e = next(x for x in r["events"] if x.get("type") == "done")
        if r["kind"] == "price":
            off = e["off"]  # no new quote keeps the list price
            return (
                bool(e["message"])
                and {"hold": not off, "small": bool(off) and 0 < off < 2, "full": bool(off) and 4 < off <= 5.1}[
                    r["price"]
                ]
            )
        return bool(e["message"]) and {"ask": not e["lines"], "earliest": bool(e["lines"])}[r["date"]]

    manual = [r for r in runs if r["price"] is not None]
    lora = rows("demo-lora/results.jsonl")
    return {
        "tickets": len(done),
        "median": round(statistics.median(secs), 1),
        "p95": round(p95(secs), 1),
        "max": round(secs[-1], 1),
        "followed": [sum(followed(r) for r in manual), len(manual)],
        "LoRA, price ticket, percent off": sorted({r["off"] for r in lora if r["kind"] == "price"}),
        "LoRA, date ticket, asked": sum(not r["lines"] and "?" in r["message"] for r in lora if r["kind"] == "date"),
        "LoRA runs each": sum(r["kind"] == "date" for r in lora),
    }


def hashes():
    """Every file listed in a run's sheet.sha256 still has that hash."""
    good = bad = 0
    for run in ("test24", "test24b"):
        for line in open(R / run / "sheet.sha256"):
            digest, name = line.split()
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest:
                good += 1
            else:
                bad += 1
    return {"sheet files matching their hash": good, "not matching": bad}


def public(n):
    """What the README and the two pages must say, built from the numbers above. Returns what is missing."""
    main_, speed, practice_, people_, learning, change, demo_ = (
        n[k] for k in ("main", "throughput", "practice", "people", "learning", "product", "demo")
    )
    by_hand = n["person"]
    minutes, seconds = divmod(round(by_hand["seconds, person"]), 60)
    timed = [
        f"{minutes} minutes {seconds} seconds, and my reply had no quote",
        f"The LoRA took {round(by_hand['seconds, LoRA'])} seconds and got it right, "
        f"and across all 48 test runs it averaged {round(main_['mean'][2])}",
    ]
    c = lambda x: f"{x:,}"
    row = lambda name, vals: f"| {name} | {vals[0]} | {vals[1]} | **{vals[2]}** |"
    pct = lambda pair: ", ".join(f"{v:.2f}%" for v in pair)
    tr = lambda v: f"<td>{v}</td>"
    readme = [
        row("Right", main_["right"]),
        row("Dangerous drafts", main_["dangerous"]),
        row("Dangerous first drafts, before the desk's checks", main_["first draft dangerous"]),
        f"{main_['nobody asked, runs, no quote, kept approved price'][1]} of the {main_['nobody asked, runs, no quote, kept approved price'][0]} runs ended without a quote",
        f"and {main_['nobody asked, runs, no quote, kept approved price'][2]} kept a still-valid approved price",
        f"followed the manual in {demo_['followed'][0]} of {demo_['followed'][1]} runs",
        f"{demo_['LoRA, price ticket, percent off'][0]}% off every time",
        f"in {main_['asked'][2]} of {main_['date runs']}",
        "with the answers added, {} and {}".format(*main_["first sheet, written rules, right by round"]),
        f"{change['before'][1]} seconds and {c(change['before'][2])} tokens with the adapter",
        f"then {change['after'][1]} seconds and {c(change['after'][2])} tokens",
        f"The other took {change['second pair, seconds'][0]} seconds, then {change['second pair, seconds'][1]}",
        f"it built {learning['rows']} rows",
        f"trained for {round(learning['train seconds'] / 60)} minutes",
        f"{ {3: 'Three'}.get(main_['LoRA first drafts naming the floor'], main_['LoRA first drafts naming the floor']) } first drafts from the LoRA",
        row("First draft held back by the desk's checks, then rewritten", main_["held first"]),
        row("Still held back after the rewrite, so a person can't approve it", main_["still held"]),
        row("Seconds per ticket, average", main_["mean"]),
        row("Seconds per ticket, median", main_["median"]),
        row("Seconds per ticket, p95", main_["p95"]),
        row("Tokens generated per ticket", [c(x) for x in main_["generated"]]),
        row("Tokens per ticket, in and out", [c(x) for x in main_["all tokens"]]),
        row("Tickets an hour on one GPU, measured alone", ["not measured", speed["per hour"][1], speed["per hour"][0]]),
        row(
            "Right tickets an hour, same run", ["not measured", speed["right per hour"][1], speed["right per hour"][0]]
        ),
        row(
            "GPU cost per ticket at full load, rented at $3.69 an hour",
            ["not measured", f"${speed['cost'][1]}", f"${speed['cost'][0]}"],
        ),
        row("Discount on a pushback, 5% allowed, two runs", [pct(x) for x in main_["first ticket"]]),
        row("Discount on a pushback, 5% would go under the floor, two runs", [pct(x) for x in main_["second ticket"]]),
        row("Discount when asked for 3% off, two runs", [pct(x) for x in main_["size ticket"]]),
        f"it took {main_['both right'][1]} seconds against {main_['both right'][2]}",
        f"all {main_['faster'][1]} paired runs",
        f"the p95 {demo_['p95']} seconds",
        f"{practice_['worked']} tickets practised, {practice_['passed']} passed the checks ({practice_['passed after a rewrite']} after a rewrite), {practice_['kept']} kept",
        f"{people_['answers']} open points answered",
    ]
    readme += timed
    both_pages = [
        f'<td>{main_["first ticket"][0][0]:.2f}%, {main_["first ticket"][0][1]:.2f}%</td><td>{main_["first ticket"][1][0]:.2f}%, {main_["first ticket"][1][1]:.2f}%</td><td class="ours">{main_["first ticket"][2][0]:.2f}%, {main_["first ticket"][2][1]:.2f}%</td>',
        tr(main_["mean"][0]) + tr(main_["mean"][1]) + f'<td class="ours">{main_["mean"][2]}</td>',
        tr(main_["p95"][0]) + tr(main_["p95"][1]) + f'<td class="ours">{main_["p95"][2]}</td>',
        f'<td>not measured</td><td>{speed["per hour"][1]}</td><td class="ours">{speed["per hour"][0]}</td>',
        f'<td>not measured</td><td>{speed["right per hour"][1]}</td><td class="ours">{speed["right per hour"][0]}</td>',
    ]
    main_page = [
        *timed,
        f"{main_['right'][0]}</a>",
        f"agreed on {main_['blind'][0]} of {main_['blind'][1]}",
        f"{speed['per hour'][0]} an hour against {speed['per hour'][1]}",
        f"or {speed['right per hour'][0]} against {speed['right per hour'][1]}",
        f"{main_['p95'][2]} seconds at p95 against {main_['p95'][1]}",
        f"it asked in {main_['asked'][2]} of {main_['date runs']}",
        f"from {change['before'][1]} seconds to {change['after'][1]} and from {change['second pair, seconds'][0]} to {change['second pair, seconds'][1]}",
        f"{practice_['worked']} in our test",
        f"{practice_['kept']} tickets, {round(practice_['train seconds'] / 60)} minutes",
    ]
    evidence = [
        f'{c(main_["generated"][0])}</td><td>{c(main_["generated"][1])}</td><td class="ours">{main_["generated"][2]}',
        f'{c(main_["all tokens"][0])}</td><td>{c(main_["all tokens"][1])}</td><td class="ours">{c(main_["all tokens"][2])}',
        f"agreed on {main_['blind'][0]} of {main_['blind'][1]}",
        f"{sum(main_['held first'])} first drafts",
    ]
    missing = []
    for where, items in (
        ("README.md", readme),
        ("site/page.html", both_pages + main_page),
        ("site/evidence.html", both_pages + evidence),
    ):
        text = (ROOT / where).read_text(encoding="utf-8")
        missing += [(where, i) for i in items if i not in text]
    return missing


def person():
    counted = [r for r in rows("person-time/results.jsonl") if not r["warmup"]]
    r = counted[0]
    return {
        "tickets by hand": len(counted),
        "seconds, person": r["seconds"],
        "verdict, person": r["verdict"],
        "seconds, LoRA": r["lora"]["seconds"],
        "verdict, LoRA": r["lora"]["verdict"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    n = {
        "main": main_test(),
        "throughput": throughput(),
        "practice": practice(),
        "people": people(),
        "learning": learning_round(),
        "product": rule_change(),
        "demo": demo(),
        "person": person(),
        "sheets": hashes(),
    }
    titles = {
        "main": "Main test: 24 tickets, 2 runs, columns none, checked, p-checked",
        "throughput": "One column alone on the GPU: LoRA, then rules",
        "practice": "Practice and training",
        "people": "Rules and answers",
        "learning": "The product's learning round",
        "product": "The real desk: a rule change and an approval",
        "demo": "The browser demo",
        "person": "One person by hand, one ticket",
        "sheets": "Frozen sheets",
    }
    for title, values in n.items():
        print(f"\n{titles[title]}")
        for k, v in values.items():
            print(f"  {k}: {v}")
    if a.check:
        missing = public(n)
        for where, text in missing:
            print(f"MISSING in {where}: {text}")
        if missing:
            sys.exit(1)
        print("\nThe README and the pages say the same for the numbers listed in public().")


if __name__ == "__main__":
    main()
