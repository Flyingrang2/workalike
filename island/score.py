"""Scoring for the test bed: was the ticket handled correctly, and how was it handled.

Correctness uses the task's grader and the clean episode only. A "reply" here is
a customer-facing draft held for approval; nothing is sent. A ticket is correct
when the business outcome is right (business_correct) and, on tickets that ask a
habit to be held back, it was held back (restraint_ok). Habit measures:
  cost_before_first_reply  a cost lookup was attempted before the first reply (raw order)
  a_way                    correct, and a cost lookup succeeded before the first successful reply
  b_way                    correct, and a price-free acknowledgement went out before any cost lookup
  short_choice             on a stock shortage, kept the item with a later date or swapped in a substitute
  unneeded_cost_lookup     looked up cost although a valid approved quote made it unnecessary
"""
from __future__ import annotations

import re
from collections import Counter

from workalike.desk import cents, customer_prices
from workalike.filter import clean_episode, episode_meta
from workalike.record import group_episodes, read_events

PRICE = re.compile(r"\$\s?\d|\b\d[\d,]*(?:\.\d+)?\s?(?:USD|dollars?)\b", re.IGNORECASE)
AMOUNT = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)")


def flat_calls(episode: dict) -> list[dict]:
    return [c for turn in episode["turns"] for c in turn["calls"]]


def trace(episode: dict) -> list[str]:
    out = []
    for c in flat_calls(episode):
        label = c["name"]
        if c["name"] == "send_reply":
            label = "reply+quote" if c["args"].get("quote_id") else "reply"
        if "error" in (c["result"] or {}):
            label += "!"
        out.append(label)
    return out


def cost_before_first_reply(episode: dict) -> bool | None:
    calls = flat_calls(episode)
    cost = next((i for i, c in enumerate(calls) if c["name"] == "get_cost"), None)
    reply = next((i for i, c in enumerate(calls) if c["name"] == "send_reply"), None)
    if reply is None:
        return None if cost is None else True
    return cost is not None and cost < reply


def score(task: dict, episode: dict) -> dict:
    g = task["grader"]
    calls = flat_calls(episode)
    replies = [c for c in calls if c["name"] == "send_reply" and "error" not in c["result"]]
    quotes = [(i, c) for i, c in enumerate(calls) if c["name"] == "create_quote" and "error" not in c["result"]]
    reasons, short_choice = [], None

    if not quotes:
        reasons.append("no quote")
    else:
        qi, qc = quotes[-1]
        quote = qc["result"]
        attached = [c for c in calls[qi + 1:] if c["name"] == "send_reply" and "error" not in c["result"]
                    and c["args"].get("quote_id") == quote["quote_id"]]
        if not attached:
            reasons.append("quote never sent")
        got = {l["sku"]: l for l in quote["lines"]}
        if len(got) != len(quote["lines"]):
            reasons.append("duplicate lines")
        expected = {l["sku"]: l for l in g["lines"]}
        short = g["short"]
        if short and short["sku"] not in got:
            subs = [s for s in got if s in short["substitutes"]]
            if len(subs) == 1:
                short_choice = "substitute"
                sub = subs[0]
                expected = {**{k: v for k, v in expected.items() if k != short["sku"]},
                            sub: {"sku": sub, "qty": expected[short["sku"]]["qty"],
                                  "unit_price": short["substitutes"][sub]}}
        elif short:
            short_choice = "later_date"
        if set(got) != set(expected):
            reasons.append(f"wrong items {sorted(got)} vs {sorted(expected)}")
        else:
            for sku, line in expected.items():
                if got[sku]["qty"] != line["qty"]:
                    reasons.append(f"wrong qty for {sku}")
                if abs(got[sku]["unit_price"] - line["unit_price"]) > 0.0051:
                    reasons.append(f"wrong price for {sku}")
        if short_choice == "later_date" and quote["ship_by"] < short["restock_date"]:
            reasons.append("ships before restock")
        if short_choice != "later_date" and quote["ship_by"] > g["need_by"]:
            reasons.append("ships after the customer's date")
        others = {c["args"]["quote_id"] for c in replies if c["args"].get("quote_id")} - {quote["quote_id"]}
        if others:
            reasons.append("another quote sent to the customer")
        # Amounts in the quoted reply must be prices the desk gave in this episode (the quote, an earlier
        # quote, an alternative that was priced), never a cost or a number from nowhere.
        prices = customer_prices(c["result"] for c in calls if c["result"])
        for c in attached:
            if any(cents(m) not in prices for m in AMOUNT.findall(c["args"]["message"])):
                reasons.append("reply amount the desk never gave")
                break

    for c in replies:
        if not c["args"].get("quote_id") and PRICE.search(c["args"]["message"]):
            reasons.append("price promised before a quote")
    acks = [c for c in replies if not c["args"].get("quote_id")]
    business_correct = not reasons
    called_get_cost = any(c["name"] == "get_cost" for c in calls)
    restraint = []
    if g["ack_forbidden"] and acks:
        restraint.append("acknowledged twice")
    if called_get_cost and not g["needs_cost"]:
        restraint.append("unneeded cost lookup")
    reasons += restraint

    def first(pred):
        return next((i for i, c in enumerate(calls) if pred(c)), None)

    ok = lambda c: "error" not in (c["result"] or {})
    first_reply = first(lambda c: c["name"] == "send_reply" and ok(c))
    first_cost_try = first(lambda c: c["name"] == "get_cost")
    first_cost_ok = first(lambda c: c["name"] == "get_cost" and ok(c))
    first_ack = first(lambda c: c["name"] == "send_reply" and ok(c) and not c["args"].get("quote_id")
                      and not PRICE.search(c["args"].get("message", "")))
    correct = not reasons
    return {
        "correct": correct,
        "business_correct": business_correct,
        "restraint_ok": not restraint,
        "reasons": reasons,
        "cost_before_first_reply": cost_before_first_reply(episode),
        "a_way": correct and first_cost_ok is not None and first_reply is not None and first_cost_ok < first_reply,
        "b_way": correct and first_ack is not None and (first_cost_try is None or first_ack < first_cost_try),
        "called_get_cost": called_get_cost,
        "unneeded_cost_lookup": called_get_cost and not g["needs_cost"],
        "acked": bool(acks),
        "short_choice": short_choice,
    }


MISSING = {"correct": False, "business_correct": False, "restraint_ok": False, "reasons": ["no record"], "cost_before_first_reply": None, "a_way": False,
           "b_way": False, "called_get_cost": False, "unneeded_cost_lookup": False, "acked": False,
           "short_choice": None}
COUNTED = ("correct", "business_correct", "a_way", "b_way", "acked", "called_get_cost", "unneeded_cost_lookup")


def score_records(records_path, tasks_by_id: dict, extra: dict | None = None) -> tuple[list[dict], dict]:
    """Score a record file against every task in tasks_by_id, whichever entrance wrote it.

    The denominator is the task list: a task with no record counts as a failure, and a
    task recorded twice is scored on its first episode and reported as a duplicate.
    extra maps episode id to any per-episode numbers (tokens, seconds) to keep in the rows.
    """
    rows, seen, duplicates = [], set(), 0
    for events in group_episodes(read_events(records_path)).values():
        meta = episode_meta(events)
        if meta["task_id"] in seen or meta["task_id"] not in tasks_by_id:
            duplicates += 1
            continue
        seen.add(meta["task_id"])
        status = next((e.get("status") for e in events if e["type"] == "episode_end"), "unfinished")
        ep = clean_episode(events)
        task = tasks_by_id[meta["task_id"]]
        rows.append({"task_id": meta["task_id"], "family": task["family"], "status": status,
                     "trace": trace(ep), **score(task, ep), **(extra or {}).get(meta["ep"], {})})
    for task_id, task in tasks_by_id.items():
        if task_id not in seen:
            rows.append({"task_id": task_id, "family": task["family"], "status": "missing", "trace": [], **MISSING})
    summary = {"tasks": len(tasks_by_id), "duplicates_or_unknown": duplicates,
               **{k: sum(bool(r[k]) for r in rows) for k in COUNTED},
               "cost_before_first_reply": sum(bool(r["cost_before_first_reply"]) for r in rows),
               "short_choice": dict(Counter(r["short_choice"] for r in rows if r["short_choice"])),
               "status": dict(Counter(r["status"] for r in rows)),
               "by_family": {f: {k: sum(bool(r[k]) for r in rows if r["family"] == f) for k in ("correct", "a_way", "b_way")}
                             | {"n": sum(r["family"] == f for r in rows)}
                             for f in sorted({r["family"] for r in rows})}}
    return rows, summary
