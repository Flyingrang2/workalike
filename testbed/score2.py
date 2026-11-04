"""Scoring for quote-desk/2: is the business outcome valid, and which way was the ticket handled.

Business (business_ok), from the clean episode and the task's hidden state:
  end   accepted    the customer accepted a quote
        closed      no quote can meet what the customer needs, and the worker wrote to them
        waiting     the worker stopped after writing to the customer, with nothing accepted
        unfinished  nothing reached the customer
  Every quote shown to the customer (attached, or named by id in a reply) must have valid prices:
  the list price for the item's total in that quote, a still-valid approved price, or a discount that
  stays at or above the floor and is within the worker's authority or agreed by the owner when asked
  before the quote was made. A reply without a quote carries no price, and amounts in a quoted reply
  are prices the desk gave. A condition the customer had not revealed is not owed: a quote the customer
  turned down for it is not an error, as long as the ticket ends accepted (or closed when nothing fits).
  The order quotes were made in never matters.

Way, compared with a reference episode on the same ticket (worker C, or the person):
  asked_first     a reply without a quote went out before the first quote was shown
  asked_owner     the owner was asked
  discounted      a quote shown carried a price under list that is not an approved price
  offered_choice  the first reply with a quote showed two or more quotes      (short_constrained only)
  shape           how the accepted quote covers the short item: later, split, substitute, mix
                                                                              (short_constrained only)
asked_about (auxiliary, not a choice point) says, for each open condition of the ticket, whether the first reply
without a quote mentions it: the item code for typo_open, a date for missing_date, the restock or another option
for short_open. The customer answers any reply without a quote with the same paragraph, so asked_first alone
also counts a bare acknowledgement as a question.
A lookup of an unknown code as written counts as a lookup miss, any other failed call as a tool error;
neither changes business_ok by itself.
"""

from __future__ import annotations

import re
from collections import Counter

from workalike.desk import cents, customer_prices, unit_price_for
from workalike.desk2 import QUOTE_ID, approved_price, floor_price, within_authority
from workalike.filter import clean_episode, episode_meta
from workalike.record import group_episodes, read_events

from .score import AMOUNT, PRICE, flat_calls, trace

BASE_POINTS = ("asked_first", "asked_owner", "discounted")
SHORT_POINTS = ("offered_choice", "shape")


def points(task: dict) -> tuple[str, ...]:
    return BASE_POINTS + (SHORT_POINTS if "short_constrained" in task["grader"]["flags"] else ())


def ok(c: dict) -> bool:
    return "error" not in (c["result"] or {})


def shown_ids(c: dict, made: dict) -> list[str]:
    """Quotes a reply showed: the attached one, then any named by id in the message."""
    ids = [c["args"]["quote_id"].upper()] if c["args"].get("quote_id") else []
    ids += [q.upper() for q in QUOTE_ID.findall(c["args"]["message"])]
    return [q for q in dict.fromkeys(ids) if q in made]


def price_problem(task: dict, line: dict, lines: list[dict], owner_ok: bool) -> str | None:
    state, g = task["hidden"]["state"], task["grader"]
    sku, p = line["sku"], line["unit_price"]
    cost = state["catalog"][sku]["unit_cost"]
    listed = unit_price_for(cost, sum(l["qty"] for l in lines if l["sku"] == sku))
    if approved_price(state, sku, p) or abs(p - listed) <= 0.0051:
        return None
    if p > listed:
        return f"over list for {sku}"
    if p < floor_price(cost) - 1e-9:
        return f"under the floor for {sku}"
    if within_authority(listed, p, g["authority"]) or owner_ok:
        return None
    return f"discount without authority for {sku}"


def shape(task: dict, lines: list[dict]) -> str | None:
    short = task["grader"]["short"]
    if not short:
        return None
    own = [l for l in lines if l["sku"] == short["sku"]]
    subs = [l for l in lines if l["sku"] in short["substitutes"]]
    if own and subs:
        return "mix"
    if subs:
        return "substitute"
    late = [l for l in own if l["ship_by"] > task["grader"]["need_by"]]
    return "split" if late and len(late) < len(own) else "later" if late else "all"


ABOUT = {
    "typo_open": re.compile(r"catalog|did you mean|confirm|meant|[A-Z]{3}-[A-Z0-9]{2,4}|which", re.I),
    "missing_date": re.compile(r"when|date|ship by|deadline|need them", re.I),
    "short_open": re.compile(r"restock|wait|rest |remaining|substitute|instead|split|now and|option", re.I),
}


def asked_about(task: dict, replies: list) -> dict:
    first_q = next((c["args"]["message"] for _, c, ids in replies if not ids), None)
    return {f: bool(first_q and rx.search(first_q)) for f, rx in ABOUT.items() if f in task["grader"]["flags"]}


def score(task: dict, episode: dict) -> dict:
    g = task["grader"]
    calls = flat_calls(episode)
    made = {
        c["result"]["quote_id"]: (i, c["result"]["lines"])
        for i, c in enumerate(calls)
        if c["name"] == "create_quote" and ok(c)
    }
    replies = [(i, c, shown_ids(c, made)) for i, c in enumerate(calls) if c["name"] == "send_reply" and ok(c)]
    owner_at = [i for i, c in enumerate(calls) if c["name"] == "ask_owner" and ok(c)]
    owner_yes = g["owner"] == "floor_ok"
    reasons = []

    shown = list(dict.fromkeys(q for _, _, ids in replies for q in ids))
    for qid in shown:
        at, lines = made[qid]
        owner_ok = owner_yes and any(i < at for i in owner_at)
        for line in lines:
            why = price_problem(task, line, lines, owner_ok)
            if why:
                reasons.append(f"{qid}: {why}" if len(shown) > 1 else why)
    prices = customer_prices(c["result"] for c in calls if c["result"])
    for _, c, ids in replies:
        if not ids and PRICE.search(c["args"]["message"]):
            reasons.append("price in a reply without a quote")
        if ids and any(cents(m) not in prices for m in AMOUNT.findall(c["args"]["message"])):
            reasons.append("reply amount the desk never gave")
    reasons = list(dict.fromkeys(reasons))

    accepted = next(
        (c["result"]["customer_accepted"] for _, c, _ in replies if c["result"].get("customer_accepted")), None
    )
    end = (
        "accepted"
        if accepted
        else "closed"
        if replies and not g["feasible"]
        else "waiting"
        if replies
        else "unfinished"
    )
    first_quote = next((i for i, _, ids in replies if ids), None)
    final = made[accepted][1] if accepted else made[shown[-1]][1] if shown else []
    way = {
        "asked_first": any(not ids and (first_quote is None or i < first_quote) for i, _, ids in replies),
        "asked_owner": bool(owner_at),
        "discounted": any(
            l["unit_price"]
            < unit_price_for(
                task["hidden"]["state"]["catalog"][l["sku"]]["unit_cost"],
                sum(x["qty"] for x in made[q][1] if x["sku"] == l["sku"]),
            )
            - 0.0051
            and not approved_price(task["hidden"]["state"], l["sku"], l["unit_price"])
            for q in shown
            for l in made[q][1]
        ),
        "offered_choice": next((len(ids) > 1 for _, _, ids in replies if ids), False),
        "shape": shape(task, final) if final else None,
    }
    errors = [c for c in calls if not ok(c)]
    misses = [c for c in errors if c["name"] == "check_stock" and c["result"]["error"] == "unknown_sku"]
    return {
        "business_ok": end in ("accepted", "closed") and not reasons,
        "end": end,
        "reasons": reasons,
        "way": way,
        "points": list(points(task)),
        "asked_about": asked_about(task, replies),
        "quotes_shown": len(shown),
        "turned_down": sum(c["result"]["customer_reply"].startswith("That won't work") for _, c, _ in replies),
        "lookup_misses": len(misses),
        "tool_errors": len(errors) - len(misses),
    }


def same_way(row: dict, ref: dict) -> bool:
    """Handled the way the reference handled the same ticket, at every choice point that applies to it."""
    return bool(row["points"]) and all(row["way"].get(k) == ref["way"].get(k) for k in row["points"])


MISSING = {
    "business_ok": False,
    "end": "unfinished",
    "reasons": ["no record"],
    "way": {},
    "points": [],
    "asked_about": {},
    "quotes_shown": 0,
    "turned_down": 0,
    "lookup_misses": 0,
    "tool_errors": 0,
}


def score_records(records_path, tasks_by_id: dict, extra: dict | None = None) -> tuple[list[dict], dict]:
    """Score a record file against every task; a task with no record fails, a second record is ignored."""
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
        rows.append(
            {
                "task_id": meta["task_id"],
                "family": task["family"],
                "pair": task["grader"]["pair"],
                "status": status,
                "trace": trace(ep),
                **score(task, ep),
                **(extra or {}).get(meta["ep"], {}),
            }
        )
    for task_id, task in tasks_by_id.items():
        if task_id not in seen:
            rows.append(
                {
                    "task_id": task_id,
                    "family": task["family"],
                    "pair": task["grader"]["pair"],
                    "status": "missing",
                    "trace": [],
                    **MISSING,
                }
            )
    summary = {
        "tasks": len(tasks_by_id),
        "duplicates_or_unknown": duplicates,
        "business_ok": sum(r["business_ok"] for r in rows),
        "end": dict(Counter(r["end"] for r in rows)),
        "tool_errors": sum(r["tool_errors"] for r in rows),
        "status": dict(Counter(r["status"] for r in rows)),
    }
    return rows, summary


def compare(rows: list[dict], ref_rows: list[dict]) -> dict:
    """Way match against a reference: per ticket, with a valid outcome too, and twins where both sides do."""
    ref = {r["task_id"]: r for r in ref_rows}
    matched = {r["task_id"]: r["task_id"] in ref and same_way(r, ref[r["task_id"]]) for r in rows}
    both = {r["task_id"]: matched[r["task_id"]] and r["business_ok"] for r in rows}
    pairs: dict[str, list[bool]] = {}
    for r in rows:
        if r["pair"]:
            pairs.setdefault(r["pair"], []).append(both[r["task_id"]])
    by_point = Counter()
    for r in rows:
        for k in r["points"]:
            by_point[k] += r["task_id"] in ref and r["way"].get(k) == ref[r["task_id"]]["way"].get(k)
    return {
        "tasks": len(rows),
        "business_ok": sum(r["business_ok"] for r in rows),
        "same_way": sum(matched.values()),
        "both": sum(both.values()),
        "points_matched": dict(by_point),
        "pairs": len(pairs),
        "pairs_both_sides": sum(len(v) == 2 and all(v) for v in pairs.values()),
    }
