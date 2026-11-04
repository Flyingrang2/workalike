"""Scripted workers A and B: two valid ways to handle the same ticket.

  A  checks cost and stock first, then answers once with a confirmed quote.
     When stock is short it keeps the item and quotes the restock ship date.
  B  first sends an acknowledgement with no price in it, then does the same
     checks and sends the quote. When stock is short it quotes an in-stock
     substitute on the original date.

Both hold back their habit when the ticket calls for it: A does not re-check
cost against a valid approved quote, B does not acknowledge twice. A worker sees
only the ticket and what its own tool calls return, never the hidden state.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

from workalike.desk import Desk

FINAL = "Done."

ACKS = [
    "Hi {first}, thanks for the request. I'm checking stock and current costs now and will send your quote shortly.",
    "Thanks {first}! Got your message. I'll confirm pricing and availability and get a quote back to you today.",
    "Hello {first}, thanks for reaching out. Give me a little time to check costs and stock, and I'll follow up with a quote.",
    "Hi {first}, received, thank you. Pricing and stock check in progress; your quote will follow soon.",
    "Thanks for the order request, {first}. I'm on it and will send a full quote once I've confirmed costs and stock.",
]
OPENERS = ["Hi {first},", "Hello {first},", "Thanks {first},"]
QUOTE_BODIES = [
    "here is quote {qid}: {lines}. Total ${total:,.2f}, ready to ship by {ship}.",
    "your quote {qid} is attached: {lines}. The total comes to ${total:,.2f} and it can ship by {ship}.",
    "quote {qid} covers {lines}, ${total:,.2f} in total, shipping by {ship}.",
]
CLOSERS = [
    "Let me know if you'd like to go ahead.",
    "Reply to confirm and we'll get started.",
    "Happy to adjust anything.",
]
SHORT_A = [
    "{name} is out of stock until {restock}, so the earliest we can ship is {ship}. Let me know if that date works.",
    "We're waiting on a restock of {name} ({restock}), so I set the ship date to {ship}.",
]
SHORT_B = [
    "{name} is short right now, so I quoted {sub} instead, which can ship by {ship}.",
    "We're low on {name}, so this quote swaps in {sub} to keep your date of {ship}.",
]
APPROVED = "These are your approved prices from {qref}."
STALE = "{qref} has expired, so this quote uses current pricing."


def _ok(result: dict) -> dict:
    if "error" in result:
        raise RuntimeError(f"scripted worker hit an unexpected error: {result}")
    return result


def _lookup(desk: Desk, kind: str, item: dict, info: dict) -> None:
    for _ in range(2):
        if kind == "cost":
            result = desk.call("get_cost", {"sku": item["sku"]})
        else:
            result = desk.call("check_stock", {"sku": item["sku"], "qty": item["qty"]})
        if result.get("error") == "unknown_sku" and result.get("did_you_mean"):
            item["sku"] = result["did_you_mean"]
            continue
        info.setdefault(item["sku"], {})[kind] = _ok(result)
        return
    raise RuntimeError(f"could not resolve {item}")


def _lookups(desk: Desk, items: list[dict], kinds: tuple[str, ...], info: dict, rng: random.Random) -> None:
    if len(kinds) == 1:
        plan = [(kinds[0], i) for i in range(len(items))]
    else:
        style = rng.choice(["per_item", "per_item_reversed", "by_kind", "by_kind_reversed"])
        order = list(kinds) if "reversed" not in style else list(reversed(kinds))
        if style.startswith("per_item"):
            plan = [(k, i) for i in range(len(items)) for k in order]
        else:
            plan = [(k, i) for k in order for i in range(len(items))]
    for kind, i in plan:
        _lookup(desk, kind, items[i], info)


def _ship_date(today: str, need_by: str, rng: random.Random) -> str:
    ship = date.fromisoformat(need_by) - timedelta(days=rng.choice([0, 1, 2]))
    return max(ship, date.fromisoformat(today) + timedelta(days=1)).isoformat()


def work(desk: Desk, task: dict, style: str, rng: random.Random) -> str:
    view, req = task["view"], task["request"]
    ticket, first = view["ticket_id"], req["first_name"]

    if style == "b" and not req["already_acked"]:
        _ok(desk.call("send_reply", {"ticket_id": ticket, "message": rng.choice(ACKS).format(first=first)}))

    approved, names, extra = None, {}, []
    if req["quote_ref"]:
        quote = _ok(desk.call("lookup_quote", {"quote_id": req["quote_ref"]}))
        items = [{"sku": l["sku"], "qty": l["qty"]} for l in quote["lines"]]
        names.update({l["sku"]: l["name"] for l in quote["lines"]})
        if quote["status"] == "approved":
            approved = {l["sku"]: l["unit_price"] for l in quote["lines"]}
            extra.append(APPROVED.format(qref=req["quote_ref"]))
        else:
            extra.append(STALE.format(qref=req["quote_ref"]))
    else:
        items = [dict(it) for it in req["items"]]

    info: dict = {}
    _lookups(desk, items, ("stock",) if approved else ("cost", "stock"), info, rng)
    for sku, got in info.items():
        names.setdefault(sku, (got.get("cost") or got.get("stock"))["name"])

    ship = _ship_date(view["today"], req["need_by"], rng)
    short = [it for it in items if not info[it["sku"]]["stock"]["enough"]]
    if short:
        item = short[0]
        stock = info[item["sku"]]["stock"]
        if style == "a":
            ship = (date.fromisoformat(stock["restock_date"]) + timedelta(days=rng.choice([0, 1, 2]))).isoformat()
            extra.append(rng.choice(SHORT_A).format(name=names[item["sku"]], restock=stock["restock_date"], ship=ship))
        else:
            sub = stock["substitutes"][0]
            old_name = names[item["sku"]]
            item["sku"] = sub["sku"]
            names[sub["sku"]] = sub["name"]
            info[sub["sku"]] = {"cost": _ok(desk.call("get_cost", {"sku": sub["sku"]}))}
            extra.append(rng.choice(SHORT_B).format(name=old_name, sub=sub["name"], ship=ship))

    lines = []
    for it in items:
        if approved:
            price = approved[it["sku"]]
        else:
            cost = info[it["sku"]]["cost"]["unit_cost"]
            price = _ok(desk.call("calc_price", {"sku": it["sku"], "qty": it["qty"], "unit_cost": cost}))["unit_price"]
        lines.append({"sku": it["sku"], "qty": it["qty"], "unit_price": price})
    quote = _ok(desk.call("create_quote", {"ticket_id": ticket, "lines": lines, "ship_by": ship}))

    summary = "; ".join(f"{l['qty']} x {names[l['sku']]} at ${l['unit_price']:.2f}" for l in quote["lines"])
    body = rng.choice(QUOTE_BODIES).format(qid=quote["quote_id"], lines=summary, total=quote["total"], ship=ship)
    message = " ".join([rng.choice(OPENERS).format(first=first), body, *extra, rng.choice(CLOSERS)])
    _ok(desk.call("send_reply", {"ticket_id": ticket, "message": message, "quote_id": quote["quote_id"]}))
    return FINAL
