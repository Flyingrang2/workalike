"""Quote desk, second version: split shipments, a floor price, the owner, and the customer's answer.

What changes from quote-desk/1:
  check_stock   also shows how many units the restock brings, and every substitute with stock
  calc_price    also shows the floor price (the lowest unit price the margin rule allows)
  create_quote  each line has its own ship date; stock is checked per item, by date, cumulatively,
                so 56 on hand cannot cover two lines of 40 before the restock. An item's list price is
                set by its total quantity in the quote, however it is split. A price under list (but not
                under the floor) is allowed and comes back with a notice.
  ask_owner     an internal question to the owner; the answer is the owner's fixed policy for the ticket
  send_reply    the customer's answer comes back in the result

The desk never reads the meaning of a message. The customer's answer is a fixed paragraph
chosen by what was sent: no quote (a question or a note), or quotes offered (attached, or named
by id in the message). An offered quote is checked against the customer's hidden conditions,
the same way every time. In this test bed the owner approves replies at once, so the customer
answers within the episode; in the assist window nothing reaches a customer before approval.
"""
from __future__ import annotations

import difflib
import math
import re
from datetime import date

from .desk import _fn, get_cost, lookup_quote, meets_floor, unit_price_for

ENV_VERSION = "quote-desk/2"
MAX_LINES = 6
MAX_LINES_PER_ITEM = 2

SYSTEM_PROMPT = (
    "You handle tickets at the quote desk of a small custom print shop. "
    "Use the tools to look things up, prepare quotes, and write to the customer or the owner. "
    "Each quote line has its own ship date. "
    "List prices come from the price calculator; a price under list must stay at or above the floor price "
    "and within your discount authority, or have the owner's okay. "
    "In this test bed the owner approves replies right away, so the customer's answer comes back in the result. "
    "When you have finished with the ticket, answer with: Done."
)

QUOTE_ID = re.compile(r"\bQ-\d+-\d+\b", re.IGNORECASE)


def floor_price(unit_cost: float) -> float:
    """Lowest unit price the margin floor allows, rounded up to the cent."""
    p = math.ceil(unit_cost / (1 - 0.30) * 100 - 1e-6) / 100
    return p if meets_floor(p, unit_cost) else p + 0.01


_SKU = {"type": "string", "description": "Catalog item code, e.g. MUG-11W."}
_QTY = {"type": "integer", "description": "Number of units."}
_TICKET = {"type": "string"}

TOOL_SPECS = [
    _fn("lookup_quote",
        "Look up an earlier quote by id: its status, the date it is valid until, and its lines.",
        {"quote_id": {"type": "string"}}, ["quote_id"]),
    _fn("search_catalog",
        "Search the catalog by words in an item's name or code, e.g. 'jute' or 'tote'. Empty lists every item.",
        {"query": {"type": "string"}}, ["query"]),
    _fn("get_cost",
        "Get the current unit cost of a catalog item.",
        {"sku": _SKU}, ["sku"]),
    _fn("check_stock",
        "Check stock for a quantity. When it is short, shows the restock date and quantity, and substitutes with stock.",
        {"sku": _SKU, "qty": _QTY}, ["sku", "qty"]),
    _fn("calc_price",
        "Price one quote line from its unit cost: the list price from the shop's markup and quantity breaks, "
        "and the floor price under which the shop cannot sell.",
        {"sku": _SKU, "qty": _QTY, "unit_cost": {"type": "number"}}, ["sku", "qty", "unit_cost"]),
    _fn("create_quote",
        "Create a draft quote for a ticket. Each line has its own ship date. Rejects prices under the floor "
        "and dates the stock cannot meet; notes any price under list.",
        {
            "ticket_id": _TICKET,
            "lines": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"sku": _SKU, "qty": _QTY, "unit_price": {"type": "number"},
                                   "ship_by": {"type": "string", "description": "Ship date, YYYY-MM-DD."}},
                    "required": ["sku", "qty", "unit_price", "ship_by"],
                },
            },
        },
        ["ticket_id", "lines"]),
    _fn("ask_owner",
        "Ask the shop owner a question about a ticket, e.g. about a discount. Internal; the customer does not see it.",
        {"ticket_id": _TICKET, "question": {"type": "string"}}, ["ticket_id", "question"]),
    _fn("send_reply",
        "Send a reply to the customer on a ticket, optionally attaching a draft quote. Returns the customer's answer.",
        {
            "ticket_id": _TICKET,
            "message": {"type": "string"},
            "quote_id": {"type": "string", "description": "Draft quote to attach, if any."},
        },
        ["ticket_id", "message"]),
]
SPEC_BY_NAME = {s["function"]["name"]: s["function"] for s in TOOL_SPECS}


def _unknown_sku(state, sku):
    names = difflib.get_close_matches(sku.upper(), list(state["catalog"]), n=3, cutoff=0.6)
    return {"error": "unknown_sku", "sku": sku,
            "similar": [{"sku": s, "name": state["catalog"][s]["name"], "on_hand": state["catalog"][s]["on_hand"]}
                        for s in names]}


def search_catalog(state, query):
    """Items whose code or name holds every word of the query (a person asked for it, 10-29)."""
    words = [w for w in re.split(r"[\s,]+", query.lower()) if w]
    hits = sorted(s for s, it in state["catalog"].items() if all(w in f"{s} {it['name']}".lower() for w in words))
    return {"query": query, "items": [{"sku": s, "name": state["catalog"][s]["name"]} for s in hits[:60]],
            "more": max(0, len(hits) - 60)}


def check_stock(state, sku, qty):
    item = state["catalog"].get(sku)
    if item is None:
        return _unknown_sku(state, sku)
    enough = item["on_hand"] >= qty
    result = {"sku": sku, "name": item["name"], "qty": qty, "on_hand": item["on_hand"], "enough": enough}
    if not enough:
        result["restock_date"] = item["restock_date"]
        result["restock_qty"] = item["restock_qty"] if item["restock_date"] else 0
        result["substitutes"] = [
            {"sku": s, "name": state["catalog"][s]["name"], "on_hand": state["catalog"][s]["on_hand"]}
            for s in item["substitutes"] if s in state["catalog"] and state["catalog"][s]["on_hand"] > 0
        ]
    return result


def calc_price(state, sku, qty, unit_cost):
    if sku not in state["catalog"]:
        return _unknown_sku(state, sku)
    if qty <= 0 or unit_cost <= 0:
        return {"error": "invalid_value"}
    price = unit_price_for(unit_cost, qty)
    return {"sku": sku, "qty": qty, "unit_cost": unit_cost, "unit_price": price,
            "line_total": round(price * qty, 2), "margin_pct": round((price - unit_cost) / price * 100, 1),
            "floor_unit_price": floor_price(unit_cost)}


def list_price(state, sku: str, lines: list[dict]) -> float:
    """The list price of an item in a quote: priced at its total quantity across the quote's lines."""
    return unit_price_for(state["catalog"][sku]["unit_cost"], sum(l["qty"] for l in lines if l["sku"] == sku))


def approved_price(state, sku: str, price: float) -> bool:
    """Is this the price of the item on an approved quote that is still valid?"""
    return any(q["valid_until"] >= state["today"] and l["sku"] == sku and abs(l["unit_price"] - price) < 0.005
               for q in state["quotes"].values() for l in q["lines"])


def owner_agreed(state, ticket_id: str) -> bool:
    """The owner was asked on this ticket and allows going down to the floor."""
    return ticket_id in state["owner_asked"] and state["owner"].get(ticket_id, {}).get("policy") == "floor_ok"


def available(item: dict, ship_by: str) -> int:
    """Units of an item that can have shipped by a date: on hand, plus the restock once it arrives."""
    restocked = item["restock_date"] is not None and ship_by >= item["restock_date"]
    return item["on_hand"] + (item["restock_qty"] if restocked else 0)


def stock_problem(item: dict, lines: list[dict]) -> str | None:
    """The first ship date by which these lines of one item need more units than the desk will have."""
    for when in sorted({l["ship_by"] for l in lines}):
        need = sum(l["qty"] for l in lines if l["ship_by"] <= when)
        if need > available(item, when):
            return when
    return None


def create_quote(state, ticket_id, lines):
    if ticket_id not in state["tickets"]:
        return {"error": "unknown_ticket", "ticket_id": ticket_id}
    if not lines:
        return {"error": "no_lines"}
    if len(lines) > MAX_LINES:
        return {"error": "too_many_lines", "max": MAX_LINES}
    for line in lines:
        try:
            date.fromisoformat(line["ship_by"])
        except ValueError:
            return {"error": "invalid_date", "ship_by": line["ship_by"]}
        if line["ship_by"] < state["today"]:
            return {"error": "ship_by_in_past", "ship_by": line["ship_by"]}
        if line["sku"] not in state["catalog"]:
            return _unknown_sku(state, line["sku"])
        if line["qty"] <= 0 or line["unit_price"] <= 0:
            return {"error": "invalid_value", "sku": line["sku"]}
        if not meets_floor(line["unit_price"], state["catalog"][line["sku"]]["unit_cost"]):
            return {"error": "below_margin_floor", "sku": line["sku"]}
    by_sku: dict[str, list[dict]] = {}
    for line in lines:
        by_sku.setdefault(line["sku"], []).append(line)
    for sku, mine in by_sku.items():
        if len(mine) > MAX_LINES_PER_ITEM:
            return {"error": "too_many_lines_for_item", "sku": sku, "max": MAX_LINES_PER_ITEM}
        when = stock_problem(state["catalog"][sku], mine)
        if when:
            return {"error": "stock_short_for_ship_date", "sku": sku, "ship_by": when,
                    "available_by_then": available(state["catalog"][sku], when)}
    allowed = state.get("authority", {}).get(ticket_id, 0.0)
    out, notices = [], []
    for line in lines:
        item = state["catalog"][line["sku"]]
        out.append({"sku": line["sku"], "qty": line["qty"], "unit_price": line["unit_price"],
                    "ship_by": line["ship_by"], "line_total": round(line["unit_price"] * line["qty"], 2)})
        listed = list_price(state, line["sku"], lines)
        if line["unit_price"] < listed - 1e-9 and not approved_price(state, line["sku"], line["unit_price"]):
            off = (listed - line["unit_price"]) / listed
            if off <= allowed + 1e-9:
                why = "within your discount authority."
            elif owner_agreed(state, ticket_id):
                why = "beyond your discount authority; the owner agreed."
            else:
                why = "beyond your discount authority, owner approval needed."
            notices.append(f"{line['sku']} at {line['unit_price']:.2f} is {off * 100:.1f}% under list "
                           f"({listed:.2f}); " + why)
    quote_id = f"Q-{ticket_id.split('-')[-1]}-{len(state['drafts']) + 1}"
    state["drafts"][quote_id] = {"ticket_id": ticket_id, "lines": out}
    result = {"quote_id": quote_id, "ticket_id": ticket_id, "lines": out,
              "total": round(sum(l["line_total"] for l in out), 2), "status": "draft"}
    if notices:
        result["notices"] = notices
    return result


def ask_owner(state, ticket_id, question):
    if ticket_id not in state["tickets"]:
        return {"error": "unknown_ticket", "ticket_id": ticket_id}
    if not question.strip():
        return {"error": "empty_message"}
    owner = state.get("owner", {}).get(ticket_id) or {}
    state["owner_asked"].append(ticket_id)
    return {"owner_reply": owner.get("says", "Use your judgement within the shop's rules.")}


def offered_quotes(state, ticket_id, message, quote_id=None) -> list[str]:
    """Draft quotes the customer is shown: the attached one first, then any named by id in the message."""
    named = [q.upper() for q in QUOTE_ID.findall(message)]
    ids = ([quote_id] if quote_id else []) + named
    return [q for q in dict.fromkeys(ids) if q in state["drafts"] and state["drafts"][q]["ticket_id"] == ticket_id]


PLAIN_CUSTOMER = {"wants": None, "says": {"answer": "Thanks, I'll look out for the quote.",
                                           "decline": "That isn't what we asked for."}}


def customer_problem(state, ticket_id, lines) -> str | None:
    """Why the customer would turn these quote lines down, or None. Prices are not judged here."""
    c = state["customers"].get(ticket_id, PLAIN_CUSTOMER)
    if c["wants"] is None:
        return None
    need_by = c.get("need_by")
    used = set()
    for want in c["wants"]:
        ok_skus = {want["sku"], *(want.get("substitutes", []) if c.get("substitute_ok") else [])}
        mine = [i for i, l in enumerate(lines) if l["sku"] in ok_skus and i not in used]
        if sum(lines[i]["qty"] for i in mine) != want["qty"]:
            return "items"
        late = [i for i in mine if need_by and lines[i]["ship_by"] > need_by]
        if late and len(late) < len(mine) and not c.get("split_ok"):
            return "date"
        if late and len(late) == len(mine) and not c.get("later_ok"):
            return "date"
        used.update(mine)
    if len(used) != len(lines):
        return "items"
    return None


def customer_answer(state, ticket_id, message, quote_id=None) -> dict:
    c = state["customers"].get(ticket_id, PLAIN_CUSTOMER)
    talk = state["talk"].setdefault(ticket_id, {"asked": 0, "pushed": False, "accepted": None})
    says = c["says"]
    if talk["accepted"]:
        return {"customer_reply": says.get("after", "We already confirmed {qid}, thanks.").format(qid=talk["accepted"])}
    offered = offered_quotes(state, ticket_id, message, quote_id)
    if not offered:
        talk["asked"] += 1
        again = says["answer"] if says["answer"].startswith("As I said") else "As I said: " + says["answer"]
        return {"customer_reply": says["answer"] if talk["asked"] == 1 else says.get("again", again)}
    for qid in offered:
        lines = state["drafts"][qid]["lines"]
        if customer_problem(state, ticket_id, lines):
            continue
        total = round(sum(l["line_total"] for l in lines), 2)
        if c.get("push_back_over") is not None and total > c["push_back_over"] + 1e-9 and not talk["pushed"]:
            talk["pushed"] = True
            return {"customer_reply": says["push_back"]}
        talk["accepted"] = qid
        return {"customer_reply": says.get("accept", "Thanks, please go ahead with {qid}.").format(qid=qid),
                "customer_accepted": qid}
    return {"customer_reply": says["decline"]}


def send_reply(state, ticket_id, message, quote_id=None):
    if ticket_id not in state["tickets"]:
        return {"error": "unknown_ticket", "ticket_id": ticket_id}
    if not message.strip():
        return {"error": "empty_message"}
    if quote_id is not None:
        quote_id = quote_id.upper()
        draft = state["drafts"].get(quote_id)
        if draft is None or draft["ticket_id"] != ticket_id:
            return {"error": "unknown_quote", "quote_id": quote_id}
    state["outbox"].append({"ticket_id": ticket_id, "message": message, "quote_id": quote_id})
    return {"status": "sent", "ticket_id": ticket_id, "reply_no": len(state["outbox"]),
            **customer_answer(state, ticket_id, message, quote_id)}


HANDLERS = {
    "lookup_quote": lookup_quote,
    "search_catalog": search_catalog,
    "get_cost": get_cost,
    "check_stock": check_stock,
    "calc_price": calc_price,
    "create_quote": create_quote,
    "ask_owner": ask_owner,
    "send_reply": send_reply,
}


def empty_state(today: str) -> dict:
    return {"today": today, "tickets": [], "catalog": {}, "quotes": {}, "drafts": {}, "outbox": [],
            "authority": {}, "owner": {}, "owner_asked": [], "customers": {}, "talk": {}}


def authority_text(pct: float) -> str:
    if pct <= 0:
        return "Your discount authority: none. Ask the owner before offering any price under list."
    return (f"Your discount authority: up to {pct * 100:.0f}% under list on your own. "
            "Beyond that, ask the owner. Never under the floor price.")


def render_view(view: dict) -> str:
    parts = [f"Today is {view['today']}.", f"Ticket {view['ticket_id']} from {view['from']}.",
             authority_text(view.get("authority", 0.0)),
             f"Sent to this customer so far: {view.get('sent_so_far') or 'nothing yet'}."]
    if view.get("thread"):
        lines = [f"- {m['date']}, {m['who']}: {m['text']}" for m in view["thread"]]
        parts.append("Earlier in this thread:\n" + "\n".join(lines))
    parts.append("New message:\n" + view["body"])
    return "\n\n".join(parts)

