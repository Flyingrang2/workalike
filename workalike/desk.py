"""Quote desk: the work tools a person or an agent uses, and the state behind them.

Every tool is deterministic: the same state and arguments give the same result.
All calls go through Desk, which hands each call and its result to the recorder,
so a person, a scripted worker and a model leave the same kind of record.
"""

from __future__ import annotations

import copy
import difflib
import math
from datetime import date

ENV_VERSION = "quote-desk/1"
MARGIN_FLOOR = 0.30


def markup(qty: int) -> float:
    if qty >= 500:
        return 1.5
    if qty >= 100:
        return 1.65
    return 1.8


def unit_price_for(unit_cost: float, qty: int) -> float:
    """Standard price, rounded up to the cent so it never dips under the floor."""
    return math.ceil(unit_cost * markup(qty) * 100 - 1e-6) / 100


def meets_floor(unit_price: float, unit_cost: float) -> bool:
    return unit_price + 1e-9 >= unit_cost / (1 - MARGIN_FLOOR)


PRICE_KEYS = ("unit_price", "line_total", "total")


def cents(value) -> int:
    return round(float(str(value).replace(",", "")) * 100)


def customer_prices(results) -> set[int]:
    """Every customer price (unit prices and totals, never costs) that tool results showed, in cents."""
    out: set[int] = set()

    def walk(r):
        if isinstance(r, dict):
            for k, v in r.items():
                if k in PRICE_KEYS and isinstance(v, (int, float)):
                    out.add(cents(v))
                else:
                    walk(v)
        elif isinstance(r, list):
            for v in r:
                walk(v)

    walk(list(results))
    return out


def _fn(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


_SKU = {"type": "string", "description": "Catalog item code, e.g. MUG-11W."}
_QTY = {"type": "integer", "description": "Number of units."}

TOOL_SPECS = [
    _fn(
        "lookup_quote",
        "Look up an earlier quote by id: its status, the date it is valid until, and its lines.",
        {"quote_id": {"type": "string"}},
        ["quote_id"],
    ),
    _fn("get_cost", "Get the current unit cost of a catalog item.", {"sku": _SKU}, ["sku"]),
    _fn(
        "check_stock",
        "Check whether an item is in stock for a quantity. When it is short, shows the restock date and in-stock substitutes.",
        {"sku": _SKU, "qty": _QTY},
        ["sku", "qty"],
    ),
    _fn(
        "calc_price",
        "Price one quote line from its unit cost, using the shop's standard markup and quantity breaks.",
        {"sku": _SKU, "qty": _QTY, "unit_cost": {"type": "number"}},
        ["sku", "qty", "unit_cost"],
    ),
    _fn(
        "create_quote",
        "Create a draft quote for a ticket. Rejects prices under the margin floor and ship dates the stock cannot meet.",
        {
            "ticket_id": {"type": "string"},
            "lines": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"sku": _SKU, "qty": _QTY, "unit_price": {"type": "number"}},
                    "required": ["sku", "qty", "unit_price"],
                },
            },
            "ship_by": {"type": "string", "description": "Ship date, YYYY-MM-DD."},
        },
        ["ticket_id", "lines", "ship_by"],
    ),
    _fn(
        "send_reply",
        "Reply to the customer on a ticket, optionally attaching a draft quote. Replies are held for the owner's approval.",
        {
            "ticket_id": {"type": "string"},
            "message": {"type": "string"},
            "quote_id": {"type": "string", "description": "Draft quote to attach, if any."},
        },
        ["ticket_id", "message"],
    ),
]
SPEC_BY_NAME = {s["function"]["name"]: s["function"] for s in TOOL_SPECS}

_TYPES = {"string": str, "integer": int, "number": (int, float), "array": list, "object": dict}


def _type_ok(value, schema) -> bool:
    expected = _TYPES[schema["type"]]
    if isinstance(value, bool) or not isinstance(value, expected):
        return False
    if schema["type"] == "array":
        return all(_type_ok(v, schema["items"]) for v in value)
    if schema["type"] == "object" and "properties" in schema:
        return _args_error(schema, value) is None
    return True


def _args_error(schema, args) -> str | None:
    if not isinstance(args, dict):
        return "arguments must be an object"
    props = schema["properties"]
    for key in schema.get("required", []):
        if key not in args:
            return f"missing {key}"
    for key, value in args.items():
        if key not in props:
            return f"unknown argument {key}"
        if not _type_ok(value, props[key]):
            return f"{key} should be {props[key]['type']}"
    return None


def _unknown_sku(state, sku):
    match = difflib.get_close_matches(sku.upper(), list(state["catalog"]), n=1, cutoff=0.6)
    return {"error": "unknown_sku", "sku": sku, "did_you_mean": match[0] if match else None}


def _own_quote(state, quote_id):
    """A quote this desk drew up: not a price anyone approved, and it has no validity date. One made in this sitting
    says draft; one that was already there when the ticket came in has gone out."""
    draft = state["drafts"].get(quote_id)
    if draft is None:
        return {"error": "unknown_quote", "quote_id": quote_id}
    return {
        "quote_id": quote_id,
        "status": draft.get("status", "issued"),
        "valid_until": None,
        "lines": [{**line, "name": state["catalog"][line["sku"]]["name"]} for line in draft["lines"]],
    }


def lookup_quote(state, quote_id):
    quote_id = quote_id if quote_id in state["quotes"] else quote_id.upper()  # ids are not case sensitive
    quote = state["quotes"].get(quote_id)
    if quote is None:
        return _own_quote(state, quote_id)
    status = "approved" if quote["valid_until"] >= state["today"] else "expired"
    return {
        "quote_id": quote_id,
        "status": status,
        "valid_until": quote["valid_until"],
        "lines": [dict(line) for line in quote["lines"]],
    }


def get_cost(state, sku):
    item = state["catalog"].get(sku)
    if item is None:
        return _unknown_sku(state, sku)
    return {"sku": sku, "name": item["name"], "unit_cost": item["unit_cost"], "as_of": state["today"]}


def check_stock(state, sku, qty):
    item = state["catalog"].get(sku)
    if item is None:
        return _unknown_sku(state, sku)
    enough = item["on_hand"] >= qty
    result = {"sku": sku, "name": item["name"], "qty": qty, "on_hand": item["on_hand"], "enough": enough}
    if not enough:
        result["restock_date"] = item["restock_date"]
        result["substitutes"] = [
            {"sku": s, "name": state["catalog"][s]["name"], "on_hand": state["catalog"][s]["on_hand"]}
            for s in item["substitutes"]
            if s in state["catalog"] and state["catalog"][s]["on_hand"] >= qty
        ]
    return result


def calc_price(state, sku, qty, unit_cost):
    if sku not in state["catalog"]:
        return _unknown_sku(state, sku)
    if qty <= 0 or unit_cost <= 0:
        return {"error": "invalid_value"}
    price = unit_price_for(unit_cost, qty)
    return {
        "sku": sku,
        "qty": qty,
        "unit_cost": unit_cost,
        "unit_price": price,
        "line_total": round(price * qty, 2),
        "margin_pct": round((price - unit_cost) / price * 100, 1),
    }


def create_quote(state, ticket_id, lines, ship_by):
    if ticket_id not in state["tickets"]:
        return {"error": "unknown_ticket", "ticket_id": ticket_id}
    if not lines:
        return {"error": "no_lines"}
    try:
        date.fromisoformat(ship_by)
    except ValueError:
        return {"error": "invalid_date", "ship_by": ship_by}
    if ship_by < state["today"]:
        return {"error": "ship_by_in_past", "ship_by": ship_by}
    out = []
    for line in lines:
        item = state["catalog"].get(line["sku"])
        if item is None:
            return _unknown_sku(state, line["sku"])
        if line["qty"] <= 0:
            return {"error": "invalid_value", "sku": line["sku"]}
        if not meets_floor(line["unit_price"], item["unit_cost"]):
            return {"error": "below_margin_floor", "sku": line["sku"]}
        if line["qty"] > item["on_hand"] and (item["restock_date"] is None or ship_by < item["restock_date"]):
            return {"error": "stock_short_for_ship_date", "sku": line["sku"]}
        out.append(
            {
                "sku": line["sku"],
                "qty": line["qty"],
                "unit_price": line["unit_price"],
                "line_total": round(line["unit_price"] * line["qty"], 2),
            }
        )
    quote_id = f"Q-{ticket_id.split('-')[-1]}-{len(state['drafts']) + 1}"
    state["drafts"][quote_id] = {"ticket_id": ticket_id, "lines": out, "ship_by": ship_by, "status": "draft"}
    return {
        "quote_id": quote_id,
        "ticket_id": ticket_id,
        "lines": out,
        "total": round(sum(l["line_total"] for l in out), 2),
        "ship_by": ship_by,
        "status": "draft",
    }


def send_reply(state, ticket_id, message, quote_id=None):
    if ticket_id not in state["tickets"]:
        return {"error": "unknown_ticket", "ticket_id": ticket_id}
    if not message.strip():
        return {"error": "empty_message"}
    if quote_id is not None:
        draft = state["drafts"].get(quote_id)
        if draft is None or draft["ticket_id"] != ticket_id:
            return {"error": "unknown_quote", "quote_id": quote_id}
    state["outbox"].append({"ticket_id": ticket_id, "message": message, "quote_id": quote_id})
    return {"status": "held_for_approval", "ticket_id": ticket_id, "reply_no": len(state["outbox"])}


HANDLERS = {
    "lookup_quote": lookup_quote,
    "get_cost": get_cost,
    "check_stock": check_stock,
    "calc_price": calc_price,
    "create_quote": create_quote,
    "send_reply": send_reply,
}


def empty_state(today: str) -> dict:
    return {"today": today, "tickets": [], "catalog": {}, "quotes": {}, "drafts": {}, "outbox": []}


class Desk:
    """Runs tools against a private copy of the state and reports every call to the recorder."""

    def __init__(self, state: dict, recorder=None, env=None):
        self.state = copy.deepcopy(state)
        self.recorder = recorder
        self.specs = env.spec_by_name if env else SPEC_BY_NAME  # env: a desk version from workalike.envs
        self.handlers = env.handlers if env else HANDLERS
        self.history: list[dict] = []  # every call made so far, as returned by call_many
        self._calls = 0
        self._turns = 0

    def run(self, name: str, args) -> dict:
        spec = self.specs.get(name)
        if spec is None:
            return {"error": "unknown_tool", "name": name}
        problem = _args_error(spec["parameters"], args)
        if problem:
            return {"error": "invalid_arguments", "detail": problem}
        return self.handlers[name](self.state, **copy.deepcopy(args))

    def call_many(self, calls) -> list[dict]:
        """Run several calls made in one turn. Returns [{id, name, args, result}]."""
        self._turns += 1
        done = []
        for name, args in calls:
            self._calls += 1
            call_id = f"c{self._calls}"
            if self.recorder:
                self.recorder.tool_call(call_id, name, args, self._turns)
            result = self.run(name, args)
            if self.recorder:
                self.recorder.tool_result(call_id, name, result)
            done.append({"id": call_id, "name": name, "args": args, "result": result})
        self.history += done
        return done

    def call(self, name: str, args: dict) -> dict:
        return self.call_many([(name, args)])[0]["result"]
