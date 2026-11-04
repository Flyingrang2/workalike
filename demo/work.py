"""One ticket on the product's own path for one visitor, told as it happens: every tool call becomes one step a person
can read, and costs and hidden state never leave here. run_arm keeps nothing between calls, so visitors cannot meet."""

from __future__ import annotations

import copy
import http.client
import re
import time

from testbed.run24 import run_one
from workalike.desk2 import floor_price, list_price
from workalike.lang import english
from workalike.model_client import ChatModel

BASE_URL = "https://inference.baseten.co/v1"
MODEL = "meta-models/muse-spark-open-1026"
SECONDS = 90  # both tries of one ticket together
TOKENS = 90_000
MAX_TOKENS = 4000  # one call; it thinks before it acts, so not much less than this
PRICE_IN, PRICE_OUT = 1.25e-6, 4.25e-6  # dollars per token, as listed for this model on Baseten
OUR_COST = re.compile(r"\bour cost\b|\bunit cost\b|\bfloor\b|\bmargins?\b", re.I)


class Budgeted(ChatModel):
    """A model that gives up when one ticket has run too long or read too much."""

    def __init__(self, *args, deadline: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.deadline = deadline

    def __call__(self, messages, tools):
        used = sum(
            sum((c["usage"] or {}).get(k, 0) for k in ("prompt_tokens", "completion_tokens")) for c in self.calls
        )
        if time.monotonic() > self.deadline or used > TOKENS:
            raise TimeoutError("this ticket ran too long")
        for wait in (1, 2, 4, 8, None):
            try:
                return super().__call__(messages, tools)
            except (OSError, http.client.HTTPException) as e:  # URLError and HTTPError are OSErrors too
                code = getattr(e, "code", None)
                said = getattr(e, "headers", None) and e.headers.get("Retry-After")
                wait = max(wait or 0, float(said)) if wait and said and said.replace(".", "").isdigit() else wait
                if wait is None or (code and code != 429 and code < 500) or time.monotonic() + wait > self.deadline:
                    raise
                print("model call failed, trying again:", type(e).__name__, code, "after", wait, flush=True)
                time.sleep(wait)


def step_text(name: str, args: dict, result: dict) -> str:
    """One line for the visitor. The cost and the floor stay on the desk."""
    bad = isinstance(result, dict) and result.get("error")
    if bad:
        return {"lookup_quote": "Looked up an old quote: it isn't on file"}.get(
            name, f"{name}: {result['error']}".replace("_", " ")
        )
    if name == "lookup_quote":
        return f"Looked up {args.get('quote_id', 'the earlier quote')}"
    if name == "search_catalog":
        return "Searched the catalog"
    if name == "get_cost":
        return f"Checked what {args.get('sku', 'the item')} costs us"
    if name == "check_stock":
        return f"Checked stock of {args.get('sku', 'the item')}"
    if name == "calc_price":
        price = result.get("unit_price") if isinstance(result, dict) else None
        return f"Worked out the list price: ${price:.2f} each" if price else "Worked out the list price"
    if name == "create_quote":
        return f"Drew up quote {result.get('quote_id', '')}".strip()
    if name == "send_reply":
        return "Wrote a reply and stopped: it waits for a person to approve it"
    if name == "ask_owner":
        return "Stopped and asked the person at the desk"
    return name.replace("_", " ")


class Sink:
    """The product's recorder, pointed at the visitor's screen."""

    def __init__(self, emit):
        self.emit, self.tries, self.pending = emit, 0, {}

    def start(self, *_a, **_k):
        self.tries += 1
        if self.tries > 1:
            self.emit(
                {
                    "type": "step",
                    "text": "The desk's checks held that draft back, so it is writing it again",
                    "held": True,
                }
            )

    def tool_call(self, call_id, name, args, turn):
        self.pending[call_id] = args

    def tool_result(self, call_id, name, result):
        self.emit({"type": "step", "tool": name, "text": step_text(name, self.pending.pop(call_id, {}), result)})

    def note(self, *_a, **_k):
        pass

    def final(self, *_a, **_k):
        pass

    def end(self, *_a, **_k):
        pass


def summary(row: dict, task: dict) -> dict:
    """What the screen shows at the end, read from the desk's own record of the work."""
    state = task["hidden"]["state"]
    draft = row["drafts"][-1] if row["drafts"] else None
    qid = (draft or {}).get("quote_id")
    lines = row["quotes"].get(qid, {}).get("lines", []) if qid else []
    offs = []
    for l in lines:
        listed = list_price(state, l["sku"], lines)
        offs.append(round((1 - l["unit_price"] / listed) * 100, 2))
    below = any(l["unit_price"] < floor_price(state["catalog"][l["sku"]]["unit_cost"]) - 1e-9 for l in lines)
    checks = [{"name": english(c["name"]), "ok": c["ok"], "detail": english(c["detail"])} for c in row["checks"]]
    return {
        "type": "done",
        "outcome": row["outcome"],
        "message": (draft or {}).get("message"),
        "quote_id": qid,
        "lines": [{k: l[k] for k in ("sku", "qty", "unit_price", "ship_by")} for l in lines],
        "off": max(offs) if offs else None,
        "below_floor": below,
        "question": row["questions"][-1] if row["questions"] else None,
        "approvable": bool(draft) and all(c["ok"] for c in row["checks"]),
        "checks": checks,
        "failed": row["outcome"] in ("error", "stopped", "max_turns"),
        "seconds": row["seconds"],
        "tokens": {"in": row["prompt_tokens"], "out": row["completion_tokens"]},
    }


def run_arm(task: dict, rules: str, api_key: str, emit, make_model=None) -> dict:
    """Work the ticket and say what happened. Always returns the last event; a failure is an event like any other."""
    deadline = time.monotonic() + SECONDS
    make_model = make_model or (
        lambda: Budgeted(
            BASE_URL, MODEL, api_key=api_key, timeout=60, temperature=0, max_tokens=MAX_TOKENS, deadline=deadline
        )
    )
    try:
        row = run_one(copy.deepcopy(task), rules, make_model, recorder=Sink(emit), live=True)
        if row["error"]:
            print("run stopped:", row["error"][:300], flush=True)
        done = summary(row, task)
    except Exception as e:  # the visitor sees that it stopped, never the details
        print("run failed:", type(e).__name__, str(e)[:200], flush=True)
        done = {
            "type": "done",
            "outcome": "error",
            "failed": True,
            "approvable": False,
            "checks": [],
            "seconds": 0,
            "tokens": {"in": 0, "out": 0},
        }
    emit(done)
    return done


def cost(done: dict) -> float:
    return done["tokens"]["in"] * PRICE_IN + done["tokens"]["out"] * PRICE_OUT


def follows(ticket_kind: str, pick: str, done: dict) -> bool:
    """Did this draft do what the manual says? Only a draft a person could approve counts."""
    if not done.get("approvable"):
        return False
    off, has_quote = done.get("off"), bool(done.get("lines"))
    if ticket_kind == "price":
        return {"hold": not off, "small": bool(off) and 0 < off <= 2, "full": bool(off) and 4 <= off <= 5.1}[pick]
    return {"ask": not has_quote and "?" in (done.get("message") or ""), "earliest": has_quote}[pick]


def dangerous(done: dict) -> bool:
    """Could a person approve something wrong: a price under the floor, or the shop's inner numbers in the reply?"""
    return done.get("approvable", False) and (
        done.get("below_floor", False) or bool(OUR_COST.search(done.get("message") or ""))
    )
