"""Work a new request with the rules a person confirmed, and stop where they run out.

    job = work(view, state, model, rules_text)   # {"outcome", "drafts", "questions", "used", "history", ...}
    checks = guard(job["state"], ticket_id, job["history"])

The model sees the quote desk with the same tools as in the test bed. Two things make it the product rather than
the simulation: a reply to the customer is held as a draft for the person to approve, and a question to the owner
goes to the person. Either one ends the run, and nothing after it is carried out. The customers and the owner the
test bed scripts are taken out of the state, so no hidden answer can leak into a draft.

guard() reads held drafts against the desk before anything can be approved: amounts and quote ids must come from
the desk, prices stay at list or within the discount authority (or the person's okay), never under the floor,
and every line has the stock by its ship date. The model's words never decide whether a draft is safe.
"""
from __future__ import annotations

import copy
import dataclasses
import re
import unicodedata
import secrets

from . import agent
from .desk import Desk, cents, customer_prices
from .desk2 import approved_price, floor_price, list_price, owner_agreed, stock_problem, within_authority
from .envs import V2

PRODUCT_SYSTEM = (
    "You handle tickets at the quote desk of a small custom print shop. "
    "Use the tools to look things up, prepare quotes, and write to the customer or the owner. "
    "Each quote line has its own ship date. "
    "List prices come from the price calculator; a price under list must stay at or above the floor price "
    "and within your discount authority, or have the owner's okay. "
    "A reply to the customer is held as a draft until the person at this desk approves it, and a question to the "
    "owner goes to that person; either one ends your work on the ticket for now. "
    "When you have finished with the ticket, answer with: Done."
)
RULES_USED = ("Your work on this ticket is held for the person at this desk. Which of the confirmed rules did you "
              "follow? Answer with one line only: Rules used: R1, R4 (or: Rules used: none)")
# Every amount a reply can state: $12.50, $.50, 12.50 USD, 12 dollars (after NFKC, so a full-width ＄ counts too).
AMOUNT = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?|\.\d+)|(?<![\w.])(\d[\d,]*(?:\.\d+)?|\.\d+)\s?(?:usd|dollars?)\b",
                    re.IGNORECASE)
QUOTE_ID = re.compile(r"\bQ-\d+-\d+\b", re.IGNORECASE)
RULE_ID = re.compile(r"\bR(\d+)\b")
# What a held-back draft is told before it is written again, by the check that failed.
HINTS = {"답장 초안": "Finish with a reply to the customer or a question to the owner.",
         "가격은 견적으로만": "Put prices only on an attached quote; a reply without a quote states no price.",
         "원가 비공개": "Never tell the customer our unit cost.",
         "금액은 데스크 값": "State only customer prices the desk gave you (list, quote or approved prices), "
                         "never a cost or the floor price.",
         "견적번호 확인": "Name only quotes that exist for this ticket or the customer's earlier quotes.",
         "quote": "Every quote must keep list price or a discount within your authority (or the owner's okay), "
                  "never under the floor, with stock for each ship date."}


def _known_ticket(state, ticket_id):
    return None if ticket_id in state["tickets"] else {"error": "unknown_ticket", "ticket_id": ticket_id}


def held_reply(state, ticket_id, message, quote_id=None):
    """send_reply in the product: kept as a draft for approval, never sent from here."""
    problem = _known_ticket(state, ticket_id)
    if problem:
        return problem
    if not message.strip():
        return {"error": "empty_message"}
    if quote_id is not None:
        quote_id = quote_id.upper()
        draft = state["drafts"].get(quote_id)
        if draft is None or draft["ticket_id"] != ticket_id:
            return {"error": "unknown_quote", "quote_id": quote_id}
    state["outbox"].append({"ticket_id": ticket_id, "message": message, "quote_id": quote_id})
    state["stop"] = state["stop"] or "draft"
    return {"status": "held", "ticket_id": ticket_id, "reply_no": len(state["outbox"]),
            "note": "Held for the person at this desk to approve; nothing was sent. Your work here ends now."}


def ask_person(state, ticket_id, question):
    """ask_owner in the product: the person at the desk is the owner. An answer they gave earlier comes back;
    otherwise the question waits for them and the run stops."""
    problem = _known_ticket(state, ticket_id)
    if problem:
        return problem
    if not question.strip():
        return {"error": "empty_message"}
    state["owner_asked"].append(ticket_id)
    answer = state["answers"].get(ticket_id)
    if answer:
        state["owner"][ticket_id] = {"policy": answer.get("policy"), "says": answer["says"]}
        return {"owner_reply": answer["says"]}
    state["questions"].append({"ticket_id": ticket_id, "question": question})
    state["stop"] = "asked"
    return {"status": "asked", "note": "Your question waits for the person at this desk. Your work here ends now."}


STREAK = 6  # a ticket has up to 6 lines, so this many calls of one tool in a row can be honest; more is a loop


def guarded(name, handler):
    """Nothing runs once the ticket waits for the person, and a tool called over and over is refused with the way out.
    A model that forgets its reasoning between turns can walk down a price a cent at a time, or call a broken tool
    until the turns run out (11-02: 21 quotes on one ticket, 22 calls of a tool that was down)."""
    def run(state, **args):
        if state.get("stop"):
            return {"error": "stopped", "detail": "this ticket is waiting for the person"}
        streak = state.setdefault("streak", {"name": None, "n": 0})
        streak.update(name=name, n=streak["n"] + 1 if streak["name"] == name else 1)
        if streak["n"] > STREAK:
            return {"error": "repeated_call",
                    "detail": f"You have called {name} {streak['n'] - 1} times in a row and it will not help. Do not call "
                              "it again. Reply to the customer, or tell the owner with ask_owner what is blocking you."}
        return handler(state, **args)
    return run


PRODUCT_ENV = dataclasses.replace(
    V2, system_prompt=PRODUCT_SYSTEM,
    handlers={name: guarded(name, h) for name, h in {**V2.handlers, "send_reply": held_reply,
                                                     "ask_owner": ask_person}.items()})


def prepare(state: dict, ticket_id: str, answers: dict | None = None) -> dict:
    """A private copy of the shop for one request: no scripted customer or owner, the person's earlier answers,
    and the quotes this ticket was already sent (so a follow-up can name them and the next one is numbered on)."""
    s = copy.deepcopy(state)
    s.update(tickets=[ticket_id], drafts={q: d for q, d in s["drafts"].items() if d["ticket_id"] == ticket_id},
             outbox=[], customers={}, talk={}, owner={}, owner_asked=[], answers=copy.deepcopy(answers or {}),
             questions=[], stop=None)
    return s


def rules_used(model, messages: list[dict]) -> list[int] | None:
    """One tool-free call after the run: which confirmed rules the model says it followed. None if unreadable."""
    reply = model(messages + [{"role": "user", "content": RULES_USED}], None)
    text = (reply.get("content") or "").split("</think>")[-1]
    line = next((l for l in text.splitlines() if "rules used" in l.lower()), None)
    if line is None:
        return None
    return sorted({int(n) for n in RULE_ID.findall(line)})


def work(view: dict, state: dict, model, rules_text: str = "", recorder=None, max_turns: int = 24,
         answers: dict | None = None, episode: str | None = None, actor: str = "agent", env=None,
         ask_used: bool = True) -> dict:
    """Work one request. outcome: draft (a reply waits for approval), asked (a question waits for the person),
    stopped (the model finished with neither: shown as stopped, never as done), max_turns, or error.
    The recorder, if any, gets one episode of its own (entrance "agent"), apart from the person's records.
    env replaces the desk's tools (the failure tests break one); ask_used=False skips the "which rules" call,
    for text that is not cards."""
    env = env or PRODUCT_ENV
    desk = Desk(prepare(state, view["ticket_id"], answers), recorder, env=env)
    if recorder:
        recorder.start(episode or "e-" + secrets.token_hex(5), view, task_id=view["ticket_id"], actor=actor,
                       env=PRODUCT_ENV.version, entrance="agent")
    system = PRODUCT_SYSTEM + ("\n\n" + rules_text.strip() if rules_text.strip() else "")

    def until_held(messages, tools):
        return {"content": "Done."} if desk.state["stop"] else model(messages, tools)
    try:
        messages, status = agent.run(desk, view, until_held, max_turns, system, env=env)
        error = None
    except Exception as e:  # the job shows where it stopped
        messages, status, error = [], "error", f"{type(e).__name__}: {e}"
    if recorder:
        recorder.end(status)
    s = desk.state
    outcome = ("error" if error else "asked" if s["stop"] == "asked" else "draft" if s["outbox"]
               else "max_turns" if status == "max_turns" else "stopped")
    used = None
    if ask_used and rules_text.strip() and messages and outcome in ("draft", "asked"):
        try:
            used = rules_used(model, messages)
        except Exception:
            used = None
    return {"outcome": outcome, "error": error, "drafts": list(s["outbox"]), "questions": list(s["questions"]),
            "quotes": {q: d for q, d in s["drafts"].items()}, "used": used, "history": desk.history,
            "state": s, "system": system}


def price_problem(state: dict, ticket_id: str, line: dict, lines: list[dict]) -> str | None:
    """The scorer's price rule (island/score2.price_problem), read from the desk state instead of the grader."""
    sku, p = line["sku"], line["unit_price"]
    listed = list_price(state, sku, lines)
    if approved_price(state, sku, p) or abs(p - listed) <= 0.0051:
        return None
    if p > listed:
        return f"{sku} 단가가 가격표보다 높습니다"
    if p < floor_price(state["catalog"][sku]["unit_cost"]) - 1e-9:
        return f"{sku} 단가가 최저 허용 단가보다 낮습니다"
    if within_authority(listed, p, state["authority"].get(ticket_id, 0.0)) or owner_agreed(state, ticket_id):
        return None
    return f"{sku} 할인이 권한을 넘습니다(사장님 동의 없음)"


def amounts(message: str) -> list[str]:
    text = unicodedata.normalize("NFKC", message)
    return [a or b for a, b in AMOUNT.findall(text)]


def guard(state: dict, ticket_id: str, history: list[dict]) -> list[dict]:
    """Checks a held draft must pass before it can be approved. Each is {"name", "ok", "detail"} in Korean.
    Every amount in a reply is read, attached quote or not, and every quote a reply attaches or names is checked."""
    replies = [r for r in state["outbox"] if r["ticket_id"] == ticket_id]
    prices = customer_prices(c["result"] for c in history)
    costs = {cents(c["result"]["unit_cost"]) for c in history
             if c["name"] == "get_cost" and isinstance(c["result"], dict) and "unit_cost" in c["result"]}
    stated = [m for r in replies for m in amounts(r["message"])]
    stray = sorted({m for m in stated if cents(m) not in prices})
    shown = sorted({m for m in stated if cents(m) in costs - prices})
    loose = sum(1 for r in replies if not r.get("quote_id") and amounts(r["message"]))
    mine = {q for q, d in state["drafts"].items() if d["ticket_id"] == ticket_id}
    named = {q.upper() for r in replies for q in QUOTE_ID.findall(r["message"])}
    unknown = sorted(named - mine - {q.upper() for q in state["quotes"]})
    checks = [
        {"name": "답장 초안", "ok": bool(replies), "detail": f"{len(replies)}건 승인 대기" if replies else "없음"},
        {"name": "가격은 견적으로만", "ok": not loose,
         "detail": "견적 없이 가격을 적은 답장이 있습니다" if loose else "견적 밖 가격 없음"},
        {"name": "원가 비공개", "ok": not shown,
         "detail": "원가가 답장에 있습니다: " + ", ".join(f"${s}" for s in shown) if shown else "원가 없음"},
        {"name": "금액은 데스크 값", "ok": not stray,
         "detail": "데스크가 준 적 없는 금액: " + ", ".join(f"${s}" for s in stray) if stray else "모두 데스크 값"},
        {"name": "견적번호 확인", "ok": not unknown,
         "detail": "이 티켓에 없는 견적번호: " + ", ".join(unknown) if unknown else "모두 있는 번호"},
    ]
    for qid in sorted({r["quote_id"] for r in replies if r.get("quote_id")} | (named & mine)):
        lines = state["drafts"][qid]["lines"]
        problems = [p for l in lines if (p := price_problem(state, ticket_id, l, lines))]
        for sku in sorted({l["sku"] for l in lines}):
            when = stock_problem(state["catalog"][sku], [l for l in lines if l["sku"] == sku])
            if when:
                problems.append(f"{sku}는 {when}까지 재고가 모자랍니다")
        checks.append({"name": f"견적 {qid}", "ok": not problems,
                       "detail": "; ".join(problems) or "가격, 권한, 최저 단가, 출고일별 재고 모두 맞음"})
    return checks


def with_one_rewrite(attempt, rules: str):
    """attempt(rules) -> (job, checks). If the desk's checks hold the draft back, say why and let the model write it
    once more. Returns (job, checks, first): both drafts are kept, and a second draft that fails stays unapprovable."""
    done, checks = attempt(rules)
    failed = [c for c in checks if not c["ok"]]
    if not failed:
        return done, checks, None
    note = ("A draft you wrote for this ticket was held back by the desk's checks. Write it again and "
            "avoid this: " + " ".join(HINTS.get(c["name"], HINTS["quote"]) for c in failed))
    first = {"drafts": done["drafts"], "failed": failed, "job": done}
    done, checks = attempt((rules + "\n\n" + note).strip())
    return done, checks, first
