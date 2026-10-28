"""Scripted worker C: the user's own rules for quote-desk/2, for checking the path and how much data it takes.

C is not a person and its records never count as human results. Its rules come from the user's
judgement notes and the contrast probes (experiments/probe-human.md in the planning folder):
  - An item code that is not in the catalog: ask the customer to confirm before quoting.
    Once the customer has confirmed it, quote straight away.
  - No ship date: ask when they need it, and say how soon we could ship.
  - Short stock and the customer's preference unknown: ask first, offering to send what we have now
    and the rest after the restock, to wait and ship it all, or a substitute. The substitute comes with
    a small apology discount (3%) when that stays at or above the floor; without the authority for it,
    ask the owner before offering it.
  - Short stock and the customer already said "all by the date, any style": no question and no discount;
    quote what we have plus a substitute for the rest on the date, and name an all-substitute quote too.
  - Short stock and "exactly this item, all by the date", which cannot be met: apologise and close.
  - A reorder of a valid quote: the approved prices, no cost check. An expired one: today's prices, said
    plainly. If the customer pushes back, offer 5% off when that stays above the floor (asking the owner
    when it is beyond our authority); when 5% would go under the floor, ask the owner, and give a little
    (2%) only if the owner agrees.
  - A follow-up after our acknowledgement: no second acknowledgement; answer with the quote and mention
    the earlier note.
  - Otherwise: quote at list price in one reply.
C sees only the ticket and what its own tool calls return, never the hidden state.
"""
from __future__ import annotations

import math
import random
import re
from datetime import date, timedelta

from workalike.desk import Desk

FINAL = "Done."
SKU = re.compile(r"\b[A-Z]{3}-[A-Z0-9]{2,4}\b")
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
LONG_DATE = re.compile(r"\b(" + "|".join(MONTHS) + r") (\d{1,2})\b")

OPENERS = ["Hi {first},", "Hello {first},", "Thanks {first},"]
SORRY_WAIT = ["Sorry for the wait since our note on {acked}.", "Thanks for your patience since our note on {acked}."]
ASK_TYPO = ["{written} isn't in our catalog. Did you mean {guess}, the {name}?",
            "I couldn't find {written} in our catalog; is it {guess} ({name}) you're after?"]
ASK_DATE = ["When do you need them shipped? The earliest we could ship is {earliest}.",
            "What ship date do you need? We could ship as early as {earliest}."]
ASK_SHORT = ["We only have {on_hand} of the {name} right now; the restock arrives on {restock}. "
             "We could send the {on_hand} now and the rest after the restock, or wait and ship all {qty} together.",
             "The {name} is short: {on_hand} on hand, with a restock on {restock}. Would you like the {on_hand} now and "
             "the rest after the restock, or everything together once it arrives?"]
ASK_SUB = [" We also have the {sub_name} ({sub}) in stock for all {qty}{off}.",
           " Or we could use the {sub_name} ({sub}) instead, all in stock{off}."]
OFF = [", and since we're short, we'd take {pct}% off it", " at {pct}% off, as an apology for the shortage"]
ASK_END = ["Let me know what works for you.", "Which would you prefer?"]
QUOTE = ["Quote {qid} is attached: {lines}. Total ${total:,.2f}.", "Here is quote {qid}: {lines}, ${total:,.2f} in total."]
APPROVED = "These are your approved prices from {qref}."
EXPIRED = "{qref} has expired, so this quote is at today's prices."
MIX_ALT = " If you'd rather have one style, {alt} is all {qty} as the {sub_name}."
CLOSERS = ["Let me know if you'd like to go ahead.", "Reply to confirm and we'll get started."]
SORRY_CLOSE = ("I'm sorry, {first}. We have only {on_hand} of the {name} now and the restock arrives on {restock}, "
               "so we can't get all {qty} to you by {date}. I'm sorry we couldn't help this time; "
               "we'll keep more on hand next time.")
DISCOUNT = ["Since this is a reorder, I can do {pct}% off: quote {qid} is {lines}, ${total:,.2f} in total.",
            "I checked what we can do: {pct}% off as a reorder. Quote {qid}: {lines}, ${total:,.2f} in total."]
HOLD = ["I checked, and I'm afraid {qid} is the best price we can offer this time.",
        "I asked, and we can't go lower than {qid} this time, sorry."]
OWNER_Q = ["May I take {pct}% off for this customer on {ticket}? {why}",
           "Is it all right to give {pct}% off on {ticket}? {why}"]


class Stuck(RuntimeError):
    pass


def _ok(result: dict) -> dict:
    if "error" in result:
        raise Stuck(f"worker C hit an unexpected error: {result}")
    return result


def cents_up(x: float) -> float:
    return math.ceil(round(x * 100, 6)) / 100


def parse_date(text: str, today: str) -> str | None:
    m = LONG_DATE.search(text)
    if not m:
        return None
    t = date.fromisoformat(today)
    d = date(t.year, MONTHS.index(m.group(1)) + 1, int(m.group(2)))
    return (d if d >= t else date(t.year + 1, d.month, d.day)).isoformat()


def parse_answer(text: str, today: str) -> dict:
    out = {"sku": None, "pref": None, "sub": None, "date": parse_date(text, today)}
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        skus = SKU.findall(sentence)
        if "instead" in sentence:
            out["pref"], out["sub"] = "substitute", skus[0] if skus else None
        elif "rest after" in sentence or "rest when" in sentence:
            out["pref"] = "split"
        elif "wait" in sentence:
            out["pref"] = "later"
        elif skus and out["sku"] is None:
            out["sku"] = skus[0]
    return out


class C:
    def __init__(self, desk: Desk, task: dict, rng: random.Random):
        self.desk, self.rng = desk, rng
        self.view, self.req = task["view"], task["request"]
        self.ticket, self.first, self.today = self.view["ticket_id"], self.req["first_name"], self.view["today"]
        self.authority = self.view.get("authority", 0.0)
        self.names: dict[str, str] = {}
        self.prices: dict[str, dict] = {}
        self.said = 0

    def say(self, *parts: str) -> str:
        head = [self.rng.choice(OPENERS).format(first=self.first)]
        if self.req["acked_on"] and not self.said:
            head.append(self.rng.choice(SORRY_WAIT).format(acked=self.req["acked_on"]))
        self.said += 1
        return " ".join(head + [p for p in parts if p])

    def reply(self, message: str, quote_id: str | None = None) -> dict:
        args = {"ticket_id": self.ticket, "message": message}
        if quote_id:
            args["quote_id"] = quote_id
        return _ok(self.desk.call("send_reply", args))

    def stock(self, sku: str, qty: int) -> dict:
        r = self.desk.call("check_stock", {"sku": sku, "qty": qty})
        if "error" not in r:
            self.names[sku] = r["name"]
        return r

    def price(self, sku: str, qty: int) -> dict:
        key = f"{sku}:{qty}"
        if key not in self.prices:
            cost = _ok(self.desk.call("get_cost", {"sku": sku}))
            self.names[sku] = cost["name"]
            self.prices[key] = _ok(self.desk.call("calc_price", {"sku": sku, "qty": qty, "unit_cost": cost["unit_cost"]}))
        return self.prices[key]

    def owner_allows(self, pct: int, why: str) -> bool:
        q = self.rng.choice(OWNER_Q).format(pct=pct, ticket=self.ticket, why=why)
        return "floor" in _ok(self.desk.call("ask_owner", {"ticket_id": self.ticket, "question": q}))["owner_reply"]

    def ship_date(self, need_by: str) -> str:
        d = date.fromisoformat(need_by) - timedelta(days=self.rng.choice([0, 1, 2]))
        return max(d, date.fromisoformat(self.today) + timedelta(days=1)).isoformat()

    def quote(self, lines: list[dict]) -> dict:
        return _ok(self.desk.call("create_quote", {"ticket_id": self.ticket, "lines": lines}))

    def summary(self, q: dict) -> str:
        return "; ".join(f"{l['qty']} x {self.names[l['sku']]} at ${l['unit_price']:.2f}, ships by {l['ship_by']}"
                         for l in q["lines"])

    def work(self) -> str:
        req, rng = self.req, self.rng
        notes, approved, qref = [], None, req["quote_ref"]
        if qref:
            old = _ok(self.desk.call("lookup_quote", {"quote_id": qref}))
            items = [{"sku": l["sku"], "qty": l["qty"]} for l in old["lines"]]
            self.names.update({l["sku"]: l["name"] for l in old["lines"]})
            if old["status"] == "approved":
                approved = {l["sku"]: l["unit_price"] for l in old["lines"]}
                notes.append(APPROVED.format(qref=qref))
            else:
                notes.append(EXPIRED.format(qref=qref))
        else:
            items = [dict(it) for it in req["items"]]

        questions, info, typo = [], {}, None
        for i, it in enumerate(items):
            r = self.stock(it["sku"], it["qty"])
            if r.get("error") == "unknown_sku" and r["similar"]:
                guess = r["similar"][0]
                questions.append(rng.choice(ASK_TYPO).format(written=it["sku"], guess=guess["sku"], name=guess["name"]))
                it["sku"], typo = guess["sku"], i
                r = self.stock(it["sku"], it["qty"])
            info[it["sku"]] = _ok(r)

        need_by = req["need_by"]
        short = [it for it in items if not info[it["sku"]]["enough"]]
        if need_by is None:
            earliest = (date.fromisoformat(self.today) + timedelta(days=1)).isoformat()
            questions.append(rng.choice(ASK_DATE).format(earliest=earliest))

        sub_off = None
        if short:
            it, st = short[0], info[short[0]["sku"]]
            fill = {"name": st["name"], "on_hand": st["on_hand"], "restock": st["restock_date"], "qty": it["qty"],
                    "first": self.first, "date": need_by}
            if req["constraint"] == "exact_all_by_date":
                self.reply(SORRY_CLOSE.format(**fill))
                return FINAL
            if req["constraint"] is None:
                sub = st["substitutes"][0]
                p = self.price(sub["sku"], it["qty"])
                target = cents_up(p["unit_price"] * 0.97)
                if target >= p["floor_unit_price"] and target < p["unit_price"]:
                    off = (p["unit_price"] - target) / p["unit_price"]
                    if off <= self.authority + 1e-9 or self.owner_allows(3, "We're short on what they asked for."):
                        sub_off = target
                questions.append(rng.choice(ASK_SHORT).format(**fill) + rng.choice(ASK_SUB).format(
                    sub_name=sub["name"], sub=sub["sku"], qty=it["qty"],
                    off=rng.choice(OFF).format(pct=3) if sub_off else ""))

        pref, sub_pick = None, None
        if questions:
            answer = self.reply(self.say(*questions, rng.choice(ASK_END)))["customer_reply"]
            got = parse_answer(answer, self.today)
            if typo is not None and got["sku"] and got["sku"] != items[typo]["sku"]:
                items[typo]["sku"] = got["sku"]
                info[got["sku"]] = _ok(self.stock(got["sku"], items[typo]["qty"]))
            need_by = got["date"] or need_by
            pref, sub_pick = got["pref"], got["sub"]
            short = [it for it in items if not info[it["sku"]]["enough"]]

        ship = self.ship_date(need_by)
        lines, alt, mixed = [], None, None
        for it in items:
            sku, qty = it["sku"], it["qty"]
            st = info[sku]
            if not st["enough"] and req["constraint"] == "any_style_all_by_date":
                sub = st["substitutes"][0]
                mixed = (sub, qty, len(lines))
                lines.append(self.line(sku, st["on_hand"], ship))
                lines.append(self.line(sub["sku"], qty - st["on_hand"], ship))
                continue
            if not st["enough"] and pref == "substitute":
                sub = sub_pick or st["substitutes"][0]["sku"]
                line = self.line(sub, qty, ship)
                if sub_off and sub == st["substitutes"][0]["sku"]:
                    line["unit_price"] = sub_off
                lines.append(line)
                continue
            if not st["enough"] and pref == "split":
                later = self.ship_after(st["restock_date"])
                lines.append(self.line(sku, qty, ship, part=st["on_hand"]))
                lines.append(self.line(sku, qty, later, part=qty - st["on_hand"]))
                continue
            if not st["enough"]:  # wait for the restock
                lines.append(self.line(sku, qty, self.ship_after(st["restock_date"])))
                continue
            lines.append(self.line(sku, qty, ship, approved=approved))

        q = self.quote(lines)
        extra = []
        if mixed:
            sub, qty, at = mixed
            a = self.quote(lines[:at] + [self.line(sub["sku"], qty, ship)] + lines[at + 2:])
            extra.append(MIX_ALT.format(alt=a["quote_id"], qty=qty, sub_name=sub["name"]))
        body = rng.choice(QUOTE).format(qid=q["quote_id"], lines=self.summary(q), total=q["total"])
        answer = self.reply(self.say(body, *notes, *extra, rng.choice(CLOSERS)), q["quote_id"])
        if answer.get("customer_accepted"):
            return FINAL
        if answer["customer_reply"].startswith("That won't work"):
            raise Stuck(f"customer turned the quote down: {answer['customer_reply']}")
        return self.push_back(q)

    def ship_after(self, restock: str) -> str:
        return (date.fromisoformat(restock) + timedelta(days=self.rng.choice([0, 1]))).isoformat()

    def line(self, sku, qty, ship, part=None, approved=None) -> dict:
        """A quote line of part (default qty) units, priced at qty: the item's total in the quote."""
        if approved and sku in approved:
            price = approved[sku]
        else:
            price = self.price(sku, qty)["unit_price"]
        return {"sku": sku, "qty": part or qty, "unit_price": price, "ship_by": ship}

    def push_back(self, q: dict) -> str:
        """The customer asked for a better price on a reorder at today's prices."""
        line = q["lines"][0]
        p = self.price(line["sku"], line["qty"])
        room = cents_up(p["unit_price"] * 0.95)
        if room >= p["floor_unit_price"]:
            pct, target = 5, room
            ok = (p["unit_price"] - target) / p["unit_price"] <= self.authority + 1e-9 or \
                self.owner_allows(5, "It's a reorder and they pushed back on the new price.")
        else:
            pct, target = 2, max(cents_up(p["unit_price"] * 0.98), p["floor_unit_price"])
            ok = self.owner_allows(2, "It's a reorder, and 5% would go under the floor.")
        if not ok:
            answer = self.reply(self.say(self.rng.choice(HOLD).format(qid=q["quote_id"])), q["quote_id"])
        else:
            new = self.quote([{"sku": line["sku"], "qty": line["qty"], "unit_price": target,
                               "ship_by": line["ship_by"]}])
            body = self.rng.choice(DISCOUNT).format(pct=pct, qid=new["quote_id"], lines=self.summary(new),
                                                    total=new["total"])
            answer = self.reply(self.say(body, self.rng.choice(CLOSERS)), new["quote_id"])
        if not answer.get("customer_accepted"):
            raise Stuck(f"customer did not accept after the price talk: {answer['customer_reply']}")
        return FINAL


def work(desk: Desk, task: dict, rng: random.Random) -> str:
    return C(desk, task, rng).work()
