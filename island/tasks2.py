"""Ticket generator for quote-desk/2.

A ticket is a set of conditions. Each condition has a twin that calls for the opposite handling,
so a way of working shows up as a switch, not as a habit:
  typo_open          an item code that is not in the catalog          twin: typo_confirmed (already confirmed)
  short_open         an item is short and the customer's preference   twin: short_constrained (the customer
                     is unknown (split, wait, or a substitute)              already said: all by the date, any style)
  short_impossible   all of exactly this item by the date, which the stock cannot meet
  reorder_room       an expired reorder; the customer pushes back on   twin: reorder_tight (a 5% cut would go
                     the new price once; 5% off stays above the floor        under the floor)
  reorder_valid      a reorder of a quote that is still valid
  ack_followup       the customer follows up after an acknowledgement (nothing priced yet)
  missing_date       the customer gave no ship date                    twin: a ticket with a date
The parts of a task are kept apart as in quote-desk/1 (island/tasks.py): view (on screen, the only part
that enters records), request (the same visible request in structured form, for scripted workers),
hidden (the desk state, with the customer's conditions and the owner's policy), grader.

train has single conditions and twins; dev adds two combinations; eval holds new combinations of
conditions that training only shows apart, and uses names, SKUs and phrasings training never shows.
"""
from __future__ import annotations

import random
from datetime import date, timedelta

from workalike.desk import unit_price_for
from workalike.desk2 import empty_state, floor_price

from .tasks import (CATALOG, COLLEAGUE_ACKS, EVAL_ONLY_SKUS, POOLS, QTY_RANGE, THREAD_SMALLTALK, _cost, _fmt_date,
                    _round_qty, _typo)

SPLITS = {
    "train": {"standard": 8, "typo_open": 6, "typo_confirmed": 4, "short_open": 9, "short_constrained": 5,
              "short_impossible": 2, "reorder_valid": 5, "reorder_room": 5, "reorder_tight": 4, "ack_followup": 6,
              "missing_date": 6},
    "dev": {"standard": 1, "typo_open": 1, "typo_confirmed": 1, "short_open": 1, "short_constrained": 1,
            "short_impossible": 1, "reorder_room": 1, "reorder_tight": 1, "ack_followup": 1, "missing_date": 1,
            "typo_open+short_open": 1, "ack_followup+short_open": 1},
    "eval": {"standard": 1, "typo_open": 1, "short_open": 2, "reorder_room": 1, "reorder_valid": 1,
             "ack_followup": 1, "missing_date": 1,
             "typo_open+short_open": 2, "ack_followup+short_open": 1, "ack_followup+typo_open": 1,
             "short_constrained+typo_confirmed": 1, "ack_followup+short_constrained": 1,
             "missing_date+typo_open": 1, "missing_date+short_open": 1,
             "typo_confirmed": 1, "short_constrained": 2, "short_impossible": 2, "reorder_tight": 2,
             "ack_followup+missing_date": 1},
    # dev2 (10-28 night): a larger dev set for tuning and the procedure loop, with names and items that neither
    # training nor eval uses; 12 twin pairs (3 of each type) and the combinations eval also tests.
    "dev2": {"typo_open": 3, "typo_confirmed": 3, "short_open": 5, "short_constrained": 3, "reorder_room": 3,
             "reorder_tight": 3, "missing_date": 3, "standard": 4, "ack_followup": 2, "short_impossible": 2,
             "reorder_valid": 2, "typo_open+short_open": 3, "ack_followup+short_open": 2, "ack_followup+typo_open": 2,
             "missing_date+short_open": 2, "short_constrained+typo_confirmed": 2},
}
# train150 (10-28 night, a reviewer's experiment 0): a balanced training set where each of the five rules worker C
# follows comes up about 25 times (ask before quoting an unknown code, ask for a missing date, offer choices on
# short stock, the reorder push-back, two quotes when the customer already set the terms), with twins.
SPLITS["train150"] = {"typo_open": 25, "typo_confirmed": 10, "missing_date": 25, "standard": 10, "short_open": 25,
                      "short_constrained": 25, "reorder_room": 13, "reorder_tight": 12, "short_impossible": 5}
# Items only dev2 asks for (sku, name, category, base cost); its tickets request nothing else.
DEV2_CATALOG = [
    ("MUG-12C", "12 oz enamel camp mug", "mug", 3.10), ("MUG-14L", "14 oz latte mug", "mug", 3.30),
    ("TMB-24V", "24 oz vacuum tumbler", "tumbler", 7.90), ("TMB-12K", "12 oz kids tumbler", "tumbler", 2.40),
    ("TSH-LSV", "long-sleeve t-shirt", "tee", 6.40), ("TSH-VNK", "v-neck t-shirt", "tee", 4.10),
    ("CAP-BKT", "bucket hat", "cap", 4.20), ("CAP-BEN", "knit beanie", "cap", 3.60),
    ("TOT-DRW", "drawstring bag", "tote", 1.40), ("TOT-ZIP", "zippered tote bag", "tote", 3.90),
    ("PST-1117", "11x17 poster", "poster", 1.70), ("PST-2740", "27x40 poster", "poster", 5.80),
    ("STK-2CL", "2 in clear sticker", "sticker", 0.21), ("STK-4OV", "4 in oval sticker", "sticker", 0.29),
    ("NTB-A6P", "A6 pocket notebook", "notebook", 2.10), ("NTB-B5S", "B5 spiral notebook", "notebook", 3.40),
    ("PEN-GEL", "gel pen", "pen", 0.55), ("PEN-BAM", "bamboo pen", "pen", 0.95),
    ("MAG-CAL", "calendar magnet", "magnet", 0.45), ("MAG-OVL", "oval car magnet", "magnet", 6.90),
]
DEV2_POOL = {**POOLS["eval"],
             "first": ["Annika", "Bruno", "Celeste", "Dmitri", "Esme", "Farid", "Greta", "Hamish", "Ines", "Joaquin"],
             "last": ["Okonkwo", "Varga", "Lindgren", "Sato", "Ferreira", "Halloran", "Nakamura", "Petrov"],
             "place": ["Juniper Row", "Tidewater", "Old Mill"],
             "kind": ["Tea Room", "Bike Shop", "Dance School", "Print Club"]}
# test24 (the main test, 10-29 plan, built 11-02): names no other set uses; the same items as dev2, which the cards
# never read. Its tickets and their answer sheet come from island/test24.py, not from SPLITS.
TEST24_POOL = {**POOLS["eval"],
               "first": ["Saoirse", "Tomasz", "Wanjiru", "Anselm", "Beatriz", "Cormac", "Ayumi", "Emeka", "Linnea",
                         "Rohan", "Margit", "Ignacio"],
               "last": ["Adeyemi", "Bergstrom", "Ivanova", "Mbeki", "Tanaka", "Quintero", "Rasmussen", "Oyelaran"],
               "place": ["Fernhill", "Bramble Lane", "Saltmarsh"],
               "kind": ["Choir", "Ceramics Studio", "Rowing Club", "Seed Library"]}
# Twins are made from the same customer, items and dates, so the only difference is the condition.
TWINS = {"typo_open": "typo_confirmed", "short_open": "short_constrained", "reorder_room": "reorder_tight",
         "missing_date": "standard"}
TWIN_PAIRS = {"train": 4, "dev": 3, "eval": 2, "dev2": 4, "train150": 4}
PAIRS_PER_TYPE = {"dev2": 3, "train150": 5}  # twin pairs of each type; one where a split is not listed
TICKET_RANGES = {"train": (1000, 49999), "dev": (100000, 199999), "eval": (50000, 99999), "dev2": (200000, 299999),
                 "train150": (300000, 399999), "demo": (400000, 449999),  # demo: the product loop's cases
                 "test24": (450000, 489999), "test24dev": (490000, 499999)}
# The sets made with seed 11. They do not change once a person has worked them. One change was made before
# any person or any model saw them beyond a first base run on dev (10-28): the owner's fixed answer now
# also leaves anything else to the clerk (first hashes 7f684edf, ff16727f, d9e986fd).
FIXED_SHA256 = {"train": "98647435c3ce92e818c6ae35de92cc6a65414c4791dec439e51eb09e5a499e0d",
                "dev": "5b7be60f3ef46be80dc866a2cd20e3b746aa9af678a58cd6ba506900a134f872",
                "eval": "253e8b843c603db05c0c8c99903f5e2ed351631fc4ada81c42dc06c5aa686e79",
                "dev2": "e519a5156476a26468b77e679856d9173f02edd4db75e27321945bcd0c0e395f",
                "train150": "2527f3de715824500015b075967eb3f8e2031c9c7ee53ef98b9560211af5a636"}
SHORT = {"short_open", "short_constrained", "short_impossible"}
REORDER = {"reorder_valid", "reorder_room", "reorder_tight"}

AUTHORITY = [0.0, 0.03, 0.05]
# The owner's fixed answer covers any question: a discount policy, and the rest left to the clerk.
OWNER = {"floor_ok": ["Fine, you can go down to the floor price on this one if it helps. Anything else is your call.",
                      "Okay to give a little off, just stay at or above the floor price. Otherwise use your judgement."],
         "hold": ["Please keep the list price on this one. Anything else is your call.",
                  "No discount on this order, sorry. Otherwise use your judgement."]}

SAYS = {
    "typo": ["Yes, I meant {sku}, the {name}.", "Sorry, typo on my end: it's {sku} ({name})."],
    "date": ["We'd need them shipped by {date}.", "Shipping by {date} would work for us."],
    "split": ["Please send the {on_hand} you have now and the rest after the restock.",
              "Ship what you have now and the rest when it arrives, that's fine."],
    "later": ["We can wait for the restock and take them all then.",
              "No rush, we'd rather wait and get all {qty} of the {name}."],
    "substitute": ["The {sub_name} ({sub}) is fine instead.", "Happy to take the {sub_name} ({sub}) instead."],
    "constrained": ["As I said, we need all {qty} by {date}; another size or style is fine."],
    "impossible": ["As I said, it has to be all {qty} of the {name} by {date}."],
    "plain": ["Thanks, looking forward to the quote.", "Great, thanks. I'll watch for the quote."],
    "push_back": ["Hmm, that's quite a bit more than last time ({qref}). Is there anything you can do on the price?",
                  "That's a jump from {qref}. Any chance you can do a bit better on the price?"],
}
CONSTRAINT_TEXT = {
    "short_constrained": ["We need all {qty} by {date}, it's for an event. A different size or style is fine.",
                          "All {qty} have to be here by {date}; the exact style doesn't matter."],
    "short_impossible": ["We need all {qty} of exactly this item by {date}, nothing else will do.",
                         "It has to be the {name}, all {qty} of them, by {date}."],
}
CONFIRM_ASK = "Thanks, {first}! {typo} isn't in our catalog. Did you mean {sku}, the {name}?"
CONFIRM_YES = ["Yes, {sku} is right, {qty} of them. When can we get the quote?",
               "That's the one, {sku}. Could you send the quote?"]


def reorder_room(unit_cost: float, qty: int) -> bool:
    """Does 5% off the list price stay at or above the floor?"""
    return round(unit_price_for(unit_cost, qty) * 0.95, 2) >= floor_price(unit_cost)


def long_date(d: date) -> str:
    return f"{d:%B} {d.day}"


class Builder:
    def __init__(self, seed: int, split: str):
        self.split = split
        self.rng = random.Random(f"v2:{seed}:{split}")
        self.training = split.startswith("train")
        self.pool = {"dev2": DEV2_POOL, "test24": TEST24_POOL, "test24dev": TEST24_POOL}.get(split) or \
            POOLS["train" if self.training else "eval"]
        self.skus = [c for c in CATALOG if not self.training or c[0] not in EVAL_ONLY_SKUS]
        self.request = {c[0] for c in self.skus}  # items a ticket may ask for
        if split in ("dev2", "test24", "test24dev"):
            self.skus = self.skus + DEV2_CATALOG
            self.request = {c[0] for c in DEV2_CATALOG}
        self.used: set[str] = set()

    def ticket_id(self) -> str:
        while True:
            t = f"T-{self.rng.randint(*TICKET_RANGES[self.split])}"
            if t not in self.used:
                self.used.add(t)
                return t

    def catalog(self) -> dict:
        rng, by_cat, out = self.rng, {}, {}
        drift = 1.0 if self.training else 1.12
        for sku, _, cat, _ in self.skus:
            by_cat.setdefault(cat, []).append(sku)
        for sku, name, cat, base in self.skus:
            hi = QTY_RANGE[cat][1]
            out[sku] = {"name": name, "unit_cost": _cost(base, rng, drift), "on_hand": rng.randint(hi * 2, hi * 4),
                        "restock_date": None, "restock_qty": 0, "substitutes": [s for s in by_cat[cat] if s != sku]}
        return out

    def items(self, flags: set, catalog: dict) -> list[tuple[str, int]]:
        rng = self.rng
        cats = sorted({c[2] for c in self.skus if c[0] in self.request})
        if flags & REORDER:  # one line; reorder_tight sits in the 500+ break, where 5% off is under the floor
            big = "reorder_tight" in flags
            while True:
                sku, _, cat, cost = rng.choice([c for c in self.skus if c[0] in self.request and c[3] >= 1 and (
                    QTY_RANGE[c[2]][1] >= 500 if big else QTY_RANGE[c[2]][0] < 450)])
                lo, hi = QTY_RANGE[cat]
                qty = min(_round_qty(rng.randint(500, hi)), hi) if big else \
                    min(_round_qty(rng.randint(lo, min(hi, 449))), hi, 449)
                if big == (qty >= 500) and (reorder_room(catalog[sku]["unit_cost"], qty) != big):
                    return [(sku, qty)]
        n = 1 if flags & (SHORT | {"typo_open", "typo_confirmed"}) and rng.random() < 0.5 else rng.choice([1, 1, 2])
        out = []
        for cat in rng.sample(cats, n):
            options = [c[0] for c in self.skus if c[2] == cat and c[0] in self.request]
            if flags & SHORT and not out:
                options = [s for s in options if catalog[s]["substitutes"]]
                if not options:
                    continue
            lo, hi = QTY_RANGE[cat]
            out.append((rng.choice(options), max(_round_qty(rng.randint(lo, hi)), 10 if flags & SHORT else 1)))
        return out or self.items(flags, catalog)

    def base(self) -> dict:
        """The parts a ticket and its twin share."""
        rng = self.rng
        today = (date(2026, 9, 1) if self.training else date(2026, 11, 2)) + timedelta(days=rng.randint(0, 50))
        first, last = rng.choice(self.pool["first"]), rng.choice(self.pool["last"])
        place = f"{last} {rng.choice(self.pool['kind'])}" if rng.random() < 0.5 else \
            f"{rng.choice(self.pool['place'])} {rng.choice(self.pool['kind'])}"
        return {"today": today, "first": first, "sender": f"{first} {last}, {place}",
                "need_by": today + timedelta(days=rng.randint(9, 28)), "catalog": self.catalog(),
                "authority": rng.choice(AUTHORITY), "owner": rng.choice(sorted(OWNER)),
                "smalltalk": rng.random() < 0.3, "seed": rng.randrange(1 << 30)}

    def task(self, index: int, family: str, base: dict | None = None, pair: str | None = None) -> dict:
        flags = set() if family == "standard" else set(family.split("+"))
        base = base or self.base()
        rng = random.Random(base["seed"])  # a twin replays the same draws up to its condition
        # Wording, names of things and quote ids come from t. Given a text_seed, a twin keeps every number of its
        # pair and reads differently, so a person does not meet the same email twice (the human sheet, 10-29).
        t = random.Random(base["text_seed"]) if "text_seed" in base else rng
        says_text, constraint_text = getattr(self, "says", SAYS), getattr(self, "constraint_text", CONSTRAINT_TEXT)
        today, first, need_by = base["today"], base["first"], base["need_by"]
        state = empty_state(today.isoformat())
        state["catalog"] = cat = {k: dict(v, substitutes=list(v["substitutes"])) for k, v in base["catalog"].items()}
        ticket = self.ticket_id()
        state["tickets"].append(ticket)
        state["authority"][ticket] = base["authority"]
        state["owner"][ticket] = {"policy": base["owner"], "says": t.choice(OWNER[base["owner"]])}

        saved = self.rng
        self.rng = rng
        items = self.items(flags, cat)
        self.rng = saved
        written = {s: s for s, _ in items}
        thread = []
        if base["smalltalk"]:
            q, a = t.choice(THREAD_SMALLTALK)
            d = (today - timedelta(days=t.randint(3, 20))).isoformat()
            thread = [{"date": d, "who": "customer", "text": q}, {"date": d, "who": "shop", "text": a}]
        request = {"first_name": first, "items": None, "quote_ref": None, "need_by": need_by.isoformat(),
                   "constraint": None, "acked_on": None, "confirmed": False}
        customer = {"wants": [{"sku": s, "qty": q, "substitutes": []} for s, q in items],
                    "need_by": need_by.isoformat(), "split_ok": True, "later_ok": False, "substitute_ok": False}
        sentences, fill = [], {"first": first, "date": long_date(need_by)}
        grader = {"family": family, "flags": sorted(flags), "pair": pair, "today": today.isoformat(),
                  "need_by": need_by.isoformat(), "feasible": True, "short": None, "approved": None,
                  "authority": base["authority"], "owner": base["owner"]}

        if flags & SHORT:
            sku, qty = items[0]
            item = cat[sku]
            item["on_hand"] = rng.randint(max(1, qty // 5), max(1, qty * 2 // 3))
            item["restock_date"] = (need_by + timedelta(days=rng.randint(3, 15))).isoformat()
            item["restock_qty"] = qty * 2
            subs = [s for s in item["substitutes"] if cat[s]["on_hand"] >= qty]
            customer["wants"][0]["substitutes"] = subs
            fill.update(sku=sku, name=item["name"], qty=qty, on_hand=item["on_hand"])
            grader["short"] = {"sku": sku, "on_hand": item["on_hand"], "restock_date": item["restock_date"],
                               "substitutes": subs}
            if "short_open" in flags:
                pref = rng.choice(["split", "later", "substitute"])
                pref = base.get("pref", pref)
                sub = subs[0]  # the first substitute check_stock lists, the one a worker would offer
                fill.update(sub=sub, sub_name=cat[sub]["name"])
                customer["wants"][0]["substitutes"] = [sub]
                customer.update(split_ok=pref in ("split", "later"), later_ok=pref == "later",
                                substitute_ok=pref == "substitute")
                sentences.append(t.choice(says_text[pref]).format(**fill))
                grader["short"]["preference"] = pref
            elif "short_constrained" in flags:
                customer.update(split_ok=False, later_ok=False, substitute_ok=True)
                request["constraint"] = "any_style_all_by_date"
                sentences.append(t.choice(says_text["constrained"]).format(**fill))
            else:
                customer.update(split_ok=False, later_ok=False, substitute_ok=False)
                request["constraint"] = "exact_all_by_date"
                grader["feasible"] = False
                sentences.append(t.choice(says_text["impossible"]).format(**fill))
        if flags & {"typo_open", "typo_confirmed"}:
            sku = items[0][0]
            written[sku] = _typo(sku, state, t)
            fill.update(sku=sku, name=cat[sku]["name"], typo=written[sku], qty=items[0][1])
            if "typo_open" in flags:
                sentences.insert(0, t.choice(says_text["typo"]).format(**fill))
        if flags & REORDER:
            sku, qty = items[0]
            qref = f"Q-{t.randint(100, 9999)}"
            if "reorder_valid" in flags:
                price = unit_price_for(cat[sku]["unit_cost"] * rng.uniform(0.95, 1.03), qty)
                price = max(price, floor_price(cat[sku]["unit_cost"]))
                valid_until = today + timedelta(days=rng.randint(3, 40))
                grader["approved"] = {sku: price}
            else:
                price = round(unit_price_for(cat[sku]["unit_cost"], qty) / rng.uniform(1.12, 1.25), 2)
                valid_until = today - timedelta(days=rng.randint(10, 90))
                customer["push_back_over"] = round(price * qty, 2)
            state["quotes"][qref] = {"valid_until": valid_until.isoformat(),
                                     "lines": [{"sku": sku, "name": cat[sku]["name"], "qty": qty, "unit_price": price}]}
            request["quote_ref"] = grader["quote_ref"] = qref
            fill["qref"] = qref

        item_lines = None if flags & REORDER else [
            t.choice(self.pool["item"]).format(qty=q, name=cat[s]["name"], sku=written[s]) for s, q in items]
        stated = next(iter(flags & set(CONSTRAINT_TEXT)), None)
        body = self.body(t, first, item_lines, None if "missing_date" in flags else need_by,
                         t.choice(self.pool["reorder"]).format(qref=fill["qref"]) if flags & REORDER else None,
                         t.choice(constraint_text[stated]).format(**fill) if stated else None)
        sent_so_far = None
        if "missing_date" in flags:
            request["need_by"] = None
            sentences.append(t.choice(says_text["date"]).format(**fill))
        if "typo_confirmed" in flags:
            asked = (today - timedelta(days=t.randint(1, 3))).isoformat()
            thread += [{"date": asked, "who": "customer", "text": body},
                       {"date": asked, "who": "shop", "text": CONFIRM_ASK.format(**fill)}]
            body = t.choice(CONFIRM_YES).format(**fill)
            if request["constraint"]:
                body += " " + t.choice(constraint_text["short_constrained"]).format(**fill)
            request["confirmed"] = True
            sent_so_far = f"a question about the item code on {asked}; no quote yet"
        if "ack_followup" in flags:
            sent = (today - timedelta(days=t.randint(1, 4))).isoformat()
            thread += [{"date": sent, "who": "customer", "text": body},
                       {"date": sent, "who": "shop", "text": t.choice(COLLEAGUE_ACKS).format(first=first)}]
            body = t.choice(self.pool["followup"])
            request["acked_on"] = sent
            sent_so_far = f"an acknowledgement on {sent}; no quote yet"
        if not flags & REORDER:
            request["items"] = [{"sku": s if request["confirmed"] else written[s], "qty": q} for s, q in items]

        says = {"answer": " ".join(sentences) or t.choice(says_text["plain"]),
                "decline": "That won't work for us. " + (" ".join(sentences) or "It isn't what we asked for.")}
        if flags & {"reorder_room", "reorder_tight"}:
            says["push_back"] = t.choice(says_text["push_back"]).format(**fill)
        customer["says"] = says
        state["customers"][ticket] = customer
        grader["wants"] = customer["wants"]

        view = {"today": today.isoformat(), "ticket_id": ticket, "from": base["sender"], "body": body,
                "authority": base["authority"], "sent_so_far": sent_so_far}
        if thread:
            view["thread"] = thread
        return {"task_id": f"{self.split}-{index:04d}", "family": family, "split": self.split, "env": "quote-desk/2",
                "view": view, "request": request, "hidden": {"state": state}, "grader": grader}

    def body(self, rng, first, item_lines, need_by, reorder, constraint) -> str:
        p = self.pool
        parts = [rng.choice(p["greeting"])] if rng.random() < 0.8 else []
        parts.append(reorder or rng.choice(p["intro"]).format(event=rng.choice(p["event"])))
        if item_lines:
            parts.append("\n".join(f"- {l}" for l in item_lines) if rng.random() < 0.5 else ", ".join(item_lines))
        if need_by:
            parts.append(rng.choice(p["deadline"]).format(date=_fmt_date(need_by, rng)))
        if constraint:
            parts.append(constraint)
        if rng.random() < 0.9:
            parts.append(f"{rng.choice(p['closing'])} {first}")
        return "\n".join(parts)


def make_tasks(seed: int, split: str) -> list[dict]:
    b = Builder(seed, split)
    families = [f for f, n in SPLITS[split].items() for _ in range(n)]
    pairs = []
    for open_, twin in list(TWINS.items())[: TWIN_PAIRS[split]]:
        for _ in range(PAIRS_PER_TYPE.get(split, 1)):
            if open_ in families and twin in families:
                families.remove(open_)
                families.remove(twin)
                pairs.append((open_, twin))
    b.rng.shuffle(families)
    tasks = []
    for f in families:
        tasks.append(b.task(len(tasks), f))
    for open_, twin in pairs:
        base = b.base()
        pid = f"{open_}:{len(tasks)}"
        tasks.append(b.task(len(tasks), open_, base, pid))
        tasks.append(b.task(len(tasks), twin, base, pid))
    b.rng.shuffle(tasks)  # twins are not worked back to back
    for i, t in enumerate(tasks):
        t["task_id"] = f"{split}-{i:04d}"
    return tasks


def main() -> None:
    import argparse
    import hashlib
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser(description="Write the quote-desk/2 ticket sets and their hashes.")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="data/v2/tasks")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for split in SPLITS:
        path = out / f"{split}.jsonl"
        path.write_text("".join(json.dumps(t) + "\n" for t in make_tasks(args.seed, split)))
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{path}: {sum(SPLITS[split].values())} tickets, sha256 {hashes[split][:16]}")
    (out / "hashes.json").write_text(json.dumps({"seed": args.seed, "sha256": hashes}, indent=1) + "\n")


if __name__ == "__main__":
    main()
