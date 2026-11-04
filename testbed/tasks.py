"""Ticket generator for the quote desk test bed.

A task keeps four parts apart:
  view     what the worker sees on screen; the only part that enters records
  request  the same visible request in structured form, for scripted workers
  hidden   the desk state, which a worker can only learn through tools
  grader   answers that only the scorer reads

Training and evaluation draw from different seeds, and evaluation also uses
SKUs, customer names and message phrasings that never appear in training.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

from workalike.desk import empty_state, get_cost, meets_floor, unit_price_for

H1_FAMILIES = {"standard": 45, "stock_short": 15, "unknown_sku": 10}
TRAP_FAMILIES = {"approved_valid": 1, "ack_already_sent": 1, "approved_stale": 1}
TRAIN_FAMILIES = {**H1_FAMILIES, "approved_valid": 10, "ack_already_sent": 10, "approved_stale": 10}
# Two dev-only families combine conditions that training only shows apart: an item runs short after
# an acknowledgement already went out, and a mistyped SKU is also short. dev also doubles the families
# where a habit has to be held back.
DEV_FAMILIES = {
    "standard": 1,
    "stock_short": 1,
    "unknown_sku": 1,
    "approved_stale": 1,
    "approved_valid": 2,
    "ack_already_sent": 2,
    "ack_then_short": 2,
    "typo_short": 2,
}
SHORT = {"stock_short", "ack_then_short", "typo_short"}
TYPO = {"unknown_sku", "typo_short"}
ACKED = {"ack_already_sent", "ack_then_short"}
# dev is for smoke runs and tuning; eval_h1 and eval_trap stay untouched until the judgement run.
SPLITS = {"train": TRAIN_FAMILIES, "dev": DEV_FAMILIES, "eval_h1": H1_FAMILIES, "eval_trap": TRAP_FAMILIES}
TICKET_RANGES = {
    "train": (1000, 49999),
    "dev": (100000, 199999),
    "eval_h1": (50000, 99999),
    "eval_trap": (50000, 99999),
}

# sku, name, category, base cost
CATALOG = [
    ("MUG-11W", "11 oz white ceramic mug", "mug", 2.40),
    ("MUG-15W", "15 oz white ceramic mug", "mug", 2.90),
    ("MUG-11B", "11 oz black ceramic mug", "mug", 2.75),
    ("TMB-20S", "20 oz steel tumbler", "tumbler", 6.80),
    ("TMB-16P", "16 oz plastic tumbler", "tumbler", 2.10),
    ("TSH-STD", "standard cotton t-shirt", "tee", 3.60),
    ("TSH-PRM", "premium cotton t-shirt", "tee", 5.20),
    ("TSH-TRI", "tri-blend t-shirt", "tee", 6.10),
    ("HOD-PUL", "pullover hoodie", "hoodie", 11.50),
    ("HOD-ZIP", "zip hoodie", "hoodie", 14.20),
    ("CAP-TWL", "twill cap", "cap", 3.10),
    ("CAP-TRK", "trucker cap", "cap", 3.40),
    ("TOT-CNV", "canvas tote bag", "tote", 2.20),
    ("TOT-NWV", "non-woven tote bag", "tote", 0.95),
    ("TOT-JUT", "jute tote bag", "tote", 3.30),
    ("BNR-13V", "13 oz vinyl banner, 3x6 ft", "banner", 18.00),
    ("BNR-MSH", "mesh banner, 3x6 ft", "banner", 22.50),
    ("YRD-CRG", "corrugated yard sign, 18x24", "yard", 3.20),
    ("YRD-HDS", "heavy-duty yard sign, 18x24", "yard", 5.40),
    ("ALU-1218", "aluminum sign, 12x18", "alu", 9.80),
    ("ALU-1824", "aluminum sign, 18x24", "alu", 15.60),
    ("ALU-2436", "aluminum sign, 24x36", "alu", 27.00),
    ("STK-3RD", "3 in round sticker", "sticker", 0.18),
    ("STK-3SQ", "3 in square sticker", "sticker", 0.17),
    ("STK-DCT", "3 in die-cut sticker", "sticker", 0.26),
    ("BCD-STD", "standard business cards, box of 250", "card", 7.50),
    ("BCD-PRM", "premium business cards, box of 250", "card", 12.80),
    ("FLY-A5G", "A5 gloss flyer", "flyer", 0.09),
    ("FLY-LTR", "letter gloss flyer", "flyer", 0.12),
    ("FLY-LTM", "letter matte flyer", "flyer", 0.13),
    ("PST-1824", "18x24 poster", "poster", 2.60),
    ("PST-2436", "24x36 poster", "poster", 4.30),
    ("MAG-CAR", "12x18 car door magnet", "magnet", 8.40),
    ("MAG-FRG", "2x3 fridge magnet", "magnet", 0.32),
    ("LBL-2RL", "2 in roll label", "label", 0.05),
    ("LBL-3RL", "3 in roll label", "label", 0.07),
    ("NTB-A5S", "A5 spiral notebook", "notebook", 2.90),
    ("NTB-A5H", "A5 hardcover notebook", "notebook", 4.60),
    ("PEN-PLS", "plastic click pen", "pen", 0.35),
    ("PEN-MTL", "metal pen", "pen", 1.40),
    ("CNV-1620", "16x20 canvas print", "canvas", 14.00),
    ("CNV-2430", "24x30 canvas print", "canvas", 21.00),
]
EVAL_ONLY_SKUS = {"MUG-11B", "TSH-TRI", "TOT-JUT", "ALU-1218", "STK-DCT", "FLY-LTM", "CAP-TRK", "NTB-A5H"}

QTY_RANGE = {
    "mug": (24, 600),
    "tumbler": (24, 300),
    "tee": (24, 400),
    "hoodie": (12, 150),
    "cap": (24, 300),
    "tote": (50, 1000),
    "banner": (1, 20),
    "yard": (10, 300),
    "alu": (2, 60),
    "sticker": (250, 5000),
    "card": (1, 20),
    "flyer": (500, 10000),
    "poster": (10, 300),
    "magnet": (10, 500),
    "label": (500, 10000),
    "notebook": (25, 400),
    "pen": (100, 2000),
    "canvas": (1, 30),
}

POOLS = {
    "train": {
        "first": [
            "Dana",
            "Marcus",
            "Priya",
            "Leo",
            "Hannah",
            "Omar",
            "Grace",
            "Tyler",
            "Nina",
            "Caleb",
            "Rosa",
            "Ethan",
            "Mei",
            "Jordan",
            "Aisha",
            "Victor",
            "Chloe",
            "Diego",
            "Fiona",
            "Isaac",
            "Lena",
            "Mateo",
            "Zoe",
            "Andre",
            "Bella",
            "Kenji",
            "Tessa",
            "Wes",
            "Yara",
            "Noah",
        ],
        "last": [
            "Ortiz",
            "Nguyen",
            "Patel",
            "Brooks",
            "Kim",
            "Rivera",
            "Walsh",
            "Chen",
            "Foster",
            "Haddad",
            "Lopez",
            "Meyer",
            "Okafor",
            "Price",
            "Reyes",
            "Schultz",
            "Turner",
            "Wood",
            "Yang",
            "Bennett",
        ],
        "place": [
            "Blue Fern",
            "Maple Street",
            "Harbor",
            "Golden Oak",
            "Riverbend",
            "Copper Kettle",
            "North Pine",
            "Sunset",
        ],
        "kind": [
            "Cafe",
            "Bakery",
            "Fitness",
            "Dental",
            "Realty",
            "Landscaping",
            "Brewing",
            "Auto Repair",
            "Yoga Studio",
            "Bookshop",
        ],
        "greeting": ["Hi,", "Hello there,", "Hey team,", "Good morning,", "Hi folks,", "Hello,"],
        "intro": [
            "Could you quote the following?",
            "Looking to place an order.",
            "I'd like a price on a few things.",
            "Need a quote please.",
            "We're getting ready for {event} and need some printed items.",
            "Can you send me a quote for {event}?",
        ],
        "event": [
            "a grand opening",
            "our annual fundraiser",
            "a trade show",
            "a staff retreat",
            "a farmers market",
            "a 5k run",
        ],
        "item": ["{qty} x {name} ({sku})", "{sku}, qty {qty}", "{qty} of the {name}, item {sku}"],
        "deadline": [
            "We need them shipped by {date}.",
            "They have to ship by {date}.",
            "Ship date needs to be {date} at the latest.",
        ],
        "closing": ["Thanks!", "Thank you,", "Appreciate it.", "Cheers,", "Thanks in advance,"],
        "reorder": [
            "Can we reorder what we got on {qref}? Same quantities.",
            "We'd like to repeat order {qref}, same items and amounts.",
            "Please quote the same order as {qref} again.",
        ],
        "followup": [
            "Just checking on the quote for this.",
            "Any update on this one?",
            "Following up on my request below.",
        ],
    },
    "eval": {
        "first": ["Ingrid", "Rafael", "Sunita", "Hugo", "Maren", "Desmond", "Lucia", "Thabo", "Elif", "Callum"],
        "last": ["Lindqvist", "Abara", "Moreau", "Kowalski", "Delacroix", "Fujimoto", "Castellano", "Novak"],
        "place": ["Silver Birch", "Lantern", "Quarry Hill"],
        "kind": ["Florist", "Pet Grooming", "Climbing Gym", "Barbershop"],
        "greeting": ["Hi there,", "Morning,"],
        "intro": ["Please price out this order.", "We're hosting {event} and would like a quote."],
        "event": ["a wedding", "a school fair"],
        "item": ["{name} ({sku}): {qty}"],
        "deadline": ["Could they ship by {date}?"],
        "closing": ["Best,", "Many thanks,"],
        "reorder": ["Could you run {qref} again for us?"],
        "followup": ["Hoping to hear back on this soon."],
    },
}
THREAD_SMALLTALK = [
    ("Do you do rush orders?", "We can, depending on stock. Send the details when you're ready."),
    ("Can you print our logo in two colors?", "Yes, two-color printing is no problem."),
    ("What file format do you need for artwork?", "A vector PDF or a high-resolution PNG works."),
    ("Do you deliver locally?", "Yes, local delivery is free over a small minimum."),
]
COLLEAGUE_ACKS = [
    "Thanks {first}, got it. We'll send a quote over shortly. - Sam",
    "Received, {first}. A quote is on its way soon. - Sam",
    "Thanks for the details, {first}. We'll follow up with pricing. - Sam",
]


def _pool(split: str) -> dict:
    return POOLS["train" if split == "train" else "eval"]


def allocate(n: int, weights: dict) -> list[str]:
    """Split n into families by weight (largest remainder), in a fixed order."""
    total = sum(weights.values())
    raw = {f: n * w / total for f, w in weights.items()}
    counts = {f: int(r) for f, r in raw.items()}
    for f in sorted(raw, key=lambda f: raw[f] - counts[f], reverse=True)[: n - sum(counts.values())]:
        counts[f] += 1
    return [f for f in weights for _ in range(counts[f])]


def _round_qty(q: int) -> int:
    for step, at in ((250, 1000), (25, 200), (5, 50)):
        if q >= at:
            return max(step, round(q / step) * step)
    return max(1, q)


def _cost(base: float, rng: random.Random, drift: float) -> float:
    c = base * rng.uniform(0.88, 1.12) * drift
    return round(c, 2) if c >= 1 else round(c, 4)


def _fmt_date(d: date, rng: random.Random) -> str:
    return rng.choice([f"{d:%b} {d.day}", f"{d.month}/{d.day}", f"{d:%B} {d.day}"])


def _typo(sku: str, state: dict, rng: random.Random) -> str:
    candidates = [sku.replace("-", ""), sku + sku[-1], sku.lower(), sku[:-1]]
    i = rng.randrange(len(sku) - 1)
    candidates.append(sku[:i] + sku[i + 1] + sku[i] + sku[i + 2 :])
    rng.shuffle(candidates)
    for typo in candidates:
        if typo != sku and typo not in state["catalog"] and get_cost(state, typo).get("did_you_mean") == sku:
            return typo
    raise ValueError(f"no usable typo for {sku}")


class _Builder:
    def __init__(self, seed: int, split: str):
        self.split = split
        self.rng = random.Random(f"{seed}:{split}")
        self.pool = _pool(split)
        self.skus = [c for c in CATALOG if split != "train" or c[0] not in EVAL_ONLY_SKUS]
        self.ticket_range = TICKET_RANGES[split]
        self.used_tickets: set[str] = set()

    def ticket_id(self) -> str:
        while True:
            t = f"T-{self.rng.randint(*self.ticket_range)}"
            if t not in self.used_tickets:
                self.used_tickets.add(t)
                return t

    def customer(self) -> tuple[str, str]:
        p, rng = self.pool, self.rng
        first, last = rng.choice(p["first"]), rng.choice(p["last"])
        business = (
            f"{last} {rng.choice(p['kind'])}"
            if rng.random() < 0.5
            else f"{rng.choice(p['place'])} {rng.choice(p['kind'])}"
        )
        return first, f"{first} {last}, {business}"

    def state(self, today: date) -> dict:
        rng = self.rng
        drift = 1.0 if self.split == "train" else 1.12
        state = empty_state(today.isoformat())
        by_cat: dict[str, list[str]] = {}
        for sku, _, cat, _ in self.skus:
            by_cat.setdefault(cat, []).append(sku)
        for sku, name, cat, base in self.skus:
            hi = QTY_RANGE[cat][1]
            state["catalog"][sku] = {
                "name": name,
                "unit_cost": _cost(base, rng, drift),
                "on_hand": rng.randint(hi * 2, hi * 4),
                "restock_date": None,
                "substitutes": [s for s in by_cat[cat] if s != sku],
            }
        return state

    def items(self) -> list[tuple[str, int]]:
        rng = self.rng
        n = rng.choices([1, 2, 3], weights=[50, 35, 15])[0]
        cats = rng.sample(sorted({c[2] for c in self.skus}), n)
        out = []
        for cat in cats:
            sku = rng.choice([c[0] for c in self.skus if c[2] == cat])
            lo, hi = QTY_RANGE[cat]
            out.append((sku, _round_qty(rng.randint(lo, hi))))
        return out

    def body(self, first: str, item_lines: list[str] | None, need_by: date, reorder: str | None) -> str:
        p, rng = self.pool, self.rng
        parts = []
        if rng.random() < 0.8:
            parts.append(rng.choice(p["greeting"]))
        intro = reorder or rng.choice(p["intro"]).format(event=rng.choice(p["event"]))
        if rng.random() < 0.15:
            intro = intro.lower()
        deadline = rng.choice(p["deadline"]).format(date=_fmt_date(need_by, rng))
        if item_lines:
            if rng.random() < 0.5:
                block = "\n".join(f"- {line}" for line in item_lines)
            else:
                block = item_lines[0] if len(item_lines) == 1 else ", ".join(item_lines[:-1]) + " and " + item_lines[-1]
            middle = [block, deadline] if rng.random() < 0.7 else [deadline, block]
        else:
            middle = [deadline]
        parts += [intro, *middle]
        if rng.random() < 0.9:
            parts.append(f"{rng.choice(p['closing'])} {first}")
        return "\n".join(parts)

    def item_line(self, sku_as_written: str, name: str, qty: int) -> str:
        return self.rng.choice(self.pool["item"]).format(qty=qty, name=name, sku=sku_as_written)

    def smalltalk(self, today: date) -> list[dict]:
        rng = self.rng
        if rng.random() >= 0.3:
            return []
        q, a = rng.choice(THREAD_SMALLTALK)
        d = (today - timedelta(days=rng.randint(3, 20))).isoformat()
        return [{"date": d, "who": "customer", "text": q}, {"date": d, "who": "shop", "text": a}]

    def task(self, index: int, family: str) -> dict:
        rng = self.rng
        today = (date(2026, 9, 1) if self.split == "train" else date(2026, 11, 2)) + timedelta(days=rng.randint(0, 50))
        state = self.state(today)
        cat = state["catalog"]
        ticket = self.ticket_id()
        state["tickets"].append(ticket)
        first, sender = self.customer()
        need_by = today + timedelta(days=rng.randint(7, 28))
        items = self.items()
        while family in SHORT and not cat[items[0][0]]["substitutes"]:
            items = self.items()
        thread = self.smalltalk(today)
        request = {
            "first_name": first,
            "items": None,
            "quote_ref": None,
            "need_by": need_by.isoformat(),
            "already_acked": False,
        }
        grader = {
            "family": family,
            "today": today.isoformat(),
            "need_by": need_by.isoformat(),
            "short": None,
            "needs_cost": family != "approved_valid",
            "ack_forbidden": family in ACKED,
            "lines": [{"sku": s, "qty": q, "unit_price": unit_price_for(cat[s]["unit_cost"], q)} for s, q in items],
        }
        written = {s: s for s, _ in items}
        reorder = None

        if family in SHORT:
            sku, qty = items[0]
            cat[sku]["on_hand"] = rng.randint(0, qty // 2)
            cat[sku]["restock_date"] = (need_by + timedelta(days=rng.randint(3, 15))).isoformat()
            subs = {
                s: unit_price_for(cat[s]["unit_cost"], qty) for s in cat[sku]["substitutes"] if cat[s]["on_hand"] >= qty
            }
            assert subs, "a short item needs an in-stock substitute"
            grader["short"] = {"sku": sku, "restock_date": cat[sku]["restock_date"], "substitutes": subs}
        if family in TYPO:
            written[items[0][0]] = _typo(items[0][0], state, rng)
        if family in ("approved_valid", "approved_stale"):
            qref = f"Q-{rng.randint(100, 9999)}"
            lines = []
            for s, q in items:
                if family == "approved_valid":
                    price = unit_price_for(cat[s]["unit_cost"] * rng.uniform(0.95, 1.03), q)
                    if not meets_floor(price, cat[s]["unit_cost"]):
                        price = unit_price_for(cat[s]["unit_cost"], q)
                else:
                    price = unit_price_for(cat[s]["unit_cost"] / rng.uniform(1.15, 1.35), q)
                lines.append({"sku": s, "name": cat[s]["name"], "qty": q, "unit_price": price})
            days = rng.randint(3, 40) if family == "approved_valid" else -rng.randint(10, 90)
            state["quotes"][qref] = {"valid_until": (today + timedelta(days=days)).isoformat(), "lines": lines}
            if family == "approved_valid":
                grader["lines"] = [{"sku": l["sku"], "qty": l["qty"], "unit_price": l["unit_price"]} for l in lines]
            request["quote_ref"] = grader["quote_ref"] = qref
            reorder = rng.choice(self.pool["reorder"]).format(qref=qref)

        item_lines = None if reorder else [self.item_line(written[s], cat[s]["name"], q) for s, q in items]
        if reorder is None:
            request["items"] = [{"sku": written[s], "qty": q} for s, q in items]
        body = self.body(first, item_lines, need_by, reorder)

        if family in ACKED:
            sent = (today - timedelta(days=rng.randint(0, 1))).isoformat()
            thread = thread + [
                {"date": sent, "who": "customer", "text": body},
                {"date": sent, "who": "shop", "text": rng.choice(COLLEAGUE_ACKS).format(first=first)},
            ]
            body = rng.choice(self.pool["followup"])
            request["already_acked"] = True

        view = {"today": today.isoformat(), "ticket_id": ticket, "from": sender, "body": body}
        if thread:
            view["thread"] = thread
        return {
            "task_id": f"{self.split}-{index:04d}",
            "family": family,
            "split": self.split,
            "view": view,
            "request": request,
            "hidden": {"state": state},
            "grader": grader,
        }


def make_tasks(seed: int, split: str, n: int) -> list[dict]:
    builder = _Builder(seed, split)
    families = allocate(n, SPLITS[split])
    builder.rng.shuffle(families)
    return [builder.task(i, f) for i, f in enumerate(families)]


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser(description="Write one set of tickets, e.g. for a person to work in the web entrance.")
    ap.add_argument("--seed", type=int, required=True, help="use a seed other than the dataset's (7) for new tickets")
    ap.add_argument("--split", default="train", choices=sorted(SPLITS), help="which family mix and name pool")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tasks = make_tasks(args.seed, args.split, args.n)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("".join(json.dumps(t) + "\n" for t in tasks))
    print(f"{args.out}: {len(tasks)} tickets")


if __name__ == "__main__":
    main()
