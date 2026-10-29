"""The human sheet (10-29): the tickets two real people work, made from quote-desk/2's generator.

Two judgments, each as twin pairs (experiments/human-sheet.md in the planning folder):
  A  short stock: short_open (preference unknown) and short_constrained (the customer already said: all by the
     date, any style). Some pairs carry the same wrapper on both sides (an earlier acknowledgement, or an item
     code the customer already confirmed), so the thread layouts in eval are not new.
  B  a reorder whose quote expired; the customer pushes back on the new price once. The twins differ only in
     the discount authority (0% and 5%). room (5% off stays above the floor) and tight are strata, not twins.
A twin keeps every number of its pair (items, stock, cost, dates, owner) and changes the customer, the
wording and the quote ids, so a person does not meet the same email twice.

    python -m island.human_sheet --out data/human        # sets, the two people's orders, and the checks

htrain has five blocks of 12 (A 3 pairs, B 3 pairs each), so a person who stops after 24, 36 or 48 still has a
balanced set. heval is 24 tickets (12 pairs) with names, item codes and wording that htrain never uses.
hretest is 8 tickets: copies of heval tickets with another customer, other wording and the item code swapped
for one with the same numbers.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from workalike.desk import unit_price_for
from workalike.desk2 import floor_price

from .tasks import EVAL_ONLY_SKUS, POOLS
from .tasks2 import CONSTRAINT_TEXT, SAYS, Builder, reorder_room

SEED = 29
# Shipping, not arriving: the desk and the scorer count ship dates (a reviewer, 10-29).
SHIP_CONSTRAINT = {
    "short_constrained": ["We need all {qty} shipped by {date}, it's for an event. A different size or style is fine.",
                          "All {qty} have to ship by {date}; the exact style doesn't matter."],
    "short_impossible": CONSTRAINT_TEXT["short_impossible"],
}
SHIP_SAYS = {"constrained": ["As I said, we need all {qty} shipped by {date}; another size or style is fine."],
             "impossible": ["As I said, it has to be all {qty} of the {name}, shipped by {date}."]}
NAMES = {
    "heval": {"first": ["Oskar", "Yusra", "Benedikt", "Mirela", "Tomasz", "Adaeze", "Lorcan", "Svetla", "Paulo",
                        "Henrike", "Kwame", "Ottilie"],
              "last": ["Brandvold", "Mensah", "Iversen", "Quiroga", "Halvorsen", "Szabo", "Whitlock", "Arrieta"],
              "place": ["Ember Lane", "Saltmarsh", "Kestrel"],
              "kind": ["Ceramics Studio", "Tutoring Center", "Food Truck", "Choir"]},
    "hretest": {"first": ["Rosalind", "Emeka", "Signe", "Vittorio", "Nadia", "Bastian", "Marisol", "Fenwick"],
                "last": ["Ashdown", "Okoye", "Thorsen", "Galli", "Rahimi", "Lindholm"],
                "place": ["Pebble Creek", "Wren Hollow"],
                "kind": ["Garden Club", "Craft Brewery", "Swim School"]},
}
# Wording only heval and hretest use, beside quote-desk/2's eval wording; none of it is in the training pool.
EVAL_WORDING = {
    "greeting": ["Hi there,", "Morning,", "Hello again,", "Good afternoon,"],
    "intro": ["Please price out this order.", "We're hosting {event} and would like a quote.",
              "Could I get pricing on the items below?", "Putting together an order for {event}."],
    "event": ["a wedding", "a school fair", "a community picnic", "a charity auction"],
    "item": ["{name} ({sku}): {qty}", "{qty} pcs {sku} ({name})"],
    "deadline": ["Could they ship by {date}?", "We'd need them out the door by {date}.",
                 "Shipping on or before {date}, please."],
    "closing": ["Best,", "Many thanks,", "Kind regards,", "Talk soon,"],
    "reorder": ["Could you run {qref} again for us?", "We'd like the same as {qref} once more.",
                "Time to restock: same order as {qref}, please."],
    "followup": ["Hoping to hear back on this soon.", "Is there any news on our request?",
                 "Just nudging this one."],
}
RANGES = {"htrain": (400000, 499999), "heval": (500000, 599999), "hretest": (600000, 699999)}
ORDER_GAP = {"htrain": 5, "heval": 6}  # least distance between the two tickets of a pair in a person's order


class HumanBuilder(Builder):
    def __init__(self, seed: int, split: str):
        super().__init__(seed, "train" if split == "htrain" else "eval")
        self.split = split
        self.rng = random.Random(f"human:{seed}:{split}")
        if split != "htrain":
            self.pool = {**POOLS["eval"], **EVAL_WORDING, **NAMES[split]}
            self.request = set(EVAL_ONLY_SKUS)
        self.says = {**SAYS, **SHIP_SAYS}
        self.constraint_text = SHIP_CONSTRAINT

    def ticket_id(self) -> str:
        while True:
            t = f"T-{self.rng.randint(*RANGES[self.split])}"
            if t not in self.used:
                self.used.add(t)
                return t

    def person(self, avoid: str) -> dict:
        while True:
            first, last = self.rng.choice(self.pool["first"]), self.rng.choice(self.pool["last"])
            if first != avoid:
                place = f"{last} {self.rng.choice(self.pool['kind'])}" if self.rng.random() < 0.5 else \
                    f"{self.rng.choice(self.pool['place'])} {self.rng.choice(self.pool['kind'])}"
                return {"first": first, "sender": f"{first} {last}, {place}"}


def a_family(side: str, wrap: str | None) -> str:
    return "+".join(sorted(filter(None, [wrap, side])))


def pair(b: HumanBuilder, spec: dict, tasks: list) -> None:
    """Two tickets that share every number and differ in the condition (A) or the authority (B)."""
    base = b.base()
    base.update(owner=spec["owner"])
    pid = f"{spec['axis']}:{len(tasks)}"
    if spec["axis"] == "A":
        base.update(authority=spec["authority"], pref=spec["pref"])
        sides = [(a_family("short_open", spec["wrap"]), {}), (a_family("short_constrained", spec["wrap"]), {})]
    else:
        sides = [(spec["kind"], {"authority": 0.0}), (spec["kind"], {"authority": 0.05})]
    first = None
    for family, extra in sides:
        who = {} if first is None else b.person(avoid=base["first"])
        t = b.task(len(tasks), family, dict(base, **extra, **who, text_seed=b.rng.randrange(1 << 30)), pid)
        t["grader"]["spec"] = dict(spec)
        t["base"] = base  # dropped before writing; hretest copies are made from it
        tasks.append(t)
        first = family


def train_specs() -> list[dict]:
    out = []
    for blk in range(5):
        auth = [0.0, 0.03, 0.05]
        prefs = ["split", "later", "substitute"][blk % 3:] + ["split", "later", "substitute"][:blk % 3]
        for i in range(3):
            out.append({"axis": "A", "block": blk, "authority": auth[i], "pref": prefs[i],
                        "owner": "hold" if i == blk % 3 else "floor_ok",
                        "wrap": ("ack_followup", "typo_confirmed")[blk % 2] if i == (blk + 1) % 3 else None})
        kinds = ["reorder_room", "reorder_room", "reorder_tight"] if blk % 2 == 0 else \
            ["reorder_tight", "reorder_tight", "reorder_room"]
        for i in range(3):
            out.append({"axis": "B", "block": blk, "kind": kinds[i],
                        "owner": "hold" if i == (blk + 1) % 3 else "floor_ok"})
    return out


EVAL_SPECS = [
    {"axis": "A", "wrap": None, "authority": 0.0, "owner": "floor_ok", "pref": "split"},
    {"axis": "A", "wrap": None, "authority": 0.05, "owner": "hold", "pref": "substitute"},
    {"axis": "A", "wrap": "ack_followup", "authority": 0.03, "owner": "floor_ok", "pref": "later"},
    {"axis": "A", "wrap": "ack_followup", "authority": 0.05, "owner": "floor_ok", "pref": "split"},
    {"axis": "A", "wrap": "typo_confirmed", "authority": 0.0, "owner": "hold", "pref": "later"},
    {"axis": "A", "wrap": "typo_confirmed", "authority": 0.03, "owner": "floor_ok", "pref": "substitute"},
    {"axis": "B", "kind": "reorder_room", "owner": "floor_ok"},
    {"axis": "B", "kind": "reorder_room", "owner": "floor_ok"},
    {"axis": "B", "kind": "reorder_room", "owner": "hold"},
    {"axis": "B", "kind": "reorder_tight", "owner": "floor_ok"},
    {"axis": "B", "kind": "reorder_tight", "owner": "floor_ok"},
    {"axis": "B", "kind": "reorder_tight", "owner": "hold"},
]
# Which heval tickets get a copy in hretest: (pair index in EVAL_SPECS, side 0 open or 0% / 1 constrained or 5%).
# No typo pairs: their thread quotes the misspelt code, which a code swap would not carry over.
RETEST = [(0, 0), (1, 1), (2, 0), (3, 1), (6, 0), (8, 1), (10, 1), (11, 0)]


def swap_codes(task: dict, rng: random.Random) -> dict:
    """The same ticket with each requested item renamed to another item of its category, numbers kept."""
    cat = task["hidden"]["state"]["catalog"]
    asked = [w["sku"] for w in task["grader"]["wants"]]
    s = json.dumps(task)
    for sku in asked:
        others = sorted(k for k in cat if k != sku and k not in asked and k.split("-")[0] == sku.split("-")[0])
        other = rng.choice(others)
        for a, b, tag in ((sku, other, "\x00A\x00"), (cat[sku]["name"], cat[other]["name"], "\x00B\x00")):
            s = s.replace(a, tag).replace(b, a).replace(tag, b)
    return json.loads(s)


def retest_tasks(heval: list[dict], seed: int) -> list[dict]:
    b = HumanBuilder(seed, "hretest")
    pairs = {}
    for t in heval:
        pairs.setdefault(t["grader"]["pair"], []).append(t)
    ordered = sorted(pairs.values(), key=lambda p: int(p[0]["grader"]["pair"].split(":")[1]))
    out = []
    for k, (pi, side) in enumerate(RETEST):
        src = ordered[pi][side]
        base = dict(src["base"], **b.person(avoid=src["base"]["first"]), text_seed=b.rng.randrange(1 << 30))
        if src["grader"]["spec"]["axis"] == "B":
            base["authority"] = src["grader"]["authority"]
        t = b.task(k, src["family"], base, None)
        t = swap_codes(t, b.rng)
        t["grader"]["spec"] = src["grader"]["spec"]
        t["grader"]["retest_of"] = src["task_id"]
        out.append(t)
    return out


def make(seed: int = SEED) -> dict[str, list[dict]]:
    sets = {}
    for split, specs in (("htrain", train_specs()), ("heval", EVAL_SPECS)):
        b, tasks = HumanBuilder(seed, split), []
        for spec in specs:
            pair(b, spec, tasks)
        for i, t in enumerate(tasks):
            t["task_id"] = f"{split}-{i:04d}"
        sets[split] = tasks
    sets["hretest"] = retest_tasks(sets["heval"], seed)
    for i, t in enumerate(sets["hretest"]):
        t["task_id"] = f"hretest-{i:04d}"
    return sets


def order(tasks: list[dict], split: str, rng: random.Random, early: set[str] = frozenset()) -> list[str]:
    """A person's order: blocks of 12 kept for htrain, the two of a pair apart, open first about half the time."""
    if split == "hretest":
        ids = [t["task_id"] for t in tasks]
        rng.shuffle(ids)
        return ids
    if split == "heval":  # one of each pair in the first half (a ticket hretest copies is always there)
        by_pair = {}
        for t in tasks:
            by_pair.setdefault(t["grader"]["pair"], []).append(t["task_id"])
        free = [p for p, ids in by_pair.items() if not set(ids) & early]
        opens = set(rng.sample(free, len(by_pair) // 2 - sum(ids[0] in early for ids in by_pair.values())))
        firsts = [ids[0] if (ids[0] in early or p in opens) else ids[1] for p, ids in by_pair.items()
                  if not (set(ids) & early and ids[1] in early)] + \
                 [ids[1] for ids in by_pair.values() if ids[1] in early]
        seconds = [x for ids in by_pair.values() for x in ids if x not in firsts]
        for _ in range(20000):
            rng.shuffle(firsts)
            rng.shuffle(seconds)
            pos = {x: i for i, x in enumerate(firsts + seconds)}
            if min(abs(pos[a] - pos[b]) for a, b in by_pair.values()) >= ORDER_GAP[split]:
                return firsts + seconds
        raise RuntimeError(f"no order for {split}")
    groups = [tasks[i:i + 12] for i in range(0, len(tasks), 12)]
    out = []
    for g in groups:
        for _ in range(20000):
            ids = [t["task_id"] for t in g]
            rng.shuffle(ids)
            pos = {x: i for i, x in enumerate(ids)}
            by_pair = {}
            for t in g:
                by_pair.setdefault(t["grader"]["pair"], []).append(t["task_id"])
            gaps = [abs(pos[a] - pos[b]) for a, b in by_pair.values()]
            first_open = sum(pos[a] < pos[b] for a, b in by_pair.values())
            if min(gaps) >= ORDER_GAP.get(split, 0) and abs(2 * first_open - len(by_pair)) <= 1 \
                    and all(pos[x] < len(ids) // 2 for x in early):
                out += ids
                break
        else:
            raise RuntimeError(f"no order for {split}")
    return out


def check(sets: dict[str, list[dict]]) -> list[str]:
    """What must hold before a person sees the sheet."""
    bad = []
    for split in ("htrain", "heval"):
        pairs = {}
        for t in sets[split]:
            pairs.setdefault(t["grader"]["pair"], []).append(t)
        for pid, (a, b) in pairs.items():
            sa, sb = a["hidden"]["state"], b["hidden"]["state"]
            ka, kb = list(sa["customers"])[0], list(sb["customers"])[0]
            items = [[(w["sku"], w["qty"]) for w in x["grader"]["wants"]] for x in (a, b)]
            if sa["catalog"] != sb["catalog"] or items[0] != items[1] \
                    or a["grader"]["need_by"] != b["grader"]["need_by"] or a["grader"]["today"] != b["grader"]["today"]:
                bad.append(f"{split} {pid}: numbers differ between twins")
            if sa["owner"][ka]["policy"] != sb["owner"][kb]["policy"]:
                bad.append(f"{split} {pid}: owner differs")
            if a["view"]["from"] == b["view"]["from"] or \
                    (a["view"]["body"], a["view"].get("thread")) == (b["view"]["body"], b["view"].get("thread")):
                bad.append(f"{split} {pid}: same customer or same email")
            if a["grader"]["spec"]["axis"] == "B":
                qa, qb = list(sa["quotes"].values())[0], list(sb["quotes"].values())[0]
                if qa["lines"] != qb["lines"] or qa["valid_until"] != qb["valid_until"]:
                    bad.append(f"{split} {pid}: old quote differs")
                if {a["grader"]["authority"], b["grader"]["authority"]} != {0.0, 0.05}:
                    bad.append(f"{split} {pid}: authority pair")
            else:
                if a["grader"]["authority"] != b["grader"]["authority"]:
                    bad.append(f"{split} {pid}: authority differs in an A pair")
    for split, tasks in sets.items():
        for t in tasks:
            g, st = t["grader"], t["hidden"]["state"]
            if set(t["view"]) - {"today", "ticket_id", "from", "body", "authority", "sent_so_far", "thread"}:
                bad.append(f"{t['task_id']}: view leaks {set(t['view'])}")
            if "here by" in json.dumps(t["view"]) + json.dumps(st["customers"]):
                bad.append(f"{t['task_id']}: 'here by'")
            if g["spec"]["axis"] == "B":
                sku, qty = g["wants"][0]["sku"], g["wants"][0]["qty"]
                room = reorder_room(st["catalog"][sku]["unit_cost"], qty)
                if room != (g["spec"]["kind"] == "reorder_room"):
                    bad.append(f"{t['task_id']}: room/tight relation broken")
            if g["spec"]["axis"] == "A" and "short_constrained" in g["flags"] and not g["short"]["substitutes"]:
                bad.append(f"{t['task_id']}: no substitute for a constrained ticket")
    asked = {s: {w["sku"] for t in ts for w in t["grader"]["wants"]} for s, ts in sets.items()}
    if asked["htrain"] & EVAL_ONLY_SKUS or not asked["heval"] <= EVAL_ONLY_SKUS:
        bad.append(f"item codes overlap: train {asked['htrain'] & EVAL_ONLY_SKUS}, eval {asked['heval'] - EVAL_ONLY_SKUS}")
    firsts = {s: {t["view"]["from"].split()[0] for t in ts} for s, ts in sets.items()}
    for a in firsts:
        for b in firsts:
            if a < b and firsts[a] & firsts[b]:
                bad.append(f"names shared by {a} and {b}: {firsts[a] & firsts[b]}")
    return bad


def main() -> None:
    import argparse
    import hashlib

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="data/human")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    sets = make(args.seed)
    bad = check(sets)
    for b in bad:
        print("CHECK", b)
    if bad:
        raise SystemExit(1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    early = {t["grader"]["retest_of"] for t in sets["hretest"]}
    manifest = {"seed": args.seed, "sha256": {}, "orders": {}}
    for split, tasks in sets.items():
        for t in tasks:
            t.pop("base", None)
        path = out / f"{split}.jsonl"
        path.write_text("".join(json.dumps(t, default=str) + "\n" for t in tasks))
        manifest["sha256"][split] = hashlib.sha256(path.read_bytes()).hexdigest()
    by_id = {t["task_id"]: t for ts in sets.values() for t in ts}
    for person in ("p1", "p2"):
        rng = random.Random(f"order:{args.seed}:{person}")
        manifest["orders"][person] = {
            "htrain": order(sets["htrain"], "htrain", rng),
            "heval": order(sets["heval"], "heval", rng, early),
            "hretest": order(sets["hretest"], "hretest", rng)}
        d = out / person
        d.mkdir(exist_ok=True)
        for split, ids in manifest["orders"][person].items():
            (d / f"{split}.jsonl").write_text("".join(json.dumps(by_id[i], default=str) + "\n" for i in ids))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    for split, tasks in sets.items():
        print(f"{split}: {len(tasks)} tickets, sha256 {manifest['sha256'][split][:16]}")


if __name__ == "__main__":
    main()
