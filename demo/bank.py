"""The recorded runs the page plays when the live model is not there: one real run per set of picks, taken from the final
evaluation in order, the first that followed the manual. The page labels them recorded.

    python -m demo.bank --results runs/demo-final/results.jsonl --out site/recorded.json
"""

import argparse
import json
from pathlib import Path

from workalike.rules import PREAMBLE
from workalike.sft import read_jsonl

from .manual import DATE, PRICE
from .server import load_tickets
from .work import follows

KEEP = ("off", "outcome", "message", "lines", "quote_id", "question", "approvable", "failed", "seconds", "checks")


def light(events: list[dict]) -> list[dict]:
    out = []
    for e in events:
        if e["type"] == "done":
            out.append({"t": e["t"], "type": "done", **{k: e.get(k) for k in KEEP}})
        else:
            out.append({k: e[k] for k in ("t", "type", "text", "held") if k in e})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    bank = {}
    for r in sorted(read_jsonl(a.results), key=lambda r: r["round"]):
        if r["price"] is None:
            bank.setdefault(f"{r['kind']}/none", light(r["events"]))
        elif follows(r["kind"], r["price"] if r["kind"] == "price" else r["date"], r["done"]):
            bank.setdefault(f"{r['kind']}/{r['price']}/{r['date']}", light(r["events"]))
    # What the page shows beside them: the manual's words and the tickets as a person sees them, nothing hidden.
    view = {k: {f: t["view"][f] for f in ("from", "body", "thread", "today")} for k, t in load_tickets().items()}
    data = {"runs": bank, "manual": {"preamble": PREAMBLE, "price": PRICE, "date": DATE}, "tickets": view}
    Path(a.out).write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    print(len(bank), "runs", Path(a.out).stat().st_size, "bytes")


if __name__ == "__main__":
    main()
