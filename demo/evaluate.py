"""Every manual a visitor can build, on a set of tickets, with the failures in the count. Keeps each run's steps so a
recorded run can be played when the live model is not there.

    BASETEN_API_KEY=... python -m demo.evaluate --tickets demo/tickets.jsonl --rounds 3 --out runs/demo-final
"""

import argparse
import json
import os
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from workalike.sft import read_jsonl

from .manual import DATE, PRICE, manual
from .work import cost, dangerous, follows, run_arm


def jobs(tickets: dict, rounds: int) -> list[tuple]:
    out = []
    for r in range(1, rounds + 1):
        for kind in tickets:
            out.append((r, kind, None, None))
            out += [(r, kind, p, d) for p in PRICE for d in DATE]
    return out


def run(a) -> None:
    tickets = {t["kind"]: t for t in read_jsonl(a.tickets)}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = out / "results.jsonl"
    done = {(r["round"], r["kind"], r["price"], r["date"]) for r in read_jsonl(results)} if results.exists() else set()
    lock = threading.Lock()
    key = os.environ["BASETEN_API_KEY"]

    def one(job):
        if job in done:
            return
        r, kind, price, date = job
        start, events = time.monotonic(), []

        def emit(e):
            events.append({"t": round(time.monotonic() - start, 1), **e})

        last = run_arm(tickets[kind], manual(price, date) if price else "", key, emit)
        row = {"round": r, "kind": kind, "price": price, "date": date, "done": last, "events": events}
        with lock, results.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(r, kind, price, date, last["outcome"], last.get("off"), last["seconds"], flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, jobs(tickets, a.rounds)))
    report(read_jsonl(results))


def report(rows: list[dict]) -> None:
    manuals = [r for r in rows if r["price"]]
    hit = sum(follows(r["kind"], r["price"] if r["kind"] == "price" else r["date"], r["done"]) for r in manuals)
    risky = [r for r in rows if dangerous(r["done"])]
    secs = sorted(r["done"]["seconds"] for r in rows)
    usd = sum(cost(r["done"]) for r in rows)
    print(
        f"followed {hit}/{len(manuals)}  dangerous {len(risky)}/{len(rows)}  failed {sum(1 for r in rows if r['done'].get('failed'))}"
    )
    print(
        f"seconds median {statistics.median(secs)} p95 {secs[int(len(secs) * 0.95) - 1]} max {secs[-1]}  cost ${usd:.2f}, per ticket ${usd / len(rows):.3f}"
    )
    for r in manuals:
        pick = r["price"] if r["kind"] == "price" else r["date"]
        if not follows(r["kind"], pick, r["done"]):
            print(
                " missed",
                r["round"],
                r["kind"],
                r["price"],
                r["date"],
                r["done"]["outcome"],
                r["done"].get("off"),
                [c["name"] for c in r["done"]["checks"] if not c["ok"]],
                (r["done"].get("message") or "")[:160].replace("\n", " "),
            )
    for r in risky:
        print(
            " dangerous",
            r["round"],
            r["kind"],
            r["price"],
            r["date"],
            (r["done"].get("message") or "")[:200].replace("\n", " "),
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tickets", required=True)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=8)
    run(p.parse_args())


if __name__ == "__main__":
    main()
