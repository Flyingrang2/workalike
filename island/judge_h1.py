"""Judge Hypothesis 1 from run folders, with the criteria fixed before measuring (2026-10-27).

    python -m island.judge_h1 --base runs/base-eval_h1 --a runs/a-eval_h1 --b runs/b-eval_h1 \
        --a-trap runs/a-eval_trap --b-trap runs/b-eval_trap

Each folder holds the scores.jsonl written by island.run_agent. Pass means all of:
  correct    each adapter gets at least 25 of the 30 eval_h1 tickets right, and no more than 3 fewer than base
  order      correct tickets done the adapter's way: a_way >= 25 for A, b_way >= 25 for B
  shortage   of the 7 stock_short tickets, at least 6 correct with the adapter's own choice
             (A keeps the item with a later date, B quotes a substitute)
  restraint  in eval_trap, at least 3 of 4 correct in each of the three families
If order passes and anything else fails, the verdict is "order imitated, conditional behaviour not reproduced".
The first criteria (raw cost-first order A >= 25, B <= 5, correct within 3 of base) are reported alongside.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

H1_TICKETS, SHORT_TICKETS = 30, 7


def load(folder) -> list[dict]:
    with (Path(folder) / "scores.jsonl").open() as f:
        return [json.loads(line) for line in f if line.strip()]


def count(rows, pred) -> int:
    return sum(1 for r in rows if pred(r))


def judge(base, a, b, a_trap, b_trap) -> dict:
    for name, rows in (("base", base), ("a", a), ("b", b)):
        if len(rows) != H1_TICKETS:
            raise ValueError(f"{name} has {len(rows)} eval_h1 rows, expected {H1_TICKETS}")
    base_correct = count(base, lambda r: r["correct"])
    checks = {}
    for name, rows, way, choice, trap in (("A", a, "a_way", "later_date", a_trap), ("B", b, "b_way", "substitute", b_trap)):
        correct = count(rows, lambda r: r["correct"])
        short = [r for r in rows if r["family"] == "stock_short"]
        by_family = {}
        for r in trap:
            by_family.setdefault(r["family"], []).append(r)
        checks[name] = {
            "correct": {"value": correct, "base": base_correct,
                        "pass": correct >= 25 and correct >= base_correct - 3},
            "order": {"value": count(rows, lambda r: r[way]), "pass": count(rows, lambda r: r[way]) >= 25},
            "shortage": {"value": count(short, lambda r: r["correct"] and r["short_choice"] == choice),
                         "of": len(short),
                         "pass": len(short) == SHORT_TICKETS
                         and count(short, lambda r: r["correct"] and r["short_choice"] == choice) >= 6},
            "restraint": {"value": {f: count(rs, lambda r: r["correct"]) for f, rs in sorted(by_family.items())},
                          "pass": len(by_family) == 3 and all(count(rs, lambda r: r["correct"]) >= 3 and len(rs) == 4
                                                              for rs in by_family.values())},
        }
    first_a = count(a, lambda r: r["cost_before_first_reply"])
    first_b = count(b, lambda r: r["cost_before_first_reply"])
    first = {"a_cost_first": first_a, "b_cost_first": first_b,
             "pass": first_a >= 25 and first_b <= 5
             and all(checks[n]["correct"]["value"] >= base_correct - 3 for n in "AB")}
    order_ok = all(checks[n]["order"]["pass"] for n in "AB")
    all_ok = all(c["pass"] for n in "AB" for c in checks[n].values())
    verdict = "pass" if all_ok else ("order imitated, conditional behaviour not reproduced" if order_ok else "fail")
    return {"verdict": verdict, "checks": checks, "first_criteria": first}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for name in ("base", "a", "b", "a-trap", "b-trap"):
        ap.add_argument(f"--{name}", required=True)
    args = ap.parse_args()
    result = judge(load(args.base), load(args.a), load(args.b), load(args.a_trap), load(args.b_trap))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
