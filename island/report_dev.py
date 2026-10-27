"""Compare dev runs with the two scripted workers, for tuning on dev before anything touches the eval splits.

    python -m island.report_dev --tasks data/h1/tasks/dev.jsonl --records data/h1/records \
        base=runs/e2/base a:A=runs/e2/a b:B=runs/e2/b proc-a:A=runs/e2/proc-a --out runs/e2/report.json

A run is name[:style]=folder, where folder holds the scores.jsonl from island.run_agent.
Every task in the tasks file must appear exactly once in every run, or the report stops.
Each run is compared with both scripted workers (A and B columns) on the tickets where
the two workers differ. With a style, each ticket also gets one outcome against that
worker, first match wins:
  unfinished       the episode did not finish (turn limit, server error, no record)
  condition_wrong  wrong on a ticket where a condition changes the usual handling
  ordinary_wrong   wrong on a standard ticket
  way              right, but not handled the way that worker handled it
  match            right, and handled that worker's way
Tickets are also counted in three groups: single conditions as in eval_h1, habit
restraint as in eval_trap, and the dev-only combinations. Token counts are per ticket;
runs that shared a server are not compared on seconds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from workalike.sft import read_jsonl

from .score import score_records

GROUPS = {"single": ("standard", "stock_short", "unknown_sku"),
          "restraint": ("approved_valid", "ack_already_sent", "approved_stale"),
          "combined": ("ack_then_short", "typo_short")}
WAY = ("a_way", "b_way", "acked", "called_get_cost", "short_choice")


def same_way(row: dict, ref: dict) -> bool:
    return row["correct"] and all(row[k] == ref[k] for k in WAY)


def outcome(row: dict, ref: dict) -> str:
    if row["status"] != "done":
        return "unfinished"
    if not row["correct"]:
        return "ordinary_wrong" if row["family"] == "standard" else "condition_wrong"
    return "match" if same_way(row, ref) else "way"


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def load_run(folder: Path, by_id: dict) -> list[dict]:
    rows = read_jsonl(folder / "scores.jsonl")
    counts = Counter(r["task_id"] for r in rows)
    missing, extra = set(by_id) - set(counts), set(counts) - set(by_id)
    twice = {t for t, n in counts.items() if n > 1}
    if missing or extra or twice:
        raise SystemExit(f"{folder}: {len(rows)} rows for {len(by_id)} tasks; missing {sorted(missing)[:5]}, "
                         f"unknown {sorted(extra)[:5]}, twice {sorted(twice)[:5]}")
    return [{**r, "family": by_id[r["task_id"]]["family"]} for r in rows]


def report(tasks: list[dict], records_dir, runs: list[tuple[str, str | None, Path]]) -> dict:
    by_id = {t["task_id"]: t for t in tasks}
    paths = {s: Path(records_dir) / f"{s.lower()}-dev.jsonl" for s in ("A", "B")}
    refs = {s: {r["task_id"]: r for r in score_records(p, by_id)[0]} for s, p in paths.items()}
    split = [t for t in by_id if not same_way(refs["A"][t], refs["B"][t])]
    out = {"inputs": {"tasks": len(tasks), "differ": len(split),
                      "hashes": {"tasks": hashlib.sha256(json.dumps(tasks, sort_keys=True).encode()).hexdigest()[:16],
                                 **{f"ref_{s}": sha(p) for s, p in paths.items()}}},
           "runs": {}}
    for name, style, folder in runs:
        rows = load_run(folder, by_id)
        summary = folder / "summary.json"
        settings = json.loads(summary.read_text()).get("settings") if summary.exists() else None
        per = []
        for r in rows:
            per.append({"task_id": r["task_id"], "family": r["family"], "status": r["status"],
                        "correct": r["correct"], "business_correct": r["business_correct"],
                        "restraint_ok": r["restraint_ok"], "short_choice": r["short_choice"],
                        "like_A": same_way(r, refs["A"][r["task_id"]]), "like_B": same_way(r, refs["B"][r["task_id"]]),
                        "outcome": outcome(r, refs[style][r["task_id"]]) if style else None,
                        "reasons": r["reasons"], "trace": r["trace"],
                        "completion_tokens": r.get("completion_tokens"), "prompt_tokens": r.get("prompt_tokens")})
        diff = [p for p in per if p["task_id"] in split]
        tokens = [p["completion_tokens"] for p in per if p["completion_tokens"] is not None]
        out["runs"][name] = {
            "style": style, "folder": str(folder), "scores_sha": sha(folder / "scores.jsonl"), "settings": settings,
            "correct": sum(p["correct"] for p in per), "business_correct": sum(p["business_correct"] for p in per),
            "restraint_failed": sum(not p["restraint_ok"] for p in per),
            "like_A": sum(p["like_A"] for p in diff), "like_B": sum(p["like_B"] for p in diff),
            "outcomes": dict(Counter(p["outcome"] for p in per)) if style else None,
            "status": dict(Counter(p["status"] for p in per)),
            "groups": {g: {"n": sum(p["family"] in fams for p in per),
                           "correct": sum(p["correct"] for p in per if p["family"] in fams),
                           "like_A": sum(p["like_A"] for p in per if p["family"] in fams),
                           "like_B": sum(p["like_B"] for p in per if p["family"] in fams)}
                       for g, fams in GROUPS.items()},
            "completion_tokens_per_ticket": round(sum(tokens) / len(tokens)) if tokens else None,
            "tickets": per,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--records", required=True, help="folder with the scripted a-dev.jsonl and b-dev.jsonl")
    ap.add_argument("--out", help="write the full report here as JSON")
    ap.add_argument("runs", nargs="+", help="name[:A|B]=folder")
    args = ap.parse_args()
    runs = []
    for spec in args.runs:
        label, folder = spec.split("=", 1)
        name, _, style = label.partition(":")
        runs.append((name, style.upper() or None, Path(folder)))
    result = report(read_jsonl(args.tasks), args.records, runs)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    n, d = result["inputs"]["tasks"], result["inputs"]["differ"]
    print(f"{n} tickets, {d} where A and B differ; like_A and like_B count those {d}")
    for name, r in result["runs"].items():
        groups = " ".join(f"{g} {v['correct']}/{v['n']}" for g, v in r["groups"].items())
        print(f"{name:10} {r['style'] or '-'} correct {r['correct']}/{n} (business {r['business_correct']}, "
              f"restraint failed {r['restraint_failed']}) like_A {r['like_A']}/{d} like_B {r['like_B']}/{d} "
              f"| {groups} | tokens/ticket {r['completion_tokens_per_ticket']} | {r['outcomes'] or r['status']}")


if __name__ == "__main__":
    main()
