"""Compare quote-desk/2 runs on dev with a reference worker (C now, the person later).

    python -m testbed.report_v2 --tasks data/v2/tasks/dev.jsonl --ref data/v2/records/c-dev.jsonl \
        base=runs/v2-dev/base c20s1=runs/v2-curve/c20s1 --out runs/v2-dev/report.json

Each run folder holds records.jsonl from testbed.run_agent (--env quote-desk/2); it is scored again
here with testbed.score2, so every run and the reference share one scorer. Per run: valid outcomes,
handled the reference's way, both, twins right on both sides, ends, and tokens per ticket.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from workalike.sft import read_jsonl

from .score2 import compare, score_records


def tokens(folder: Path) -> int | None:
    path = folder / "scores.jsonl"
    if not path.exists():
        return None
    got = [r.get("completion_tokens") for r in read_jsonl(path) if r.get("completion_tokens") is not None]
    return round(sum(got) / len(got)) if got else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--ref", required=True, help="the reference worker's records on the same tickets")
    ap.add_argument("--out")
    ap.add_argument("runs", nargs="+", help="name=folder")
    args = ap.parse_args()
    by_id = {t["task_id"]: t for t in read_jsonl(args.tasks)}
    ref_rows, _ = score_records(args.ref, by_id)
    report = {"tasks": len(by_id), "ref": args.ref, "runs": {}}
    for spec in args.runs:
        name, folder = spec.split("=", 1)
        rows, summary = score_records(Path(folder) / "records.jsonl", by_id)
        c = compare(rows, ref_rows)
        report["runs"][name] = {
            **c,
            "end": summary["end"],
            "tool_errors": summary["tool_errors"],
            "completion_tokens_per_ticket": tokens(Path(folder)),
            "by_family": {
                f: {
                    "n": sum(r["family"] == f for r in rows),
                    "business_ok": sum(r["business_ok"] for r in rows if r["family"] == f),
                }
                for f in sorted({r["family"] for r in rows})
            },
            "tickets": rows,
        }
        print(
            f"{name:10} valid {c['business_ok']}/{c['tasks']}  same way {c['same_way']}  both {c['both']}  "
            f"twins {c['pairs_both_sides']}/{c['pairs']}  ends {dict(Counter(r['end'] for r in rows))}  "
            f"tool errors {summary['tool_errors']}  tokens/ticket {report['runs'][name]['completion_tokens_per_ticket']}"
        )
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
