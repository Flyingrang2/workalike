"""Serve test-bed tickets through the web entrance so a person can work them.

    python -m testbed.serve_web --tasks data/h1/tasks/eval_h1.jsonl --limit 5 --out runs/web-me --actor me

Open http://127.0.0.1:8765, work each ticket and press Finish ticket. Ctrl+C stops
the server and scores whatever was recorded, with the same scorer as the tool entrance.
Run it again with the same --out to carry on: finished tickets are kept and skipped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from workalike.record import Recorder
from workalike.sft import read_jsonl
from workalike.web import WebDesk, keep_finished, make_server, sessions_for

from .score import score_records


def write_scores(out: Path, tasks: list[dict]) -> dict:
    rows, summary = score_records(out / "records.jsonl", {t["task_id"]: t for t in tasks})
    (out / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", required=True)
    ap.add_argument("--actor", default="person")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    tasks = read_jsonl(args.tasks)[: args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    done = keep_finished(out / "records.jsonl")
    todo = [t for t in tasks if t["task_id"] not in done]
    server = make_server(WebDesk(sessions_for(todo, args.actor), Recorder(out / "records.jsonl")), port=args.port)
    print(
        f"http://127.0.0.1:{args.port}  ({len(todo)} tickets to go, {len(tasks) - len(todo)} already finished, "
        "Ctrl+C to stop)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    print(json.dumps(write_scores(out, tasks), indent=1))


if __name__ == "__main__":
    main()
