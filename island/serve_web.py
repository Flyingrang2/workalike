"""Serve test-bed tickets through the web entrance so a person can work them.

    python -m island.serve_web --tasks data/h1/tasks/eval_h1.jsonl --limit 5 --out runs/web-me --actor me

Open http://127.0.0.1:8765, work each ticket and press Finish ticket. Ctrl+C stops
the server and scores whatever was recorded, with the same scorer as the tool entrance.
Run it again with the same --out to carry on: finished tickets are kept and skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from workalike.record import Recorder, group_episodes, read_events
from workalike.sft import read_jsonl
from workalike.web import WebDesk, make_server

from .score import score_records


def sessions_for(tasks: list[dict], actor: str) -> list[dict]:
    return [{"episode": "e-" + hashlib.sha256(f"{actor}:{t['task_id']}".encode()).hexdigest()[:10],
             "view": t["view"], "state": t["hidden"]["state"], "task_id": t["task_id"], "actor": actor}
            for t in tasks]


def keep_finished(records: Path) -> set[str]:
    """Keep the finished tickets of an earlier sitting and drop any half-worked one, so a person can
    stop and come back. Returns the task ids already finished."""
    if not records.exists():
        records.write_text("")
        return set()
    episodes = group_episodes(read_events(records)).values()
    kept = [events for events in episodes if events[-1]["type"] == "episode_end"]
    records.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for events in kept for e in events))
    return {events[0]["task_id"] for events in kept}


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
    print(f"http://127.0.0.1:{args.port}  ({len(todo)} tickets to go, {len(tasks) - len(todo)} already finished, "
          "Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    print(json.dumps(write_scores(out, tasks), indent=1))


if __name__ == "__main__":
    main()
