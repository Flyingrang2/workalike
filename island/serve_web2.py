"""Serve quote-desk/2 tickets to a person who reads and writes Korean (workalike.web2).

    python -m island.serve_web2 --tasks data/v2/tasks/train.jsonl --limit 10 --out runs/web-me --actor me \
        --korean <translations ko.json> --translator "<command: Korean on stdin, English on stdout>"

Open http://127.0.0.1:8765, work each ticket and press 이 티켓 끝내기. Ctrl+C stops the server and scores
what was recorded with the quote-desk/2 scorer. Run it again with the same --out to carry on: finished
tickets are kept and skipped.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from workalike.record import Recorder
from workalike.sft import read_jsonl
from workalike.web import make_server
from workalike.web2 import Korean, KoreanDesk, command_translator

from .score2 import score_records
from .serve_web import keep_finished, sessions_for


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--skip", type=int, default=0, help="start after this many tickets of the file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--actor", default="person")
    ap.add_argument("--korean", required=True, help="JSON with the tickets' Korean and the fixed answers")
    ap.add_argument("--translator", required=True)
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    tasks = read_jsonl(args.tasks)[args.skip:][: args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    done = keep_finished(out / "records.jsonl")
    todo = [t for t in tasks if t["task_id"] not in done]
    desk = KoreanDesk(sessions_for(todo, args.actor), Recorder(out / "records.jsonl"),
                      Korean(json.loads(Path(args.korean).read_text())), command_translator(args.translator))
    server = make_server(desk, port=args.port)
    print(f"http://127.0.0.1:{args.port}  ({len(todo)} tickets to go, {len(tasks) - len(todo)} already finished, "
          "Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    rows, summary = score_records(out / "records.jsonl", {t["task_id"]: t for t in tasks})
    (out / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
