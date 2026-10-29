"""One round of fixing a written procedure from the cases where following it went another way than the worker.

    python -m island.refine_procedure --procedure p1.txt --run runs/x/h20-p1 --ref data/v2/records/c-train.jsonl \
        --tasks data/v2/tasks/heldout-c20-s1.jsonl --env quote-desk/2 --model muse-glimmer --out p2.txt

The run is a model working tickets with the procedure in its prompt. Tickets where it made a different choice from
the reference worker at a choice point (island.score2) become cases: the ticket as shown, the points that differ,
and both action sequences. The model gets the current procedure and only those cases, and writes the whole
procedure again. Tickets it already handled the worker's way are not shown. The same prompt serves any model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from workalike.envs import BY_VERSION
from workalike.filter import clean_episode, episode_meta
from workalike.model_client import ChatModel
from workalike.record import group_episodes, read_events
from workalike.sft import read_jsonl

from .score2 import score_records

SYSTEM = ("You keep the written procedure for the quote desk of a small print shop, so that anyone following it "
          "handles tickets the way one particular worker does. Below are the current procedure and the tickets where "
          "someone following it made a different choice from that worker. Revise the procedure so that a person "
          "following it would make the worker's choices in cases like these, and keep what already works. Write the "
          "whole revised procedure, addressed to the next person at the desk.")


def steps(ep: dict) -> str:
    out = []
    for turn in ep["turns"]:
        for c in turn["calls"]:
            a = {k: v for k, v in c["args"].items() if k != "ticket_id"}
            r = c["result"] or {}
            said = r.get("customer_reply") or r.get("owner_reply")
            out.append(f"- {c['name']} {json.dumps(a, ensure_ascii=False)[:300]}"
                       + (f" -> {'error ' + r['error'] if 'error' in r else ''}" if "error" in r else "")
                       + (f"\n    answer: {said[:200]}" if said else ""))
    return "\n".join(out)


def cases(run_records, ref_records, tasks: dict, env, limit: int) -> list[str]:
    run_rows = {r["task_id"]: r for r in score_records(run_records, tasks)[0]}
    ref_rows = {r["task_id"]: r for r in score_records(ref_records, tasks)[0]}
    eps = {}
    for path, key in ((run_records, "run"), (ref_records, "ref")):
        for events in group_episodes(read_events(path)).values():
            tid = episode_meta(events)["task_id"]
            if tid in tasks:
                eps.setdefault(tid, {})[key] = clean_episode(events)
    out = []
    for tid, row in run_rows.items():
        ref = ref_rows[tid]
        diff = {k: (ref["way"].get(k), row["way"].get(k)) for k in row["points"] if row["way"].get(k) != ref["way"].get(k)}
        if not diff or tid not in eps or "run" not in eps[tid]:
            continue
        what = "; ".join(f"{k}: the worker {v[0]}, following the procedure {v[1]}" for k, v in diff.items())
        out.append(f"Ticket:\n{env.render_view(tasks[tid]['view'])}\n\nWhere it differed: {what}\n\n"
                   f"What the worker did:\n{steps(eps[tid]['ref'])}\n\nWhat was done following the procedure:\n"
                   f"{steps(eps[tid]['run'])}")
    return out[:limit]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--procedure", required=True)
    ap.add_argument("--run", required=True, help="folder with records.jsonl of a run that followed the procedure")
    ap.add_argument("--ref", required=True, help="the reference worker's records on the same tickets")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--env", required=True, choices=sorted(BY_VERSION))
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--limit", type=int, default=8, help="most cases to show at once (context length)")
    ap.add_argument("--max-tokens", type=int, default=12000, help="room for the model's thinking and the procedure")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    env = BY_VERSION[args.env]
    tasks = {t["task_id"]: t for t in read_jsonl(args.tasks)}
    shown = cases(Path(args.run) / "records.jsonl", args.ref, tasks, env, args.limit)
    current = Path(args.procedure).read_text().strip()
    if not shown:
        Path(args.out).write_text(current + "\n")
        print("no case differed; procedure unchanged")
        return
    user = f"Current procedure:\n{current}\n\n" + "\n\n---\n\n".join(f"Case {i}:\n{c}" for i, c in enumerate(shown, 1))
    model = ChatModel(args.base_url, args.model, temperature=0, max_tokens=args.max_tokens)
    revised = (model([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], None).get("content")
               or "").strip()
    if not revised:
        raise SystemExit(f"the model returned no procedure (finish reason {model.calls[-1].get('finish_reason')}, "
                         f"usage {model.calls[-1].get('usage')})")
    Path(args.out).write_text(revised + "\n")
    Path(args.out).with_suffix(".json").write_text(json.dumps({
        "from": args.procedure, "run": args.run, "cases": len(shown), "words": len(revised.split()),
        "writer": args.model}, indent=1) + "\n")
    print(f"{args.out}: {len(revised.split())} words after {len(shown)} cases")


if __name__ == "__main__":
    main()
