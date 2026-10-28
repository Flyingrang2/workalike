"""Teacher-forced next action: give a model a reference worker's exact history up to each of its turns and see
whether it takes the same next action.

    python -m island.next_action --records data/v2/records/c-train.jsonl --tasks data/v2/tasks/train.jsonl \
        --env quote-desk/2 --model fA60 --out runs/x/next-fA60.json

A full run can go wrong at one turn and drift from there. Here every turn starts from the reference's own history,
so a miss means the model would not take that step even on the right path. The action compared is the first
tool call of the turn: its name, and for send_reply whether a quote is attached (an answer with a quote, or a
message without one). Turns where the reference makes a choice the ticket's conditions call for (a reply without
a quote, a question to the owner, a second quote) are reported apart as choice turns.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from workalike.envs import BY_VERSION
from workalike.filter import clean_episode, episode_meta
from workalike.model_client import ChatModel
from workalike.record import group_episodes, read_events
from workalike.serialize import to_messages
from workalike.sft import read_jsonl


def action(call: dict | None) -> str:
    if call is None:
        return "stop"
    if call["name"] == "send_reply":
        args = call.get("arguments") or call.get("args") or {}
        return "reply+quote" if args.get("quote_id") else "reply"
    return call["name"]


def choice_turn(calls_before: list[dict], call: dict) -> bool:
    """A turn where the reference made a conditional choice rather than a routine step."""
    a = action(call)
    if a == "ask_owner":
        return True
    if a == "reply" and not any(action(c) == "reply+quote" for c in calls_before):
        return True
    return a == "create_quote" and any(c["name"] == "create_quote" for c in calls_before)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--env", required=True, choices=sorted(BY_VERSION))
    ap.add_argument("--model", required=True)
    ap.add_argument("--procedure", help="text file added to the system prompt, as run_agent --procedure")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--choice-only", action="store_true", help="ask only at the reference's choice turns")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    env = BY_VERSION[args.env]
    families = {t["task_id"]: t["family"] for t in read_jsonl(args.tasks)}
    model = ChatModel(args.base_url, args.model, temperature=0, max_tokens=4096)
    rows = []
    for events in group_episodes(read_events(args.records)).values():
        task_id = episode_meta(events)["task_id"]
        if task_id not in families:
            continue
        ep = clean_episode(events)
        messages = to_messages(ep, env)
        if args.procedure:
            messages[0] = {**messages[0], "content": messages[0]["content"] + "\n\n" + Path(args.procedure).read_text().strip()}
        done = []
        at = 2  # system, user; then one assistant message (and its tool messages) per turn
        for turn in ep["turns"]:
            ref = turn["calls"][0]
            if args.choice_only and not choice_turn(done, ref):
                done += turn["calls"]
                at += 1 + len(turn["calls"])
                continue
            reply = model(messages[:at], env.tool_specs)
            got = (reply.get("tool_calls") or [None])[0]
            rows.append({"task_id": task_id, "family": families[task_id], "turn": len(done),
                         "ref": action(ref), "got": action(got), "same": action(ref) == action(got),
                         "choice": choice_turn(done, ref)})
            done += turn["calls"]
            at += 1 + len(turn["calls"])
    choice = [r for r in rows if r["choice"]]
    summary = {"model": args.model, "procedure": args.procedure, "turns": len(rows),
               "same": sum(r["same"] for r in rows), "choice_turns": len(choice),
               "choice_same": sum(r["same"] for r in choice),
               "choice_by_ref": {a: f"{sum(r['same'] for r in choice if r['ref'] == a)}/{sum(r['ref'] == a for r in choice)}"
                                 for a in sorted({r["ref"] for r in choice})},
               "choice_misses": dict(Counter(f"{r['family']}: {r['ref']} -> {r['got']}" for r in choice if not r["same"]))}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
