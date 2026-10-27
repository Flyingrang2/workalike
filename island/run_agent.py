"""Run a model on test-bed tickets through the tool entrance, record it, and score it.

    python -m island.run_agent --tasks data/h1/tasks/eval_h1.jsonl \
        --base-url http://localhost:8000/v1 --model a --out runs/a-eval_h1

--model is the served name: the base model, or a LoRA adapter registered with
vLLM's --lora-modules. --memory puts earlier records of one worker in the system
prompt instead (the memory baseline); the records come from the same files the
adapters were trained on. --procedure adds a short written procedure (the prompt
baseline). Every task in the file counts: one with no record is scored as failed.
Tokens and seconds are kept per episode.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from workalike import agent
from workalike.desk import Desk
from workalike.filter import clean_episode
from workalike.model_client import ChatModel
from workalike.record import Recorder, group_episodes, read_events
from workalike.serialize import SYSTEM_PROMPT, canonical_json, order_args, render_view
from workalike.sft import read_jsonl

from .score import score_records


def memory_block(records_path, k: int) -> str:
    episodes = [clean_episode(evs) for evs in group_episodes(read_events(records_path)).values()][:k]
    parts = ["Records of how earlier tickets at this desk were handled:"]
    for i, ep in enumerate(episodes, 1):
        lines = [f"Ticket {i}:", render_view(ep["view"]), "Actions:"]
        for turn in ep["turns"]:
            for c in turn["calls"]:
                lines.append(f"- {c['name']} {canonical_json(order_args(c['name'], c['args']))} -> {canonical_json(c['result'])}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-turns", type=int, default=24)
    ap.add_argument("--memory", help="records file whose episodes go into the system prompt")
    ap.add_argument("--memory-k", type=int, default=30)
    ap.add_argument("--procedure", help="text file with a procedure written from the records (prompt baseline)")
    args = ap.parse_args()

    tasks = read_jsonl(args.tasks)[: args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = out / "records.jsonl"
    records.write_text("")
    system = SYSTEM_PROMPT
    if args.memory:
        system += "\n\n" + memory_block(args.memory, args.memory_k)
    if args.procedure:
        system += "\n\n" + Path(args.procedure).read_text().strip()
    params = {"max_tokens": args.max_tokens}
    if args.temperature is not None:
        params["temperature"] = args.temperature
    model = ChatModel(args.base_url, args.model, **params)
    recorder = Recorder(records)
    actor = f"model:{args.model}" + (":memory" if args.memory else "") + (":procedure" if args.procedure else "")
    per_episode = {}

    for task in tasks:
        ep = "e-" + hashlib.sha256(f"{actor}:{task['task_id']}".encode()).hexdigest()[:10]
        before = len(model.calls)
        recorder.start(ep, task["view"], task_id=task["task_id"], actor=actor, entrance="tool")
        try:
            _, status = agent.run(Desk(task["hidden"]["state"], recorder), task["view"], model,
                                  args.max_turns, system)
        except Exception as e:  # keep going; the record shows where it stopped
            status = f"error: {type(e).__name__}: {e}"
        recorder.end(status)
        calls = model.calls[before:]
        per_episode[ep] = {"model_calls": len(calls), "seconds": round(sum(c["seconds"] for c in calls), 1),
                           "prompt_tokens": sum((c["usage"] or {}).get("prompt_tokens", 0) for c in calls),
                           "completion_tokens": sum((c["usage"] or {}).get("completion_tokens", 0) for c in calls)}
        print(task["task_id"], status, per_episode[ep], flush=True)

    rows, summary = score_records(records, {t["task_id"]: t for t in tasks}, per_episode)
    seconds = [c["seconds"] for c in model.calls]
    usage = [c["usage"] or {} for c in model.calls]
    summary.update({
        "model": args.model, "memory": args.memory, "calls": len(model.calls),
        "settings": {"temperature": args.temperature, "max_tokens": args.max_tokens, "max_turns": args.max_turns,
                     "memory": args.memory, "memory_k": args.memory_k if args.memory else None,
                     "procedure": args.procedure,
                     "system_sha": hashlib.sha256(system.encode()).hexdigest()[:16],
                     "tasks_sha": hashlib.sha256(Path(args.tasks).read_bytes()).hexdigest()[:16]},
        "seconds_total": round(sum(seconds), 1),
        "prompt_tokens_total": sum(u.get("prompt_tokens", 0) for u in usage),
        "completion_tokens_total": sum(u.get("completion_tokens", 0) for u in usage),
    })
    (out / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "calls.json").write_text(json.dumps(model.calls) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
