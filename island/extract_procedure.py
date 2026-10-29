"""Write a procedure from one worker's records: the "no fine-tuning" baseline built from the same data.

    python -m island.extract_procedure --records data/v2/records/c-train.jsonl --env quote-desk/2 \
        --model muse-glimmer --base-url http://localhost:8000/v1 --out baselines/c-procedure.glimmer.txt
    python -m island.extract_procedure ... --command "<cmd: prompt on stdin, text on stdout>" --out ...

The model reads the records as the worker left them (the ticket, then every call and result, in order)
and writes how this worker handles tickets, so that someone else could work the same way without the
records. Records that do not fit one prompt are read in chunks of --chunk episodes; each chunk gives
notes, and a last call merges them. The same prompts serve every model, so Glimmer's and another
model's procedures differ only in who wrote them. The procedure then goes into run_agent --procedure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shlex
import subprocess
from pathlib import Path

from workalike.envs import BY_VERSION, V1
from workalike.filter import clean_episode
from workalike.model_client import ChatModel
from workalike.record import group_episodes, read_events
from workalike.serialize import canonical_json, order_args

READ = ("Below are records of one worker at the quote desk of a small print shop: each ticket as the worker saw "
        "it, then every action they took and what came back, in order. Read them for how this worker handles "
        "tickets: what they check first, when they ask the customer or the owner before quoting, how they handle "
        "short stock, prices and discounts, reorders and follow-ups, and what they leave out. Write notes on the "
        "worker's way of working, with the conditions under which it changes. Facts about these particular "
        "tickets are not needed.")
MERGE = ("Below are notes on one worker at the quote desk of a small print shop, taken from several batches of "
         "their records. Merge them into one procedure written to the next person at this desk, so they handle "
         "tickets the way this worker does, including when to do something and when not to. Keep it to what the "
         "notes support.")
WRITE = ("Write the procedure itself, addressed to the next person at this desk, so they handle tickets the way "
         "this worker does, including when to do something and when not to. Keep it to what the records support.")


def render(ep: dict, env) -> str:
    lines = [env.render_view(ep["view"]), "Actions:"]
    for turn in ep["turns"]:
        for c in turn["calls"]:
            lines.append(f"- {c['name']} {canonical_json(order_args(c['name'], c['args'], env.spec_by_name))}"
                         f" -> {canonical_json(c['result'])}")
    if ep.get("final"):
        lines.append(f"Final: {ep['final']}")
    return "\n".join(lines)


def extract(episodes: list[dict], env, ask, chunk: int) -> tuple[str, list[str]]:
    """ask(system, user) -> text. Returns the procedure and the chunk notes (empty when one call sufficed)."""
    texts = [f"Ticket {i}:\n{render(ep, env)}" for i, ep in enumerate(episodes, 1)]
    if len(texts) <= chunk:
        return ask(READ + " " + WRITE, "\n\n".join(texts)), []
    notes = [ask(READ, "\n\n".join(texts[i:i + chunk])) for i in range(0, len(texts), chunk)]
    return ask(MERGE, "\n\n".join(f"Notes {i}:\n{n}" for i, n in enumerate(notes, 1))), notes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", required=True)
    ap.add_argument("--env", default=V1.version, choices=sorted(BY_VERSION))
    ap.add_argument("--limit", type=int, help="use the first N episodes, as a LoRA trained on N would")
    ap.add_argument("--shuffle", help="shuffle the episodes with this random key first, the way gen_v2 orders a "
                                      "training file (e.g. 11:curve:1), so --limit takes the same episodes")
    ap.add_argument("--chunk", type=int, default=12, help="episodes per prompt")
    ap.add_argument("--model", help="served model name (with --base-url)")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--command", help="instead of a served model: system + user as JSON on stdin, text on stdout")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    env = BY_VERSION[args.env]
    episodes = [clean_episode(e) for e in group_episodes(read_events(args.records)).values()]
    if args.shuffle:
        random.Random(args.shuffle).shuffle(episodes)
    episodes = episodes[: args.limit]

    if args.command:
        def ask(system, user):
            return subprocess.run(shlex.split(args.command), input=json.dumps({"system": system, "user": user}),
                                  capture_output=True, text=True, check=True, timeout=900).stdout.strip()
        who = args.command
    else:
        model = ChatModel(args.base_url, args.model, temperature=0, max_tokens=12000)

        def ask(system, user):
            return (model([{"role": "system", "content": system}, {"role": "user", "content": user}], None)
                    .get("content") or "").strip()
        who = args.model
    procedure, notes = extract(episodes, env, ask, args.chunk)
    out = Path(args.out)
    out.write_text(procedure + "\n")
    out.with_suffix(".json").write_text(json.dumps({
        "records": args.records, "records_sha": hashlib.sha256(Path(args.records).read_bytes()).hexdigest()[:16],
        "episodes": len(episodes), "shuffle": args.shuffle, "chunk": args.chunk, "writer": who, "notes": notes,
        "words": len(procedure.split())}, ensure_ascii=False, indent=1) + "\n")
    print(f"{out}: {len(procedure.split())} words from {len(episodes)} episodes by {who}")


if __name__ == "__main__":
    main()
