"""Propose rule cards from one worker's records with a served model, the first step of the product loop.

    python -m testbed.propose_cards --records data/v2/records/c-train.jsonl --env quote-desk/2 \
        --shuffle 11:curve:1 --limit 60 --model muse-glimmer --out runs/v2-cards/c60-s1.glimmer.json

The records are the ones extract_procedure reads (same order with the same --shuffle and --limit), so the cards
and the written procedure come from the same work. --json-schema also asks the server for structured output;
the cards are parsed the same way either way.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from workalike.envs import BY_VERSION, V1
from workalike.filter import clean_episode, episode_meta
from workalike.model_client import ChatModel
from workalike.record import group_episodes, read_events
from workalike.rules import CardError, merge, propose

SCHEMA = {
    "type": "object",
    "required": ["cards"],
    "properties": {
        "cards": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["situation", "action", "unknowns"],
                "properties": {
                    "situation": {"type": "string"},
                    "action": {"type": "string"},
                    "evidence": {"type": "array", "items": {"type": "integer"}},
                    "from": {"type": "array", "items": {"type": "integer"}},
                    "unknowns": {"type": "array", "items": {"type": "string"}},
                },
            },
        }
    },
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", required=True)
    ap.add_argument("--env", default=V1.version, choices=sorted(BY_VERSION))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--shuffle", help="the same key extract_procedure and gen_v2 use (e.g. 11:curve:1)")
    ap.add_argument("--batch", type=int, default=8, help="records per prompt")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--json-schema", action="store_true", help="ask the server for structured output too")
    ap.add_argument("--remerge", help="an earlier output: keep its batches' cards and only merge them again")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    env = BY_VERSION[args.env]
    records = [(episode_meta(e)["ep"], clean_episode(e)) for e in group_episodes(read_events(args.records)).values()]
    if args.shuffle:
        random.Random(args.shuffle).shuffle(records)
    records = records[: args.limit]

    extra = (
        {"response_format": {"type": "json_schema", "json_schema": {"name": "cards", "schema": SCHEMA}}}
        if args.json_schema
        else {}
    )
    calls = []

    def chat(messages):
        model = ChatModel(args.base_url, args.model, temperature=0, max_tokens=args.max_tokens, **extra)
        text = (model(messages, None).get("content") or "").strip()
        calls.extend(model.calls)
        if not text:
            raise CardError(f"empty reply (finish reason {model.calls[-1].get('finish_reason')})")
        return text

    if args.remerge:
        before = json.loads(Path(args.remerge).read_text())
        result = {
            "cards": merge(before["batches"], records, chat),
            "batches": before["batches"],
            "failed": before.get("failed", []),
            "remerged_from": args.remerge,
        }
    else:
        result = propose(records, env, chat, args.batch)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "records": args.records,
                "records_sha": hashlib.sha256(Path(args.records).read_bytes()).hexdigest()[:16],
                "episodes": len(records),
                "shuffle": args.shuffle,
                "batch": args.batch,
                "writer": args.model,
                "json_schema": args.json_schema,
                "seconds": round(sum(c["seconds"] for c in calls), 1),
                "completion_tokens": sum((c["usage"] or {}).get("completion_tokens", 0) for c in calls),
                **result,
            },
            ensure_ascii=False,
            indent=1,
        )
        + "\n"
    )
    print(f"{out}: {len(result['cards'])} cards from {len(records)} records by {args.model}")


if __name__ == "__main__":
    main()
