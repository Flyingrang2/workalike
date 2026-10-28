"""Ask a served model the diagnostic probes: the same short cases a person answered, one question each.

    python -m island.probe_run --cases baselines/probe-cases.en.json --model muse-glimmer --out runs/v2-probe/base

Each case is shown as the person saw it (who, date, earlier thread, the new email, what the shop's system
shows) under the same rules, and the model answers NEXT (what it does right away) and MESSAGE (what it
would send the customer). No tools: this is a reading of judgement, not a worked ticket. Answers are kept
as written, with the raw reply and tokens, for coding next to the person's answers.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from workalike.model_client import ChatModel


def case_text(c: dict) -> str:
    parts = [f"Today is {c['today']}.", f"Email from {c['who']}."]
    if c["thread"]:
        parts.append("Earlier in this thread:\n" + "\n".join(f"- {m['who']}: {m['text']}" for m in c["thread"]))
    parts.append("New email:\n" + c["body"])
    parts.append("What the shop's system shows:\n" + "\n".join(
        f"- {'(!) ' if f['warn'] else ''}{f['text']}" for f in c["facts"]))
    return "\n\n".join(parts)


def split_answer(text: str) -> dict:
    m = re.search(r"NEXT:\s*(.*?)(?:\n\s*MESSAGE:\s*(.*))?\s*$", text, re.S)
    if not m:
        return {"now": text.strip(), "blocked": None}
    msg = (m.group(2) or "").strip()
    return {"now": m.group(1).strip(), "blocked": None if msg.lower() in ("", "none", "(none)") else msg}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cases", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data = json.loads(Path(args.cases).read_text())
    model = ChatModel(args.base_url, args.model, max_tokens=args.max_tokens, temperature=args.temperature)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    answers = {}
    for c in data["cases"]:
        messages = [{"role": "system", "content": data["rules"]},
                    {"role": "user", "content": case_text(c) + "\n\n" + data["question"]}]
        reply = model(messages, None)
        text = reply.get("content") or ""
        answers[c["id"]] = {**split_answer(text), "raw": text, "reasoning": reply.get("reasoning"),
                            "usage": model.calls[-1]["usage"]}
        print(c["id"], json.dumps(answers[c["id"]]["now"])[:200], flush=True)
    (out / "answers.json").write_text(json.dumps({"model": args.model, "cases": args.cases, "answers": answers},
                                                 ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
