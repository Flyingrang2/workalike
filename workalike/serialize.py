"""The one place that turns episodes into chat messages.

Training examples and the inference loop both build messages with these
functions, so the model is trained on exactly the prompts it will later see.
Tool-call arguments stay dicts because the Glimmer chat template requires a
mapping; to_wire() converts them to JSON strings for OpenAI-compatible servers.
"""
from __future__ import annotations

import json

from .desk import SPEC_BY_NAME

SYSTEM_PROMPT = (
    "You handle tickets at the quote desk of a small custom print shop. "
    "Use the tools to look things up, prepare quotes and reply to the customer. "
    "Every reply and quote is held for the owner's approval before it reaches the customer. "
    "If an item is short, a later ship date for the same item and an in-stock substitute that meets the date "
    "are both acceptable; the customer confirms before ordering. "
    "When you have finished with the ticket, answer with: Done."
)


def canonical_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def order_args(name: str, args):
    """Arguments in schema order, so every source writes them the same way."""
    spec = SPEC_BY_NAME.get(name)
    if spec is None or not isinstance(args, dict):
        return args
    props = list(spec["parameters"]["properties"])
    known = [k for k in props if k in args]
    return {k: args[k] for k in known + sorted(k for k in args if k not in props)}


def render_view(view: dict) -> str:
    parts = [f"Today is {view['today']}.", f"Ticket {view['ticket_id']} from {view['from']}."]
    if view.get("thread"):
        lines = [f"- {m['date']}, {m['who']}: {m['text']}" for m in view["thread"]]
        parts.append("Earlier in this thread:\n" + "\n".join(lines))
    parts.append("New message:\n" + view["body"])
    return "\n\n".join(parts)


def head_messages(view: dict, system: str = SYSTEM_PROMPT) -> list[dict]:
    return [{"role": "system", "content": system}, {"role": "user", "content": render_view(view)}]


def turn_messages(calls: list[dict]) -> list[dict]:
    """One assistant turn of tool calls followed by one tool message per call."""
    assistant = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": c["id"], "type": "function",
             "function": {"name": c["name"], "arguments": order_args(c["name"], c["args"])}}
            for c in calls
        ],
    }
    tools = [{"role": "tool", "tool_call_id": c["id"], "name": c["name"], "content": canonical_json(c["result"])}
             for c in calls]
    return [assistant, *tools]


def final_message(text: str) -> dict:
    return {"role": "assistant", "content": text}


def to_messages(episode: dict) -> list[dict]:
    messages = head_messages(episode["view"])
    for turn in episode["turns"]:
        messages += turn_messages(turn["calls"])
    if episode.get("final") is not None:
        messages.append(final_message(episode["final"]))
    return messages


def to_wire(messages: list[dict]) -> list[dict]:
    """OpenAI wire format: tool-call arguments as JSON strings."""
    out = []
    for m in messages:
        if m.get("tool_calls"):
            m = {**m, "content": None, "tool_calls": [
                {**tc, "function": {**tc["function"], "arguments": json.dumps(tc["function"]["arguments"], ensure_ascii=False)}}
                for tc in m["tool_calls"]]}
        out.append(m)
    return out
