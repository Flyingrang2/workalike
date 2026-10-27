"""Inference loop. The model is any function (messages, tools) -> reply.

A reply is {"tool_calls": [{"name": ..., "arguments": {...}}]} or {"content": "..."},
optionally with "reasoning". Messages are built only with serialize.py, the same
code that builds training examples, and every call goes through the Desk, so the
agent leaves the same record a person would. Reasoning is kept as a private note
and is not carried into later turns, matching how training examples look.
"""
from __future__ import annotations

from .desk import TOOL_SPECS, Desk
from .serialize import SYSTEM_PROMPT, final_message, head_messages, turn_messages


def run(desk: Desk, view: dict, model, max_turns: int = 24, system: str = SYSTEM_PROMPT) -> tuple[list[dict], str]:
    """Work one ticket. Returns (messages, status); the caller starts and ends the record."""
    messages = head_messages(view, system)
    for _ in range(max_turns):
        reply = model(messages, TOOL_SPECS)
        if reply.get("reasoning") and desk.recorder:
            desk.recorder.note(reply["reasoning"], source="reasoning")
        calls = reply.get("tool_calls") or []
        if not calls:
            text = (reply.get("content") or "").strip()
            if desk.recorder:
                desk.recorder.final(text)
            messages.append(final_message(text))
            return messages, "done"
        done = desk.call_many([(c["name"], c["arguments"]) for c in calls])
        messages += turn_messages(done)
    return messages, "max_turns"
