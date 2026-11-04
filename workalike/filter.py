"""Turn a raw episode into a clean one: only what the worker saw and what they did.

This is an allow-list. Anything that is not a seen observation, tool call, tool
result or final message is dropped, and an observation keeps only the fields a
worker has on screen at the start. Notes, thoughts, incident text, actor ids and
task metadata therefore never reach training data, whatever else a record holds.
Error results stay: a mistake followed by a correct recovery is worth learning.
"""

from __future__ import annotations

# Every field a worker has on screen. quote-desk/2 adds the discount authority and what was sent so far;
# leaving a screen field out makes training rows differ from what the model sees when it works.
VIEW_KEYS = ("today", "ticket_id", "from", "body", "thread", "authority", "sent_so_far")
THREAD_KEYS = ("date", "who", "text")


def clean_view(view: dict) -> dict:
    out = {k: view[k] for k in VIEW_KEYS if k in view}
    if "thread" in out:
        out["thread"] = [{k: m[k] for k in THREAD_KEYS if k in m} for m in out["thread"]]
    return out


def clean_episode(events: list[dict]) -> dict:
    view, final, turns, calls = None, None, [], {}
    for event in events:
        if event.get("vis") != "seen":
            continue
        kind = event["type"]
        if kind == "observation":
            view = clean_view(event["view"])
        elif kind == "tool_call":
            call = {"id": event["call_id"], "name": event["name"], "args": event["args"], "result": None}
            if turns and turns[-1]["turn"] == event["turn"]:
                turns[-1]["calls"].append(call)
            else:
                turns.append({"turn": event["turn"], "calls": [call]})
            calls[event["call_id"]] = call
        elif kind == "tool_result":
            calls[event["call_id"]]["result"] = event["result"]
        elif kind == "final":
            final = event["text"]
    if view is None:
        raise ValueError("episode has no observation")
    return {"view": view, "turns": [{"calls": t["calls"]} for t in turns], "final": final}


def episode_meta(events: list[dict]) -> dict:
    """Bookkeeping for joining records with tasks; never passed to serialization."""
    for event in events:
        if event["type"] == "episode_start":
            return {"ep": event["ep"], "task_id": event.get("task_id"), "actor": event.get("actor")}
    return {"ep": events[0]["ep"], "task_id": None, "actor": None}
