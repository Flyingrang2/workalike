"""Screen entrance check: can a model finish quotes looking only at screenshots of the web desk?

    uv run --no-project --with playwright python -m testbed.screen_agent \
        --tasks data/h1/tasks/dev.jsonl --one-per-family --limit 5 \
        --base-url http://localhost:8000/v1 --model muse-glimmer --out runs/screen-base

Needs Chromium for Playwright (python -m playwright install chromium). The model
gets the computer.computer and computer.stop tools from Meta's metacua harness
(coordinates are integers 0 to 1000) and sees nothing but the browser viewport.
The web desk records every form submission as a Desk call, so the run is scored
with the same scorer as the tool entrance. Screenshots and UI actions are kept in
screen.jsonl beside the desk records.
"""

from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path

from workalike.model_client import ChatModel
from workalike.record import Recorder
from workalike.serialize import turn_messages
from workalike.sft import read_jsonl
from workalike.web import WebDesk, serve_in_background, sessions_for

from .serve_web import write_scores

WIDTH, HEIGHT = 1280, 800
KEEP_IMAGES = 3

_COORD = {
    "anyOf": [{"items": {"type": "integer"}, "type": "array"}, {"type": "null"}],
    "default": None,
    "description": "[x, y] relative coordinates (integers in [0, 1000]) for mouse actions. "
    "Required for click, move, and drag actions.",
}
SCREEN_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "computer.computer",
            "description": "Control the computer via mouse, keyboard, and screen actions.\n\nThis matches Anthropic's "
            "computer_20251124 tool interface, but uses relative coordinates (integers in [0, 1000]) "
            "instead of absolute pixels.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["action"],
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "The action to perform. One of: left_click, right_click, double_click, middle_click, "
                        "triple_click, left_press, left_click_drag, mouse_move, key, type, hold_key, release_key, "
                        "left_mouse_down, left_mouse_up, scroll, screenshot, wait",
                    },
                    "coordinate": _COORD,
                    "text": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "default": None,
                        "description": "Text input for keyboard actions. For 'key': key combo string, e.g. 'ctrl+c' or 'Return'. "
                        "For 'type': text to type character by character.",
                    },
                    "start_coordinate": {
                        **_COORD,
                        "description": "[x, y] relative starting coordinates for left_click_drag.",
                    },
                    "scroll_direction": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "default": None,
                        "description": "Scroll direction: 'up', 'down', 'left', or 'right'.",
                    },
                    "scroll_amount": {
                        "anyOf": [{"type": "integer"}, {"type": "null"}],
                        "default": None,
                        "description": "Number of scroll clicks.",
                    },
                    "duration": {
                        "anyOf": [{"type": "number"}, {"type": "null"}],
                        "default": None,
                        "description": "Duration in seconds for wait actions.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "computer.stop",
            "description": "Stop the session and submit your final answer.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["answer"],
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Brief description of what you accomplished, or why you cannot proceed safely.",
                    }
                },
            },
        },
    },
]

SCREEN_SYSTEM = """You are an AI assistant specialized in computer use.
You perceive the screen via screenshots and control a web browser using a fixed set of APIs.

# Available Tools
Use `computer.computer` for computer actions and `computer.stop` when the task is complete or you cannot proceed safely.

# Screen And Coordinates
Coordinates are integers from 0 to 1000 relative to the screenshot, with (0, 0) at the top-left. Always read coordinates off the screenshot rather than guessing.

# Task Execution Strategy
- Focus on ONE action at a time.
- After each action, a screenshot is automatically returned showing the result.
- `type` does not submit; click the form's button afterwards.
- Verify results on screen before stopping."""

GOAL = (
    "This browser shows the quote desk web app of a small custom print shop, open on one customer ticket. "
    "Handle the ticket using the forms on the page: look things up, prepare a quote and reply to the customer. "
    "Replies and quotes are held for the owner's approval. When you have finished with the ticket, click "
    "Finish ticket, then call computer.stop."
)

KEYS = {
    "return": "Enter",
    "enter": "Enter",
    "ctrl": "Control",
    "control": "Control",
    "cmd": "Meta",
    "super": "Meta",
    "meta": "Meta",
    "alt": "Alt",
    "option": "Alt",
    "shift": "Shift",
    "tab": "Tab",
    "esc": "Escape",
    "escape": "Escape",
    "backspace": "Backspace",
    "delete": "Delete",
    "space": "Space",
    "up": "ArrowUp",
    "down": "ArrowDown",
    "left": "ArrowLeft",
    "right": "ArrowRight",
    "pageup": "PageUp",
    "pagedown": "PageDown",
    "home": "Home",
    "end": "End",
}


def _xy(coord) -> tuple[float, float]:
    x, y = coord
    return min(max(x, 0), 1000) / 1000 * WIDTH, min(max(y, 0), 1000) / 1000 * HEIGHT


def execute(page, a: dict) -> str:
    """Run one computer.computer action in the browser. Returns a short status for the model."""
    act = a.get("action")
    clicks = {
        "left_click": ("left", 1),
        "right_click": ("right", 1),
        "middle_click": ("middle", 1),
        "double_click": ("left", 2),
        "triple_click": ("left", 3),
        "left_press": ("left", 1),
    }
    if act in clicks:
        if not a.get("coordinate"):
            return "error: coordinate is required"
        button, count = clicks[act]
        page.mouse.click(*_xy(a["coordinate"]), button=button, click_count=count)
    elif act == "mouse_move":
        page.mouse.move(*_xy(a["coordinate"]))
    elif act == "left_click_drag":
        page.mouse.move(*_xy(a["start_coordinate"]))
        page.mouse.down()
        page.mouse.move(*_xy(a["coordinate"]))
        page.mouse.up()
    elif act == "type":
        page.keyboard.type(a.get("text") or "")
    elif act == "key":
        combo = "+".join(KEYS.get(k.strip().lower(), k.strip()) for k in (a.get("text") or "").split("+"))
        page.keyboard.press(combo)
    elif act == "scroll":
        if a.get("coordinate"):
            page.mouse.move(*_xy(a["coordinate"]))
        step = 100 * (a.get("scroll_amount") or 3)
        dx, dy = {"up": (0, -step), "down": (0, step), "left": (-step, 0), "right": (step, 0)}.get(
            a.get("scroll_direction") or "down", (0, step)
        )
        page.mouse.wheel(dx, dy)
    elif act == "wait":
        page.wait_for_timeout(1000 * (a.get("duration") or 1))
    elif act != "screenshot":
        return f"error: unsupported action {act}"
    try:
        page.wait_for_timeout(300)  # let a form submission start navigating before checking the load state
        page.wait_for_load_state("load", timeout=5000)
    except Exception:
        pass
    return "ok"


def _image(png: bytes) -> dict:
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}


def _drop_old_images(messages: list[dict]) -> None:
    seen = 0
    for m in reversed(messages):
        if isinstance(m.get("content"), list):
            for i, part in enumerate(m["content"]):
                if part.get("type") == "image_url":
                    seen += 1
                    if seen > KEEP_IMAGES:
                        m["content"][i] = {"type": "text", "text": "[earlier screenshot removed]"}


def run_ticket(page, url: str, webdesk: WebDesk, model, screen: Recorder, shots: Path, max_steps: int) -> str:
    episode, task_id = webdesk.status()["episode"], webdesk.status()["task_id"]
    page.goto(url)
    screen.episode, screen.seq = episode, 0
    screen.event("episode_start", "meta", task_id=task_id, entrance="screen")

    def snap(step):
        try:
            png = page.screenshot(timeout=30000)
        except Exception:
            png = page.screenshot(timeout=60000)
        path = shots / f"{episode}-{step:03d}.png"
        path.write_bytes(png)
        screen.event("screenshot", "seen", path=str(path.name))
        return png

    messages = [
        {"role": "system", "content": SCREEN_SYSTEM},
        {"role": "user", "content": [{"type": "text", "text": GOAL}, _image(snap(0))]},
    ]
    for step in range(1, max_steps + 1):
        reply = model(messages, SCREEN_TOOLS)
        if reply.get("reasoning"):
            screen.note(reply["reasoning"], source="reasoning")
        calls = reply.get("tool_calls") or []
        if not calls or any(c["name"] == "computer.stop" for c in calls):
            return "stopped" if webdesk.status()["episode"] != episode else "stopped_without_finish"
        done = []
        for i, c in enumerate(calls):
            status = (
                execute(page, c["arguments"])
                if c["name"] == "computer.computer"
                else f"error: unknown tool {c['name']}"
            )
            screen.event("ui_action", "seen", action=c["arguments"], status=status)
            done.append({"id": f"s{step}-{i}", "name": c["name"], "args": c["arguments"], "result": {"status": status}})
            if webdesk.status()["episode"] != episode:
                return "finished"  # Finish ticket was clicked; the next ticket is already on screen
        messages += turn_messages(done)
        messages.append(
            {"role": "user", "content": [{"type": "text", "text": "Screenshot after the action."}, _image(snap(step))]}
        )
        _drop_old_images(messages)
    return "max_steps"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tasks", required=True, nargs="+")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--one-per-family", action="store_true", help="take the first ticket of each family")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--port", type=int, default=8766)
    args = ap.parse_args()
    from playwright.sync_api import sync_playwright

    tasks = [t for path in args.tasks for t in read_jsonl(path)]
    if args.one_per_family:
        tasks = [next(t for t in tasks if t["family"] == f) for f in dict.fromkeys(t["family"] for t in tasks)]
    tasks = tasks[: args.limit]
    out = Path(args.out)
    (out / "shots").mkdir(parents=True, exist_ok=True)
    (out / "records.jsonl").write_text("")
    (out / "screen.jsonl").write_text("")
    webdesk = WebDesk(sessions_for(tasks, f"screen:{args.model}"), Recorder(out / "records.jsonl"))
    server = serve_in_background(webdesk, port=args.port)
    url = f"http://127.0.0.1:{args.port}/"
    model = ChatModel(args.base_url, args.model, max_tokens=args.max_tokens)
    screen = Recorder(out / "screen.jsonl")
    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})
        while webdesk.status()["episode"]:
            task_id, start = webdesk.status()["task_id"], time.monotonic()
            try:
                status = run_ticket(page, url, webdesk, model, screen, out / "shots", args.max_steps)
            except Exception as e:
                status = f"error: {type(e).__name__}: {e}"
            if webdesk.status()["task_id"] == task_id:
                webdesk.finish(status=status, final=None)
            results.append({"task_id": task_id, "status": status, "seconds": round(time.monotonic() - start, 1)})
            print(results[-1], flush=True)
        browser.close()
    server.shutdown()
    summary = write_scores(out, tasks)
    summary.update({"model": args.model, "tickets": results, "model_calls": len(model.calls)})
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
