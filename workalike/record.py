"""Append-only JSONL recorder: what was on screen and what was done, one event per line.

Every event carries a visibility mark:
  seen     what the worker saw or did; the only events that may reach training data
  private  notes and thoughts; never leave the record
  meta     bookkeeping (who, which task, which seed); never reach the model
Each line is written and flushed on its own, so a crash loses at most one event.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from .desk import ENV_VERSION


class SimClock:
    """Deterministic clock for scripted runs: a fixed step per event."""

    def __init__(self, start: str, step_seconds: int = 11):
        self.now = datetime.fromisoformat(start)
        self.step = timedelta(seconds=step_seconds)

    def __call__(self) -> str:
        self.now += self.step
        return self.now.isoformat(timespec="seconds")


def wall_clock() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Recorder:
    def __init__(self, path, clock=wall_clock):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self.episode = None
        self.seq = 0

    def _write(self, type_, vis, **fields):
        event = {"ep": self.episode, "seq": self.seq, "t": self.clock(), "type": type_, "vis": vis, **fields}
        self.seq += 1
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
            f.flush()

    def start(
        self,
        episode: str,
        view: dict,
        task_id: str | None = None,
        actor: str | None = None,
        seed=None,
        env: str = ENV_VERSION,
        **meta,
    ):
        self.episode, self.seq = episode, 0
        self._write("episode_start", "meta", task_id=task_id, actor=actor, env=env, seed=seed, **meta)
        self._write("observation", "seen", view=view)

    def event(self, type_: str, vis: str, **fields):
        """Any other event, e.g. a screenshot or a click from the web entrance."""
        self._write(type_, vis, **fields)

    def tool_call(self, call_id: str, name: str, args, turn: int):
        self._write("tool_call", "seen", call_id=call_id, name=name, args=args, turn=turn)

    def tool_result(self, call_id: str, name: str, result: dict):
        self._write("tool_result", "seen", call_id=call_id, name=name, result=result)

    def note(self, text: str, source: str = "note"):
        self._write("note", "private", text=text, source=source)

    def final(self, text: str):
        self._write("final", "seen", text=text)

    def end(self, status: str = "done"):
        self._write("episode_end", "meta", status=status)
        self.episode = None


def read_events(path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def group_episodes(events) -> dict[str, list[dict]]:
    episodes: dict[str, list[dict]] = {}
    for event in events:
        episodes.setdefault(event["ep"], []).append(event)
    return episodes


class RecordSwitch:
    """A recorder the person can switch off. The switch applies from the next episode, so no episode is ever
    half-recorded: start() decides for the whole episode, and every later call follows that decision."""

    def __init__(self, recorder: Recorder, on: bool = True):
        self.recorder, self.on, self.live = recorder, on, False

    def start(self, *args, **kwargs):
        self.live = self.on
        if self.live:
            self.recorder.start(*args, **kwargs)

    def __getattr__(self, name):
        method = getattr(self.recorder, name)
        if not callable(method):
            return method
        return lambda *a, **k: method(*a, **k) if self.live else None
