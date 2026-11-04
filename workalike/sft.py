"""Clean episodes to supervised fine-tuning rows ({"messages", "tools"} JSONL).

Rows carry no task id, worker label or split; that bookkeeping lives in the
run manifest next to the files.
"""

from __future__ import annotations

import json
from pathlib import Path

from .desk import TOOL_SPECS
from .serialize import to_messages


def sft_row(episode: dict, env=None) -> dict:
    """env is a desk version from workalike.envs; None is quote-desk/1."""
    return {"messages": to_messages(episode, env), "tools": env.tool_specs if env else TOOL_SPECS}


def write_jsonl(rows, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
