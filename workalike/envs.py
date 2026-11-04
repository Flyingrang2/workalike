"""Desk versions. Each bundles the tools, their handlers, the system prompt and how a ticket is shown.

quote-desk/1 is the desk the first experiments used and stays the default everywhere, so its
records, training files and adapters keep their hashes. quote-desk/2 adds split shipments,
the floor price, the owner and the customer's answer (workalike/desk2.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from . import desk, desk2, serialize


@dataclass(frozen=True)
class Env:
    version: str
    tool_specs: list
    handlers: dict
    system_prompt: str
    render_view: Callable[[dict], str]

    @property
    def spec_by_name(self) -> dict:
        return {s["function"]["name"]: s["function"] for s in self.tool_specs}


V1 = Env(desk.ENV_VERSION, desk.TOOL_SPECS, desk.HANDLERS, serialize.SYSTEM_PROMPT, serialize.render_view)
V2 = Env(desk2.ENV_VERSION, desk2.TOOL_SPECS, desk2.HANDLERS, desk2.SYSTEM_PROMPT, desk2.render_view)
BY_VERSION = {e.version: e for e in (V1, V2)}
