"""Minimal OpenAI-compatible chat client (stdlib only), for a local vLLM or llama.cpp server.

    model = ChatModel("http://localhost:8000/v1", "muse-glimmer")
    reply = model(messages, tools)   # {"tool_calls": [...], "content": ..., "reasoning": ...}

With vLLM's --enable-lora, an adapter is selected by passing its name as the model.
Every call's token usage and wall time are kept in .calls for later comparison.
"""
from __future__ import annotations

import json
import time
import urllib.request

from .serialize import to_wire


class ChatModel:
    def __init__(self, base_url: str, model: str, api_key: str = "none", timeout: float = 900, **params):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.params = params
        self.calls: list[dict] = []

    def __call__(self, messages: list[dict], tools: list[dict]) -> dict:
        body = {"model": self.model, "messages": to_wire(messages), **self.params}
        if tools:
            body.update(tools=tools, tool_choice="auto")
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.api_key}"})
        start = time.monotonic()
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read())
        msg = data["choices"][0]["message"]
        self.calls.append({"seconds": round(time.monotonic() - start, 3), "usage": data.get("usage"),
                           "finish_reason": data["choices"][0].get("finish_reason")})
        calls = []
        for tc in msg.get("tool_calls") or []:
            raw = tc["function"].get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
            except json.JSONDecodeError:
                args = {"_unparsed": raw}
            calls.append({"name": tc["function"]["name"], "arguments": args})
        return {"tool_calls": calls, "content": msg.get("content"),
                "reasoning": msg.get("reasoning") or msg.get("reasoning_content")}
