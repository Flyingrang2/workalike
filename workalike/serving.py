"""The served model on the GPU box, for workalike.autolearn: start it with an adapter, stop it, and work the fixed check."""
from __future__ import annotations

import os
import subprocess
import time
import urllib.request
from pathlib import Path


class Server:
    def __init__(self, base: str, log: str = "runs/autolearn-server.log"):
        self.base, self.log, self.proc = base, log, None

    def stop(self):
        if self.proc:
            self.proc.terminate()
            self.proc.wait()
            self.proc = None

    def start(self, adapters: dict[str, str]):
        self.stop()
        env = {**os.environ, "M": self.base, "QUANT": os.environ.get("QUANT", "fp8_per_channel"),
               "ADAPTERS": " ".join(f"{k}={v}" for k, v in adapters.items())}
        Path(self.log).parent.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(["bash", "scripts/serve.sh", self.log], env=env)
        for _ in range(180):
            try:
                urllib.request.urlopen("http://localhost:8000/v1/models", timeout=5)
                return
            except OSError:
                time.sleep(10)
        raise RuntimeError("the model server did not come up")


def fixed_check(server: Server, tasks_path: str, answers_path: str, base_url: str):
    """check(name, adapter path): serve that adapter and work the fixed tickets through the product's own path."""
    from island.grade24 import grade, tally
    from island.run24 import run_one

    from .envs import V2
    from .model_client import ChatModel
    from .sft import read_jsonl

    def check(name: str, adapter: str) -> dict:
        server.start({name: adapter})
        tasks = read_jsonl(tasks_path)
        answers = {a["task_id"]: a for a in read_jsonl(answers_path)}
        make = lambda: ChatModel(base_url, name, temperature=0, max_tokens=16000)  # noqa: E731
        rows = [grade(answers[t["task_id"]], run_one(t, "", make, base_system=V2.system_prompt)) for t in tasks]
        return tally(rows)
    return check
