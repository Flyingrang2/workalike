"""The browser demo's two new tickets, worked by Muse Glimmer with the LoRA and no rules in its prompt, on your GPU.
The page shows these recorded runs beside the live ones, because a browser visitor has no GPU to run the LoRA on.

    python -m demo.lora_run --out runs/demo-lora --rounds 5     # with scripts/serve.sh running and the adapter loaded
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from demo.server import load_tickets
from demo.work import summary
from testbed.run24 import run_one
from workalike.model_client import ChatModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="p-checked")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tickets = load_tickets()
    with open(out / "results.jsonl", "w") as f:
        for round_no in range(1, a.rounds + 1):
            for kind, task in tickets.items():
                make = lambda: ChatModel(a.base_url, a.model, temperature=0, max_tokens=16000)  # noqa: E731
                row = run_one(copy.deepcopy(task), "", make, live=True)
                done = summary(row, task)
                print(kind, round_no, done["outcome"], done["off"], done["seconds"], flush=True)
                f.write(
                    json.dumps(
                        {
                            "kind": kind,
                            "round": round_no,
                            **done,
                            "tools": [c["name"] for c in row["calls"]],
                            "first": row.get("first"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                f.flush()


if __name__ == "__main__":
    main()
