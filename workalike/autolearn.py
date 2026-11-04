"""Keep learning from what a person does: their records become a LoRA adapter, with no rules written and no labels.

    python -m workalike.autolearn watch --records runs/me/records.jsonl --versions versions/me --base <model dir> --every 10
    python -m workalike.autolearn once  --records ... --versions ... --base ...

Runs on the GPU box, beside the served model. Each round: build a version from the person's finished tickets, repeat
the rarer ways of working (workalike.rebalance), stop the server, train, serve the new adapter, and work a fixed check.
The adapter becomes active only if it does at least as well as the active one and adds no dangerous draft; otherwise the
active one stays. Every step goes to <versions>/status.json, which the agent pane shows. Only the person's records are
read: the agent's own drafts are kept in another file and are never learned from as if a person had done them.
Each version keeps which records it came from, how many rows were repeats, its training time and its check.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from .envs import V2
from .learn import build, episodes
from .rebalance import rebalance
from .record import wall_clock
from .sft import read_jsonl, write_jsonl


class Status:
    def __init__(self, versions):
        self.path = Path(versions) / "status.json"
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {"steps": [], "active": None}

    def step(self, what: str, **extra):
        self.data.update(state=what, updated=wall_clock())
        self.data["steps"] = (self.data["steps"] + [{"t": wall_clock(), "what": what, **extra}])[-40:]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1) + "\n")
        tmp.replace(self.path)


def active_adapter(versions) -> dict | None:
    """What the agent pane should work with: {"version", "name", "path", "check"}, or None before the first round."""
    path = Path(versions) / "status.json"
    return json.loads(path.read_text()).get("active") if path.exists() else None


def finished(records) -> int:
    return sum(e["status"] == "done" for e in episodes(records)) if Path(records).exists() else 0


def better(new: dict, old: dict | None) -> bool:
    """At least as good as the active adapter, and no more dangerous drafts. The first one must have none."""
    if old is None:
        return new["dangerous"] == 0 and new["failed"] < new["tickets"]
    return new["good"] >= old["good"] and new["dangerous"] <= old["dangerous"]


def learn_once(records, versions, shell, check, target: int = 150) -> dict:
    """shell(step name, command) runs one training or serving step; check(version name, adapter path) works the fixed
    check on the adapter and returns its tally. Both are passed in so the round can run with fakes."""
    versions = Path(versions)
    status = Status(versions)
    status.step("데이터 만드는 중", records=finished(records))
    manifest = build(records, versions, env=V2)
    out = versions / manifest["version"]
    train = next((out / "sft").glob("*-train.jsonl"))
    rows, balance = rebalance(read_jsonl(train), target)
    write_jsonl(rows, train)
    manifest.update(rows=len(rows), unique_rows=balance["unique_rows"], groups=balance["by_group"])
    status.step("학습 중", version=manifest["version"], tickets=balance["unique_rows"], rows=len(rows))
    start = time.monotonic()
    shell("train", {"data": str(out), "adapter": str(out / "adapter"), "tok": str(train.with_suffix(".tok.jsonl"))})
    manifest["train_seconds"] = round(time.monotonic() - start)
    status.step("시험 중", version=manifest["version"], train_seconds=manifest["train_seconds"])
    tally = check(manifest["name"], str(out / "adapter"))
    manifest["check"] = tally
    old = status.data.get("active")
    won = better(tally, old and old["check"])
    manifest["activated"] = won
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    if won:
        status.data["active"] = {
            "version": manifest["version"],
            "name": manifest["name"],
            "path": str(out / "adapter"),
            "check": tally,
            "tickets": balance["unique_rows"],
            "rows": len(rows),
            "train_seconds": manifest["train_seconds"],
        }
    status.data["learned_from"] = finished(records)
    status.step(
        f"{manifest['version']} 켰습니다" if won else f"{manifest['version']}는 켜지 않았습니다(전보다 나쁨)",
        check=tally,
    )
    return manifest


def gpu_shell(base: str, python: str, server_stop):
    """The real steps on the box: the server gives the card back, then the tokens and the adapter are made."""

    def run(step, paths):
        server_stop()
        subprocess.run([python, "scripts/render_sft.py", "--tokenizer", base, "--data", paths["data"]], check=True)
        subprocess.run(
            [
                python,
                "scripts/train_lora.py",
                "--base",
                base,
                "--data",
                paths["tok"],
                "--seed",
                "1",
                "--out",
                paths["adapter"],
            ],
            check=True,
        )

    return run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=("once", "watch"))
    ap.add_argument("--records", required=True)
    ap.add_argument("--versions", required=True)
    ap.add_argument("--base", required=True, help="model dir")
    ap.add_argument("--train-python", default=".venv-train/bin/python")
    ap.add_argument("--every", type=int, default=10, help="learn again after this many new finished tickets")
    ap.add_argument("--target", type=int, default=150, help="rows after repeating the rarer ways of working")
    ap.add_argument("--check-tasks", default="baselines/test24/dev.jsonl")
    ap.add_argument("--check-answers", default="baselines/test24/dev-answers.jsonl")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    args = ap.parse_args()
    from .serving import Server, fixed_check  # the GPU side: imported only here

    server = Server(args.base)
    shell = gpu_shell(args.base, args.train_python, server.stop)
    check = fixed_check(server, args.check_tasks, args.check_answers, args.base_url)

    def round_once():
        return learn_once(args.records, args.versions, shell, check, args.target)

    if args.mode == "once":
        print(json.dumps(round_once()["check"]))
        return
    status = Status(args.versions)
    while True:
        new = finished(args.records) - status.data.get("learned_from", 0)
        if new >= args.every:
            round_once()
            status = Status(args.versions)
        else:
            status.data.update(waiting=new, every=args.every)
            status.step(f"기록 중, 다음 학습까지 {args.every - new}건")
            status.data["steps"] = status.data["steps"][:-1]  # waiting is a state, not a step worth keeping
        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
