"""Generate the quote-desk/2 data: tickets, worker C's records, and C's training files.

    python -m island.gen_v2 --out data/v2 --seed 11

Writes
  tasks/{train,dev,eval,dev2}.jsonl  tickets from island.tasks2 (eval is generated and hashed, never worked here)
  records/c-{train,dev,dev2}.jsonl   worker C's raw records, through the Desk and Recorder
  sft/c150-s1-train.jsonl        C's 150 balanced rows (train150), shuffled once
  sft/c-s{k}-train.jsonl         C's 60 training rows in the order seed k shuffles them, so --limit N takes
                                 a different first N per seed (the C data-size curve: 20, 40, 60, seeds 1-3)
  manifest.json                  counts, C's scores, file hashes
C's records check the path and the amount of data a way of working needs. They are never human results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from workalike.desk import Desk
from workalike.envs import V2
from workalike.record import Recorder, SimClock
from workalike.sft import sft_row, write_jsonl

from . import worker_c
from .gen_h1 import episode_id, load_episodes, sha256
from .score2 import compare, score_records
from .tasks2 import FIXED_SHA256, SPLITS, make_tasks

CURVE_SEEDS = (1, 2, 3)


def run_c(task: dict, actor: str, recorder: Recorder, seed: int) -> None:
    recorder.clock = SimClock(f"{task['view']['today']}T09:00:00")
    recorder.start(episode_id(actor, task["task_id"]), task["view"], task_id=task["task_id"], actor=actor, seed=seed,
                   env=V2.version)
    desk = Desk(task["hidden"]["state"], recorder, env=V2)
    recorder.final(worker_c.work(desk, task, random.Random(f"{seed}:{task['task_id']}:c")))
    recorder.end()


def build(out, seed: int = 11) -> dict:
    out = Path(out)
    tasks = {split: make_tasks(seed, split) for split in SPLITS}
    actor = "w-" + hashlib.sha256(f"{seed}:c".encode()).hexdigest()[:6]
    for split, ts in tasks.items():
        write_jsonl(ts, out / "tasks" / f"{split}.jsonl")
        if seed == 11 and sha256(out / "tasks" / f"{split}.jsonl") != FIXED_SHA256[split]:
            raise SystemExit(f"{split} tickets differ from the fixed set; the sets do not change once made")
    summary = {}
    for split in ("train", "dev", "dev2", "train150"):
        path = out / "records" / f"c-{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
        recorder = Recorder(path)
        for task in tasks[split]:
            run_c(task, actor, recorder, seed)
        rows, s = score_records(path, {t["task_id"]: t for t in tasks[split]})
        summary[f"c/{split}"] = {**s, "same_way_as_itself": compare(rows, rows)["same_way"]}

    rows = [sft_row(ep, V2) for _, ep in load_episodes(out / "records" / "c-train.jsonl")]
    for k in CURVE_SEEDS:
        order = rows[:]
        random.Random(f"{seed}:curve:{k}").shuffle(order)
        write_jsonl(order, out / "sft" / f"c-s{k}-train.jsonl")

    rows150 = [sft_row(ep, V2) for _, ep in load_episodes(out / "records" / "c-train150.jsonl")]
    random.Random(f"{seed}:c150:1").shuffle(rows150)
    write_jsonl(rows150, out / "sft" / "c150-s1-train.jsonl")  # a reviewer's experiment 0: all 150 rows, 3 epochs

    files = sorted(p for p in out.rglob("*.jsonl") if "tok" not in p.name)
    manifest = {"env_version": V2.version, "seed": seed, "actor": actor, "curve_seeds": list(CURVE_SEEDS),
                "families": {split: dict(Counter(t["family"] for t in ts)) for split, ts in tasks.items()},
                "summary": summary, "files": {str(p.relative_to(out)): sha256(p) for p in files}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="data/v2")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    m = build(args.out, args.seed)
    print(json.dumps(m["summary"], indent=1))
    for name, h in m["files"].items():
        print(f"{h[:16]}  {name}")


if __name__ == "__main__":
    main()
