"""Generate the Hypothesis 1 dataset: paired A and B records on the same tickets.

    python -m island.gen_h1 --out data/h1 --seed 7

Writes
  tasks/{train,dev,eval_h1,eval_trap}.jsonl  view, request, hidden state and grader per task
                                       (dev is for smoke runs and tuning; the eval splits are kept for judging)
  records/{a,b}-{split}.jsonl            raw event records, written through the Desk and Recorder
  sft/{a,b,mixed}-train.jsonl            training rows from the filtered train records
  manifest.json                          which anonymous actor is which worker, counts, file hashes
Evaluation records are reference behaviour for scoring, never training data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from workalike.desk import ENV_VERSION, Desk
from workalike.filter import clean_episode, episode_meta
from workalike.record import Recorder, SimClock, group_episodes, read_events
from workalike.sft import sft_row, write_jsonl

from . import scripted
from .score import score_records, trace
from .tasks import EVAL_ONLY_SKUS, make_tasks

STYLES = ("a", "b")


def actor_id(seed: int, style: str) -> str:
    return "w-" + hashlib.sha256(f"{seed}:{style}".encode()).hexdigest()[:6]


def episode_id(actor: str, task_id: str) -> str:
    return "e-" + hashlib.sha256(f"{actor}:{task_id}".encode()).hexdigest()[:10]


def run_scripted(task: dict, style: str, actor: str, recorder: Recorder, seed: int) -> None:
    recorder.clock = SimClock(f"{task['view']['today']}T09:00:00")
    recorder.start(episode_id(actor, task["task_id"]), task["view"], task_id=task["task_id"], actor=actor, seed=seed)
    desk = Desk(task["hidden"]["state"], recorder)
    final = scripted.work(desk, task, style, random.Random(f"{seed}:{task['task_id']}:{style}"))
    recorder.final(final)
    recorder.end()


def load_episodes(path) -> list[tuple[dict, dict]]:
    return [(episode_meta(evs), clean_episode(evs)) for evs in group_episodes(read_events(path)).values()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(out, seed: int = 7, n_train: int = 150, n_h1: int = 30, n_trap: int = 12, n_dev: int = 24) -> dict:
    out = Path(out)
    tasks = {"train": make_tasks(seed, "train", n_train),
             "dev": make_tasks(seed, "dev", n_dev),
             "eval_h1": make_tasks(seed, "eval_h1", n_h1),
             "eval_trap": make_tasks(seed, "eval_trap", n_trap)}
    actors = {s: actor_id(seed, s) for s in STYLES}

    for split, ts in tasks.items():
        write_jsonl(ts, out / "tasks" / f"{split}.jsonl")
        for style in STYLES:
            path = out / "records" / f"{style}-{split}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("")
            recorder = Recorder(path)
            for task in ts:
                run_scripted(task, style, actors[style], recorder, seed)

    rows = {}
    for style in STYLES:
        rows[style] = [sft_row(ep) for _, ep in load_episodes(out / "records" / f"{style}-train.jsonl")]
        write_jsonl(rows[style], out / "sft" / f"{style}-train.jsonl")
    # Mixed control: the same tickets, each taken from one worker, half and half within every family,
    # so it has as many rows as A or B and differs from them only in whose way each row shows.
    rng = random.Random(f"{seed}:mixed")
    by_family: dict[str, list[int]] = {}
    for i, t in enumerate(tasks["train"]):
        by_family.setdefault(t["family"], []).append(i)
    picks = []
    for family in sorted(by_family):
        ids = by_family[family][:]
        rng.shuffle(ids)
        picks += ids
    assert len(rows["a"]) == len(rows["b"]) == len(tasks["train"])
    mixed = [rows["a" if k % 2 == 0 else "b"][i] for k, i in enumerate(picks)]
    rng.shuffle(mixed)
    write_jsonl(mixed, out / "sft" / "mixed-train.jsonl")

    summary = {}
    for split, ts in tasks.items():
        for style in STYLES:
            _, s = score_records(out / "records" / f"{style}-{split}.jsonl", {t["task_id"]: t for t in ts})
            summary[f"{style}/{split}"] = {k: v for k, v in s.items() if k != "by_family"}

    def eval_only_seen(ts):
        skus = Counter(l["sku"] for t in ts for l in t["grader"]["lines"] if l["sku"] in EVAL_ONLY_SKUS)
        return {"tasks_with_one": sum(any(l["sku"] in EVAL_ONLY_SKUS for l in t["grader"]["lines"]) for t in ts),
                "skus": dict(sorted(skus.items()))}

    files = sorted(p for p in out.rglob("*.jsonl"))
    manifest = {
        "env_version": ENV_VERSION, "seed": seed, "actors": actors,
        "families": {split: dict(Counter(t["family"] for t in ts)) for split, ts in tasks.items()},
        "eval_only_skus": {split: eval_only_seen(ts) for split, ts in tasks.items() if split != "train"},
        "summary": summary,
        "files": {str(p.relative_to(out)): sha256(p) for p in files},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def show_pairs(out, n: int) -> None:
    out = Path(out)
    a = load_episodes(out / "records" / "a-train.jsonl")
    b = load_episodes(out / "records" / "b-train.jsonl")
    tasks = {json.loads(l)["task_id"]: json.loads(l) for l in (out / "tasks" / "train.jsonl").open()}
    for (ma, ea), (_, eb) in list(zip(a, b))[:n]:
        print(f"{ma['task_id']} ({tasks[ma['task_id']]['family']})")
        print("  A:", ", ".join(trace(ea)))
        print("  B:", ", ".join(trace(eb)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="data/h1")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--train", type=int, default=150)
    ap.add_argument("--eval", type=int, default=30)
    ap.add_argument("--trap", type=int, default=12)
    ap.add_argument("--dev", type=int, default=24)
    ap.add_argument("--show", type=int, default=8, help="print this many A/B trace pairs")
    args = ap.parse_args()
    manifest = build(args.out, args.seed, args.train, args.eval, args.trap, args.dev)
    print(json.dumps({"families": manifest["families"], "summary": manifest["summary"]}, indent=1))
    show_pairs(args.out, args.show)


if __name__ == "__main__":
    main()
