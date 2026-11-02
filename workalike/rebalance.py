"""Rebalance a worker's records without being told any rule: rare ways of working get repeated, common ones stay.

    python -m workalike.rebalance --rows data/v2/sft/c-s1-train.jsonl --limit 60 --target 150 --out data/v2/sft/auto60-s1-train.jsonl

A person should not have to work each kind of ticket 25 times. Natural records hold the common path many times and each
exception a few times, and a LoRA learns the common path. This groups episodes by what the worker did (the order of
tools called, nothing about the content), and repeats the rarer groups, at most `cap` times each, until the file has
about `target` rows. It reads no rule, no label and no ticket family. It can only weigh what was seen: an exception the
records never show cannot be learned this way, and the repeated rows are not new observations (the manifest counts both).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from .sft import read_jsonl, write_jsonl


def signature(row: dict) -> str:
    """The tools in the order they were called, a run of one tool counted once."""
    names = [c["function"]["name"] for m in row["messages"] if m["role"] == "assistant" for c in m.get("tool_calls") or []]
    return ">".join(n for i, n in enumerate(names) if i == 0 or names[i - 1] != n) or "(none)"


def rebalance(rows: list[dict], target: int, cap: int = 4, seed: int = 1) -> tuple[list[dict], dict]:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(signature(r), []).append(r)
    copies = {k: 1 for k in groups}
    total = len(rows)
    while total < target:  # one more copy of the smallest group that can still grow
        growable = [k for k in groups if copies[k] < cap]
        if not growable:
            break
        k = min(growable, key=lambda g: (len(groups[g]) * copies[g], g))
        copies[k] += 1
        total += len(groups[k])
    out = [r for k, v in groups.items() for r in v for _ in range(copies[k])]
    random.Random(f"autobalance:{seed}").shuffle(out)
    return out, {"unique_rows": len(rows), "rows": len(out), "groups": len(groups),
                 "by_group": {k: {"seen": len(v), "copies": copies[k]} for k, v in sorted(groups.items(), key=lambda x: -len(x[1]))}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--target", type=int, default=150)
    ap.add_argument("--cap", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows, manifest = rebalance(read_jsonl(args.rows)[: args.limit], args.target, args.cap, args.seed)
    write_jsonl(rows, args.out)
    Path(args.out).with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: manifest[k] for k in ("unique_rows", "rows", "groups")}), Counter(signature(r) for r in rows).most_common(3))


if __name__ == "__main__":
    main()
