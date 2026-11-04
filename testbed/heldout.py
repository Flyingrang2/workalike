"""Write the training tickets a curve adapter never saw: all of train minus the first N in a curve order.

    python -m testbed.heldout --tasks data/v2/tasks/train.jsonl --records data/v2/records/c-train.jsonl \
        --shuffle 11:curve:1 --n 20 --out data/v2/tasks/heldout-c20-s1.jsonl

The order is the one gen_v2 gave the training file (the same random key over the same episodes), so the
first N here are exactly the rows the adapter trained on with --limit N. The held-out tickets come from the
same mix as training and were never trained on; they are not eval tickets.
"""

from __future__ import annotations

import argparse
import random

from workalike.filter import episode_meta
from workalike.record import group_episodes, read_events
from workalike.sft import read_jsonl, write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--shuffle", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    order = [episode_meta(e)["task_id"] for e in group_episodes(read_events(args.records)).values()]
    random.Random(args.shuffle).shuffle(order)
    seen = set(order[: args.n])
    kept = [t for t in read_jsonl(args.tasks) if t["task_id"] not in seen]
    write_jsonl(kept, args.out)
    print(f"{args.out}: {len(kept)} tickets not in the first {args.n}")


if __name__ == "__main__":
    main()
