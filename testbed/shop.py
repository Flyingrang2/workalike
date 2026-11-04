"""Write a demo shop for the assist window: the catalog and quotes behind one test-bed ticket, without the ticket.

python -m testbed.shop --tasks data/h1/tasks/dev.jsonl --out data/shop.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from workalike.sft import read_jsonl


def shop_from(task: dict) -> dict:
    state = task["hidden"]["state"]
    return {**state, "tickets": [], "drafts": {}, "outbox": []}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    shop = shop_from(read_jsonl(args.tasks)[args.index])
    Path(args.out).write_text(json.dumps(shop, indent=1) + "\n")
    print(f"{args.out}: {len(shop['catalog'])} items, {len(shop['quotes'])} quotes, today {shop['today']}")


if __name__ == "__main__":
    main()
