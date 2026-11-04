"""The tickets the demo serves, made once and frozen before any model has seen them.

python -m demo.make_tickets --seed 1103 --out demo/tickets.jsonl
"""

import argparse
import hashlib
import json
from pathlib import Path

from testbed.sheet24 import Sheet


def build(seed: int) -> dict:
    sh = Sheet(seed, "demo")
    price = sh.reorder_push(sh.base(authority=0.05), "reorder_room")
    date = sh.plain(sh.base(authority=0.05), "missing_date")
    return {"price": price, "date": date}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1103)
    p.add_argument("--out", default="demo/tickets.jsonl")
    a = p.parse_args()
    t = build(a.seed)
    text = "".join(json.dumps({"kind": k, **v}, ensure_ascii=False) + "\n" for k, v in t.items())
    Path(a.out).write_text(text)
    print(a.out, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
