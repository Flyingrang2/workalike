"""The fixed cases the product loop works and re-runs after a rule changes: six requests a person can follow.

    python -m island.loop_cases --out data/loop/cases.jsonl [--ko "python3 .../tools/translate.py --to ko"]

They come from the v2 generator under their own split ("demo", seed 31), so they share no ticket with any test
set. Two are not generator families:
  pushback      a customer answers the reorder quote the shop already sent at today's list price: too much.
                The generator only plays that inside an episode, as the customer's answer to a reply.
  out_of_scope  a plain order with a question no rule at this desk covers (paying in two parts).
--ko adds a Korean reading of each request for the person's screen; the model only ever sees the English.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from datetime import date, timedelta
from pathlib import Path

from workalike.desk import unit_price_for

from .tasks2 import SAYS, Builder

ORDER = ("reorder_room", "pushback", "short_open", "short_constrained", "missing_date", "out_of_scope")
OUT_OF_SCOPE = "\nAlso, could we pay half now and the other half after delivery?"
TITLES = {"reorder_room": "만료된 견적의 재주문", "pushback": "재주문 견적에 가격 반발",
          "short_open": "재고 부족", "short_constrained": "재고 부족, 날짜를 꼭 맞춰야 함",
          "missing_date": "출고일이 없는 요청", "out_of_scope": "규칙 밖 질문이 붙은 주문"}


def pushback(t: dict) -> dict:
    """A reorder whose expired quote the shop re-priced at list two days ago; the new message pushes back."""
    s, view, g = t["hidden"]["state"], t["view"], t["grader"]
    tid, want = view["ticket_id"], g["wants"][0]
    item = s["catalog"][want["sku"]]
    price = unit_price_for(item["unit_cost"], want["qty"])
    qid, sent = f"Q-{tid.split('-')[-1]}-1", (date.fromisoformat(view["today"]) - timedelta(days=2)).isoformat()
    line = {"sku": want["sku"], "qty": want["qty"], "unit_price": price, "ship_by": g["need_by"],
            "line_total": round(price * want["qty"], 2)}
    s["drafts"][qid] = {"ticket_id": tid, "lines": [line]}
    first = view["from"].split()[0]
    shop = (f"Hello {first}, {g['quote_ref']} has expired, so this is at today's price: quote {qid} is attached, "
            f"{want['qty']} x {item['name']} at ${price:.2f}, ships by {g['need_by']}. Total ${line['line_total']:,.2f}.")
    view["thread"] = view.get("thread", []) + [{"date": sent, "who": "customer", "text": view["body"]},
                                               {"date": sent, "who": "shop", "text": shop}]
    view["body"] = SAYS["push_back"][0].format(qref=g["quote_ref"])
    view["sent_so_far"] = f"quote {qid} at list price on {sent}"
    t["grader"]["sent_quote"] = qid
    return t


def make_cases(seed: int = 31) -> list[dict]:
    b = Builder(seed, "demo")
    cases = []
    for i, name in enumerate(ORDER):
        family = {"pushback": "reorder_room", "out_of_scope": "standard"}.get(name, name)
        t = b.task(i, family)
        if name == "pushback":
            t = pushback(t)
        if name == "out_of_scope":
            t["view"]["body"] += OUT_OF_SCOPE
        t.update(case=name, title=TITLES[name], task_id=f"demo-{i:02d}")
        cases.append(t)
    return cases


def korean(cases: list[dict], cmd: str) -> None:
    def ko(text, tries=2):
        for i in range(tries):  # a translator call can time out now and then; one more try
            run = subprocess.run(shlex.split(cmd), input=text, capture_output=True, text=True)
            if run.returncode == 0:
                return json.loads(run.stdout)["ko"]
        run.check_returncode()
    for t in cases:
        v = t["view"]
        t["ko"] = {"body": ko(v["body"]), "thread": [ko(m["text"]) for m in v.get("thread", [])]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--ko", help="command: English on stdin, {\"ko\": ...} on stdout")
    args = ap.parse_args()
    cases = make_cases(args.seed)
    if args.ko:
        korean(cases, args.ko)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in cases))
    print(f"{out}: {len(cases)} cases")


if __name__ == "__main__":
    main()
