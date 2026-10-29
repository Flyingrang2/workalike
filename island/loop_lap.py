"""Drive the product loop once with a served model, the way a person would press its buttons, and keep the pages.

    python -m island.loop_lap --out runs/loop-lap --cases data/loop/cases.jsonl --cards <cards.json> \
        --edit-card R3 --edit-action "..." [--translator "python3 .../tools/translate.py"]

1. load the cards Glimmer proposed and confirm them all (a check of the machinery: here the one confirming is
   the script, not a person)
2. work every fixed case with the confirmed rules
3. edit one card, preview the edit on the fixed cases (before and after side by side), apply it
4. work every case again under the new version; the drafts from step 2 now ask for a re-check
The agent page after each step is saved as HTML, and lap.json keeps every job.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path

from workalike.loop import LoopApp
from workalike.model_client import ChatModel
from workalike.sft import read_jsonl
from workalike.web2 import command_translator


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--cards", required=True)
    ap.add_argument("--edit-card", required=True)
    ap.add_argument("--edit-action", required=True, help="the card's new action (English or Korean)")
    ap.add_argument("--model", default="muse-glimmer")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--translator")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    to_ko = None
    if args.translator:
        def to_ko(text):
            res = subprocess.run(shlex.split(args.translator) + ["--to", "ko"], input=text, capture_output=True,
                                 text=True, check=True, timeout=120).stdout
            return json.loads(res)["ko"]
    app = LoopApp(out / "app", lambda: ChatModel(args.base_url, args.model, temperature=0, max_tokens=16000),
                  read_jsonl(args.cases), to_en=command_translator(args.translator) if args.translator else None,
                  rule_en=command_translator(args.translator + " --rule") if args.translator else None,
                  to_ko=to_ko, cached_cards=args.cards)

    def step(name):
        app.todo.join()
        (out / f"{name}.html").write_text(app.page_for("/agent"))
        print(name, app.status(), flush=True)

    app.post("/agent/load", {})
    step("1-cards")
    for c in app.book.cards:
        app.post("/agent/status", {"card": c["id"], "status": "confirmed"})
    app.post("/agent/work", {"case": "all"})
    step("2-work-v%d" % app.book.version)
    app.post("/agent/edit", {"card": args.edit_card, "action": args.edit_action})
    if app.message:
        raise SystemExit(app.message)
    app.post("/agent/preview", {"card": args.edit_card})
    step("3-preview")
    app.post("/agent/apply", {"card": args.edit_card})
    app.post("/agent/work", {"case": "all"})
    step("4-work-v%d" % app.book.version)
    (out / "lap.json").write_text(json.dumps({"version": app.book.version, "cards": app.book.cards,
                                              "jobs": list(app.jobs.values())}, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
