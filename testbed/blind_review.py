"""Blind review: for each ticket a person sees several drafts side by side, unlabeled and shuffled, picks the one
closest to how they would have handled it, and approves, edits or rejects each one.

    python -m testbed.blind_review --tasks data/v2/tasks/eval.jsonl --korean <ko.json> --out runs/review-me \
        --translator "<command: English on stdin, Korean on stdout>" base=runs/x/base proc=runs/x/proc me=runs/x/me

A draft is what reached the customer and the owner: the messages sent (with a Korean translation made when the
ticket opens and kept in out/ko-cache.json), the quotes shown with their lines, the answers that came back, and
how the ticket ended. Lookups are left out: drafts are judged on what they do. Which run is which goes to
out/key.json, which the page never reads; the letters are shuffled per ticket from --seed. Answers go to
out/answers.jsonl, one line per ticket, and a restart skips the tickets already answered. The style of the
drafts can give a run away, so this is a review without labels, not a blind test in the strict sense.
"""

from __future__ import annotations

import argparse
import html
import json
import random
import secrets
import shlex
import subprocess
import threading
from pathlib import Path

from workalike.desk2 import QUOTE_ID
from workalike.filter import clean_episode, episode_meta
from workalike.person_desk import CSS, Korean
from workalike.record import group_episodes, read_events
from workalike.sft import read_jsonl
from workalike.web import make_server

LETTERS = "ABCDE"
VERDICTS = {"approve": "그대로 승인", "edit": "고쳐서 승인", "reject": "거절"}


def episodes_by_task(records) -> dict[str, dict]:
    out = {}
    for events in group_episodes(read_events(records)).values():
        out.setdefault(episode_meta(events)["task_id"], clean_episode(events))
    return out


def draft_steps(ep: dict) -> list[dict]:
    """What reached people: messages to the customer and the owner, quotes shown, and the answers."""
    calls = [c for t in ep["turns"] for c in t["calls"]]
    made = {
        c["result"]["quote_id"]: c["result"]
        for c in calls
        if c["name"] == "create_quote" and "error" not in c["result"]
    }
    steps = []
    for c in calls:
        r = c["result"] or {}
        if "error" in r:
            continue
        if c["name"] == "send_reply":
            shown = [
                q.upper()
                for q in ([c["args"].get("quote_id")] if c["args"].get("quote_id") else [])
                + QUOTE_ID.findall(c["args"]["message"])
            ]
            steps.append(
                {
                    "to": "customer",
                    "text": c["args"]["message"],
                    "answer": r.get("customer_reply"),
                    "quotes": [made[q] for q in dict.fromkeys(shown) if q in made],
                }
            )
        elif c["name"] == "ask_owner":
            steps.append({"to": "owner", "text": c["args"]["question"], "answer": r.get("owner_reply"), "quotes": []})
    return steps


class Review:
    def __init__(self, tasks, runs: dict[str, dict], korean: Korean, translate, out: Path, seed: int):
        self.out, self.korean, self.translate, self.seed = out, korean, translate, seed
        self.runs = runs
        self.token = secrets.token_urlsafe(16)
        self.lock = threading.Lock()
        done = (
            {json.loads(l)["task_id"] for l in (out / "answers.jsonl").open()}
            if (out / "answers.jsonl").exists()
            else set()
        )
        self.queue = [t for t in tasks if t["task_id"] not in done]
        self.cache_path = out / "ko-cache.json"
        self.cache = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        key = json.loads((out / "key.json").read_text()) if (out / "key.json").exists() else {}
        for t in tasks:
            order = sorted(runs)
            random.Random(f"{seed}:{t['task_id']}").shuffle(order)
            key[t["task_id"]] = dict(zip(LETTERS, order))
        (out / "key.json").write_text(json.dumps(key, indent=1) + "\n")
        self.key = key

    def ko(self, text: str | None) -> str | None:
        if not text:
            return None
        if text not in self.cache:
            try:
                self.cache[text] = self.translate(text)
            except Exception as e:  # shown as English only; the person can still judge
                return f"(번역 실패: {type(e).__name__})"
            self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=1) + "\n")
        return self.cache[text]

    def status(self) -> dict:
        return {"task_id": self.queue[0]["task_id"] if self.queue else None, "left": len(self.queue)}

    def post(self, path: str, form: dict[str, str]) -> None:
        with self.lock:
            if path != "/answer" or not self.queue or form.get("task_id") != self.queue[0]["task_id"]:
                return
            t = self.queue[0]
            letters = list(self.key[t["task_id"]])
            row = {
                "task_id": t["task_id"],
                "closest": form.get("closest"),
                "drafts": {L: {"verdict": form.get(f"v{L}"), "note": form.get(f"n{L}", "").strip()} for L in letters},
            }
            with (self.out / "answers.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.queue.pop(0)

    def draft_html(self, task: dict, ep: dict | None) -> str:
        e = html.escape
        if ep is None:
            return "<p>기록이 없습니다.</p>"
        parts = []
        for s in draft_steps(ep):
            who = "손님에게" if s["to"] == "customer" else "사장님께"
            quotes = "".join(
                f"<div class='facts'>견적 {e(q['quote_id'])}: "
                + "; ".join(
                    f"{e(l['sku'])} {l['qty']:,}개 단가 ${l['unit_price']:,.2f} 출고 {e(l['ship_by'])}"
                    for l in q["lines"]
                )
                + f". 합계 ${q['total']:,.2f}</div>"
                for q in s["quotes"]
            )
            answer_ko = (
                self.korean.answer(task["task_id"], s["answer"], task["hidden"]["state"]) if s["answer"] else None
            )
            parts.append(
                f"<p><small>{who}</small><br>{e(self.ko(s['text']) or '')}"
                f"<br><span class='en'>{e(s['text'])}</span>{quotes}"
                + (
                    f"<span class='cust'>{'손님' if s['to'] == 'customer' else '사장님'} 답: "
                    f"{e(answer_ko or s['answer'])}</span>"
                    if s["answer"]
                    else ""
                )
                + "</p>"
            )
        return "".join(parts) or "<p>아무에게도 보내지 않고 끝냈습니다.</p>"

    def page(self) -> str:
        e = html.escape
        head = (
            f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>초안 비교</title><style>{CSS}"
            ".drafts{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(320px,1fr))}"
            ".drafts .card{overflow:auto;max-height:60vh}</style></head><body>"
        )
        if not self.queue:
            return (
                head
                + "<header><b>초안 비교</b></header><main><div class='card'>모두 끝났습니다. 수고하셨습니다.</div></main>"
            )
        t = self.queue[0]
        ko = self.korean.tickets.get(t["task_id"], {})
        v = t["view"]
        letters = self.key[t["task_id"]]
        cards = "".join(
            f"<div class='card'><h2>초안 {L}</h2>{self.draft_html(t, self.runs[run].get(t['task_id']))}"
            f"<div>"
            + "".join(
                f"<label><input type='radio' name='v{L}' value='{k}' required> {lab}</label>"
                for k, lab in VERDICTS.items()
            )
            + f"</div><textarea name='n{L}' placeholder='고칠 곳이나 거절한 이유' style='height:48px'></textarea></div>"
            for L, run in letters.items()
        )
        closest = "".join(
            f"<label><input type='radio' name='closest' value='{L}' required> 초안 {L}</label>" for L in letters
        )
        return (
            head + f"<header><b>초안 비교</b><span>남은 티켓 {len(self.queue)}건</span></header>"
            f"<main style='display:block;height:auto'><div class='card ticket' style='max-height:none'>"
            f"<h2>티켓 {e(v['ticket_id'])} · {e(ko.get('who', v['from']))}</h2>"
            f"<div class='body'>{e(ko.get('body', v['body']))}</div>"
            f"<details><summary>영어 원문 보기</summary><div class='en'>{e(v['body'])}</div></details></div>"
            f"<form method='post' action='/answer'><input type='hidden' name='csrf' value='{self.token}'>"
            f"<input type='hidden' name='task_id' value='{e(t['task_id'])}'>"
            f"<div class='drafts' style='margin:12px 0'>{cards}</div>"
            f"<div class='card'><h2>내가 했을 방식에 가장 가까운 것</h2>{closest} "
            f"<button style='margin-left:16px'>저장하고 다음</button></div></form></main></body></html>"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--korean", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--translator", required=True, help='English on stdin, Korean (or JSON {"ko"}) on stdout')
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("runs", nargs="+", help="name=folder with records.jsonl (up to five)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def translate(text):
        raw = subprocess.run(
            shlex.split(args.translator), input=text, capture_output=True, text=True, timeout=180, check=True
        ).stdout
        try:
            return json.loads(raw)["ko"]
        except (json.JSONDecodeError, KeyError, TypeError):
            return raw.strip()

    runs = {
        name: episodes_by_task(Path(folder) / "records.jsonl")
        for name, folder in (s.split("=", 1) for s in args.runs[: len(LETTERS)])
    }
    review = Review(
        read_jsonl(args.tasks), runs, Korean(json.loads(Path(args.korean).read_text())), translate, out, args.seed
    )
    server = make_server(review, port=args.port)
    print(f"http://127.0.0.1:{args.port}  ({len(review.queue)} tickets to go, Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
