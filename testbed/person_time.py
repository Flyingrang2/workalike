"""One person against the LoRA on the main test's tickets: the same tickets, desk tools, answer sheet and grader.

    python -m testbed.person_time plan --out baselines/person-time/plan.json
    python -m testbed.person_time translate --plan <plan> --translator "<command>" --out runs/person-time/ko.json
    python -m testbed.person_time serve --plan <plan> --ko runs/person-time/ko.json --translator "<command>" \
        --out runs/person-time

Open http://127.0.0.1:8771. Half the tickets the person works by hand, half they review the LoRA's recorded first-round
draft and either approve it or mark that it needs fixing. A ticket is timed from Start to the reply or question, with
pauses and the wait for the translator taken out. Nothing here calls the LoRA: its drafts and seconds come from
baselines/results/test24b/graded.jsonl, so no GPU is needed. As for the model, a reply or a question ends the ticket, and
a draft the desk's checks hold back can be written once more.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
import re
import shlex
import statistics
import subprocess
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from workalike.desk import Desk
from workalike.person_desk import CSS, AsWritten, PersonDesk, command_translator, describe_ko
from workalike.product import PRODUCT_ENV, guard, prepare
from workalike.record import Recorder
from workalike.sft import read_jsonl
from workalike.web import make_server

from .grade24 import grade
from .run24 import compact

SHEET = Path("baselines/test24b")
LORA_ROWS = Path("baselines/results/test24b/graded.jsonl")
RULES = Path("baselines/test24/arms/checked.txt")
LORA_ARM, LORA_ROUND = "p-checked", 1
PATTERN = "HRRH"  # hand and review take turns, so getting used to the desk does not favour either
GOOD = ("ready", "right_hold")
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
VERDICT_KO = {
    "ready": "맞음 (보낼 수 있는 견적)",
    "right_hold": "맞음 (물어야 할 것을 물음)",
    "needless_hold": "아쉬움 (할 수 있는 일을 안 함)",
    "dangerous": "위험 (승인하면 안 되는 초안)",
    "failed": "실패 (결과가 없거나 검사에 막힘)",
}
EXTRA_CSS = """
.page{max-width:860px;margin:0 auto;padding:16px;display:flex;flex-direction:column;gap:12px}
.big{font-size:30px;font-weight:600;font-variant-numeric:tabular-nums}.clock{font-variant-numeric:tabular-nums;font-size:17px;font-weight:600;opacity:1}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:5px 8px;border-bottom:1px solid #d9d5c8;vertical-align:top}
th{font-size:13px;color:#5e5b55;font-weight:600}.good{color:#1c6b36;font-weight:600}.bad{color:#a40000;font-weight:600}
.notice{background:#fff3cd;border:1px solid #d9b84c;border-radius:6px;padding:8px 12px}.grid .notice{grid-column:1/3}
.pre{white-space:pre-wrap}.wrap{overflow-x:auto}header a{color:#fff}header .pause{margin-left:0}
.review{display:grid;grid-template-columns:480px 1fr;gap:12px;padding:12px;min-height:calc(100vh - 48px)}
@media(max-width:900px){main,.review{display:block}.review>*,main>*{margin-bottom:12px}}
"""
# Keeps what was typed for this ticket across the desk's page reloads, and the scroll position, so that the person
# does not retype an item code into each form: the timer should measure the work, not the forms.
KEEP_TYPED = """
<script>
(() => {
  const ticket = document.body.dataset.ticket, k = n => `typed:${ticket}:${n}`;
  const get = n => { try { return sessionStorage.getItem(k(n)); } catch (e) { return null; } };
  const put = (n, v) => { try { sessionStorage.setItem(k(n), v); } catch (e) {} };
  const nameOf = i => (i.name === 'quote_id' || i.tagName === 'TEXTAREA')
    ? `${i.form.getAttribute('action')}:${i.form.elements.kind?.value || ''}:${i.name}` : i.name;
  const boxes = 'input[name]:not([type=hidden]),textarea[name]';
  document.querySelectorAll(boxes).forEach(i => {
    if (!i.value) i.value = get(nameOf(i)) || (/^(sku|qty)1$/.test(i.name) ? get(i.name.slice(0, -1)) : '') || '';
  });
  document.querySelectorAll('form').forEach(f => f.addEventListener('submit', () => {
    f.querySelectorAll(boxes).forEach(i => put(nameOf(i), i.value));
    put('scroll', document.querySelector('.grid')?.scrollTop || 0);
  }));
  const g = document.querySelector('.grid'); if (g) g.scrollTop = +(get('scroll') || 0);
})();
</script>
"""


def clock_text(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 60}:{s % 60:02d}"


def make_plan(tasks: list[dict], answers: dict, seed: str = "person-time") -> list[dict]:
    """Split each group of the sheet in two at random, then interleave so that any first part has about the sheet's mix."""
    rng = random.Random(seed)
    groups: dict[str, list[str]] = {}
    for t in tasks:
        groups.setdefault(answers[t["task_id"]]["group"], []).append(t["task_id"])
    sides: dict[str, list[tuple[float, str, str]]] = {"hand": [], "review": []}
    for group in sorted(groups):
        ids = sorted(groups[group])
        rng.shuffle(ids)
        first = "hand" if len(sides["hand"]) <= len(sides["review"]) else "review"
        second = "review" if first == "hand" else "hand"
        for side, part in ((first, ids[::2]), (second, ids[1::2])):
            for i, task_id in enumerate(part):
                sides[side].append(((i + 0.5) / len(part) + rng.random() / 1000, task_id, group))
    for side in sides:
        sides[side].sort()
    plan = []
    for i in range(len(tasks)):
        mode = "hand" if PATTERN[i % len(PATTERN)] == "H" else "review"
        _, task_id, group = sides[mode].pop(0)
        plan.append({"task_id": task_id, "mode": mode, "group": group})
    return plan


def load_sheet(sheet: Path = SHEET) -> tuple[dict, dict]:
    tasks = {t["task_id"]: t for name in ("tickets", "dev") for t in read_jsonl(sheet / f"{name}.jsonl")}
    answers = {a["task_id"]: a for name in ("answers", "dev-answers") for a in read_jsonl(sheet / f"{name}.jsonl")}
    return tasks, answers


def load_lora(path: Path = LORA_ROWS) -> dict:
    return {r["task_id"]: r for r in read_jsonl(path) if r["arm"] == LORA_ARM and r["round"] == LORA_ROUND}


def summary(rows: list[dict], lora: dict) -> dict:
    """What the two ways of working cost and got right. Practice tickets do not count."""
    counted = [r for r in rows if not r.get("warmup")]
    hand = [r for r in counted if r["mode"] == "hand"]
    review = [r for r in counted if r["mode"] == "review"]

    def mid(xs):
        return round(statistics.median(xs), 1) if xs else None

    half = len(hand) // 2
    wrong = [r for r in review if r["lora"]["verdict"] not in GOOD]
    return {
        "hand": {
            "tickets": len(hand),
            "median_seconds": mid([r["seconds"] for r in hand]),
            "right": sum(r["verdict"] in GOOD for r in hand),
            "dangerous": sum(r["verdict"] == "dangerous" for r in hand),
            "first_half_median": mid([r["seconds"] for r in hand[:half]]),
            "second_half_median": mid([r["seconds"] for r in hand[half:]]) if half else None,
            "lora_right_same_tickets": sum(r["lora"]["verdict"] in GOOD for r in hand),
            "lora_median_seconds_same_tickets": mid([r["lora"]["seconds"] for r in hand]),
        },
        "review": {
            "tickets": len(review),
            "median_seconds": mid([r["seconds"] for r in review]),
            "approved": sum(r["decision"] == "approve" for r in review),
            "lora_right": len(review) - len(wrong),
            "lora_wrong": len(wrong),
            "caught": sum(r["decision"] == "fix" for r in wrong),
            "missed": sum(r["decision"] == "approve" for r in wrong),
            "marked_right_draft": sum(r["decision"] == "fix" and r["lora"]["verdict"] in GOOD for r in review),
            "lora_median_seconds": mid([r["lora"]["seconds"] for r in review]),
        },
    }


class PersonTime(PersonDesk):
    """The Korean desk with a stopwatch, one ticket at a time: ready, then work, then what happened beside the LoRA."""

    def __init__(self, items, tasks, answers, lora, ko, recorder, translate, translate_term, out: Path):
        self.tasks, self.answers, self.lora, self.ko, self.out = tasks, answers, lora, ko, Path(out)
        self.results = self.out / "results.jsonl"
        self.rows = read_jsonl(self.results) if self.results.exists() else []
        done = {r["task_id"] for r in self.rows}
        sessions = [
            {
                "episode": "e-" + hashlib.sha256(f"person-time:{i['task_id']}".encode()).hexdigest()[:10],
                "view": tasks[i["task_id"]]["view"],
                "state": tasks[i["task_id"]]["hidden"]["state"],
                "task_id": i["task_id"],
                "actor": "person-time",
                "mode": i["mode"],
                "warmup": bool(i.get("warmup")),
            }
            for i in items
            if i["task_id"] not in done
        ]
        self.planned = sum(1 for i in items if not i.get("warmup"))
        self.ended = False
        korean = AsWritten()
        korean.tickets = ko["tickets"]
        super().__init__(sessions, recorder, korean, self._waited(translate), self._waited(translate_term), self._clock)
        self.describe = self._describe

    def _waited(self, translate):
        """The wait for a machine is not the person's time."""

        def run(text):
            start = time.monotonic()
            try:
                return translate(text)
            finally:
                self.machine += time.monotonic() - start

        return run

    @staticmethod
    def _describe(name, args, r, ko=None):
        if r.get("status") == "held":
            return "답장을 승인 대기로 올렸습니다."
        if r.get("status") == "asked":
            return "사장님께 질문을 올렸습니다."
        try:
            return describe_ko(name, args, r, ko)
        except (KeyError, TypeError):
            return json.dumps(r, ensure_ascii=False)

    def _next(self) -> None:
        self.current = self.queue.pop(0) if self.queue else None
        self.log, self.draft, self.first, self.notice = [], None, None, None
        self.machine, self.paused_for, self.paused_at, self.t0 = 0.0, 0.0, None, None
        self.phase = "ready" if self.current else "done"

    def _start(self) -> None:
        s = self.current
        self.recorder.start(
            s["episode"],
            s["view"],
            task_id=s["task_id"],
            actor=s["actor"],
            entrance=f"person-time-{s['mode']}",
            env=PRODUCT_ENV.version,
        )
        self.desk = Desk(prepare(s["state"], s["view"]["ticket_id"]), self.recorder, env=PRODUCT_ENV)
        self.started_at = time.time()
        self.t0 = time.monotonic()
        if s["mode"] == "review":
            self.log = list(reversed(self.lora[s["task_id"]]["calls"]))
        self.phase = "work"

    def worked(self) -> float:
        if self.t0 is None:
            return 0.0
        now = self.paused_at if self.paused_at is not None else time.monotonic()
        return max(0.0, now - self.t0 - self.paused_for - self.machine)

    def _done(self) -> dict:
        s = self.desk.state
        return {
            "outcome": "asked" if s.get("stop") == "asked" else "draft" if s["outbox"] else "stopped",
            "error": None,
            "drafts": list(s["outbox"]),
            "questions": list(s["questions"]),
            "quotes": dict(s["drafts"]),
            "history": self.desk.history,
            "state": s,
        }

    def _after_send(self) -> None:
        state, ticket = self.desk.state, self.current["view"]["ticket_id"]
        if not state.get("stop"):
            return
        done = self._done()
        checks = guard(state, ticket, done["history"]) if done["drafts"] else []
        failed = [c for c in checks if not c["ok"]]
        if failed and self.first is None:  # the model gets one rewrite for this, and so does the person
            self.first = {"drafts": done["drafts"], "failed": failed}
            self.notice = failed
            state["outbox"].clear()
            state["stop"] = None
            return
        self._close_hand(done, checks)

    def _lora_brief(self, task_id: str) -> dict | None:
        r = self.lora.get(task_id)
        if r is None:
            return None
        return {
            "round": r["round"],
            "seconds": r["seconds"],
            "verdict": r["verdict"],
            "reasons": r["reasons"],
            "first_verdict": r.get("first_verdict"),
        }

    def _base_row(self) -> dict:
        s = self.current
        return {
            "order": len(self.rows) + 1,
            "task_id": s["task_id"],
            "key": self.answers[s["task_id"]]["key"],
            "group": self.answers[s["task_id"]]["group"],
            "mode": s["mode"],
            "warmup": s["warmup"],
            "seconds": round(self.worked(), 1),
            "machine_seconds": round(self.machine, 1),
            "paused_seconds": round(self.paused_for, 1),
            "started": self.started_at,
            "lora": self._lora_brief(s["task_id"]),
        }

    def _close_hand(self, done: dict, checks: list[dict]) -> None:
        s = self.current
        task = self.tasks[s["task_id"]]
        result = compact(done, checks, task)
        row = self._base_row()
        row.update(result=result, rewrote=self.first is not None, first=self.first)
        row.update(grade(self.answers[s["task_id"]], result))
        self.recorder.end("done" if done["outcome"] != "stopped" else "skipped")
        self._save(row)

    def _close_review(self, decision: str, reason: str) -> None:
        row = self._base_row()
        row.update(decision=decision, reason=reason)
        self.recorder.end("reviewed")
        self._save(row)

    def _save(self, row: dict) -> None:
        self.rows.append(row)
        self.out.mkdir(parents=True, exist_ok=True)
        with self.results.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        (self.out / "summary.json").write_text(json.dumps(summary(self.rows, self.lora), ensure_ascii=False, indent=1))
        self.last = row
        self.t0 = None
        self.phase = "reveal"

    def post(self, path: str, form: dict[str, str]) -> None:
        with self.lock:
            if path == "/end":
                self.ended = True
            elif path == "/continue":
                self.ended = False
            elif self.phase == "ready" and path == "/start":
                self._start()
            elif self.phase == "reveal" and path == "/next":
                self.finished.append(self.current["task_id"])
                self._next()
            elif self.phase == "work":
                self._work(path, form)

    def _work(self, path: str, form: dict[str, str]) -> None:
        if path == "/pause" and self.paused_at is None:
            self.paused_at = time.monotonic()
        elif path == "/resume" and self.paused_at is not None:
            self.paused_for += time.monotonic() - self.paused_at
            self.paused_at = None
        elif self.paused_at is not None:
            return
        elif self.current["mode"] == "review":
            held = self.lora[self.current["task_id"]]["verdict"] == "failed"
            if path == "/approve" and not held:
                self._close_review("approve", "")
            elif path == "/fix":
                self._close_review("fix", form.get("reason", "").strip())
        elif path == "/act":
            self.act(form)
        elif path == "/draft":
            self.make_draft(form)
        elif path == "/send":
            self.send()
            self._after_send()
        elif path == "/discard":
            self.draft = None
        elif path == "/finish":
            self._close_hand(self._done(), [])

    def status(self) -> dict:
        return {
            "phase": self.phase,
            "task_id": self.current["task_id"] if self.current else None,
            "done": len(self.rows),
        }

    # Pages

    def _head(self, ticket: str = "") -> str:
        return (
            "<!doctype html><html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,"
            f"initial-scale=1'><title>사람과 LoRA 시간 재기</title><style>{CSS}{EXTRA_CSS}</style></head>"
            f"<body data-ticket='{html.escape(ticket)}'>"
        )

    def _form(self, action: str, label: str, cls: str = "", extra: str = "") -> str:
        return (
            f"<form method='post' action='{action}' style='display:inline'>"
            f"<input type='hidden' name='csrf' value='{self.token}'>{extra}"
            f"<button class='{cls}'>{label}</button></form>"
        )

    def _clock(self) -> str:
        """The desk's header badge: the stopwatch, a pause and the rules."""
        if self.paused_at is not None:
            return f"<span class='clock'>{clock_text(self.worked())} 멈춤</span>" + self._form(
                "/resume", "계속", "quiet"
            )
        return (
            f"<span class='clock' id='clock'>{clock_text(self.worked())}</span>"
            + self._form("/pause", "잠깐 멈춤", "quiet")
            + "<a href='/rules' target='_blank'>규칙 장부</a>"
            f"<script>let s={int(self.worked())};const c=document.getElementById('clock');setInterval(()=>{{s++;"
            "c.textContent=Math.floor(s/60)+':'+String(s%60).padStart(2,'0')},1000)</script>"
        )

    def _position(self) -> str:
        n = sum(not r.get("warmup") for r in self.rows)
        if self.current and self.current["warmup"]:
            return "연습 한 건 (세지 않습니다)"
        return f"{n + 1}번째, 권장 12건, 전체 {self.planned}건"

    def _ready_page(self) -> str:
        s = self.current
        if s["mode"] == "hand":
            what = "<h2>직접 처리합니다</h2><p>왼쪽에 문의가 나오고 오른쪽 칸들이 LoRA가 쓰는 도구와 같습니다. 답장이나 사장님 질문을 보내면 이 문의는 끝납니다. 규칙이 헷갈리면 위쪽 규칙 장부를 펼쳐 보셔도 됩니다.</p>"
        else:
            what = "<h2>LoRA 초안을 검토합니다</h2><p>LoRA가 규칙 없이 처리한 실제 초안을 봅니다. 평소 하시듯 확인한 뒤 승인하거나 고쳐야 한다고 표시합니다. 실제로 고치지는 않고, 이유만 한 줄 적으면 됩니다.</p>"
        if s["warmup"]:
            what += "<p>연습이라 세지 않습니다. 보통 품목 검색, 재고 보기, 원가 보기, 가격 계산, 견적 만들기, 답장 순서로 갑니다.</p>"
        return (
            self._head()
            + "<div class='page'><div class='card'>"
            + f"<p><small>{self._position()}</small></p>{what}"
            + self._form("/start", "시작 (시간이 갑니다)", "finish")
            + "</div>"
            + self._table()
            + (self._form("/end", "여기서 끝내고 결과 보기", "quiet") if self.rows else "")
            + "</div></body></html>"
        )

    def _quotes_html(self, quotes: dict, ids=None) -> str:
        e = html.escape
        out = []
        for qid, q in quotes.items():
            if ids is not None and qid not in ids:
                continue
            rows = "".join(
                f"<tr><td>{e(str(l['sku']))}</td><td>{l['qty']:,}</td><td>${l['unit_price']:,.2f}</td>"
                f"<td>{e(str(l['ship_by']))}</td><td>${l['line_total']:,.2f}</td></tr>"
                for l in q["lines"]
            )
            out.append(
                f"<p><b>견적 {e(qid)}</b></p><div class='wrap'><table><tr><th>품번</th><th>수량</th><th>단가</th>"
                f"<th>출고일</th><th>합계</th></tr>{rows}</table></div>"
            )
        return "".join(out)

    def _lora_draft(self, task_id: str) -> str:
        """The LoRA's draft or question, in Korean with the English it wrote underneath."""
        e = html.escape
        r, ko = self.lora[task_id], self.ko["lora"].get(task_id, {})
        out = []
        for i, d in enumerate(r["drafts"]):
            text = (ko.get("messages") or [])[i] if i < len(ko.get("messages") or []) else d["message"]
            attached = {d["quote_id"]} if d.get("quote_id") else set()
            out.append(
                f"<h2>손님에게 보낼 답장</h2><div class='pre'>{e(text)}</div>"
                f"<details><summary>LoRA가 쓴 영어 원문</summary><div class='en'>{e(d['message'])}</div></details>"
                + self._quotes_html(r["quotes"], attached)
            )
        for i, q in enumerate(r["questions"]):
            text = (ko.get("questions") or [])[i] if i < len(ko.get("questions") or []) else q["question"]
            out.append(
                f"<h2>사장님께 올린 질문</h2><div class='pre'>{e(text)}</div>"
                f"<details><summary>LoRA가 쓴 영어 원문</summary><div class='en'>{e(q['question'])}</div></details>"
            )
        return "".join(out) or "<p>LoRA가 남긴 초안이 없습니다.</p>"

    def _review_page(self) -> str:
        e = html.escape
        task_id = self.current["task_id"]
        r = self.lora[task_id]
        held = [c for c in r.get("checks") or [] if not c["ok"]]
        head = (
            self._head(self.current["view"]["ticket_id"])
            + f"<header><b>LoRA 초안 검토</b><span>{self._position()}</span>{self._clock()}</header>"
        )
        if self.paused_at is not None:
            return (
                head
                + "<div class='page'><div class='card'>멈춰 있습니다. 위쪽 계속을 누르면 이어집니다.</div></div></body></html>"
            )
        blocked = (
            "<div class='notice'><b>데스크 검사에 막혀 승인할 수 없습니다.</b><br>"
            + "<br>".join(e(f"{c['name']}: {c['detail']}") for c in held)
            + "</div>"
            if held
            else ""
        )
        approve = "" if held else self._form("/approve", "이대로 승인", "finish") + " "
        fix = (
            f"<form method='post' action='/fix' style='margin-top:10px'><input type='hidden' name='csrf' value='{self.token}'>"
            "<label>고칠 점 한 줄 <input name='reason' style='width:320px' autocomplete='off'></label>"
            "<button class='quiet'>고쳐야 함</button></form>"
        )
        return (
            head
            + f"<div class='review'><div class='col'>{self._ticket()}"
            + f"<div class='card log'><h2>LoRA가 한 일 (최근 것이 위)</h2>{self._log()}</div></div>"
            + f"<div class='col'><div class='card'>{blocked}{self._lora_draft(task_id)}"
            + f"<div style='margin-top:12px'>{approve}</div>{fix}</div></div></div></body></html>"
        )

    def _hand_page(self) -> str:
        if self.paused_at is not None:
            return (
                self._head() + f"<header><b>직접 처리</b>{self._clock()}</header><div class='page'><div class='card'>"
                "멈춰 있습니다. 위쪽 계속을 누르면 이어집니다.</div></div></body></html>"
            )
        page = self._page().replace(
            f"<style>{CSS}</style></head><body>",
            f"<style>{CSS}{EXTRA_CSS}</style></head><body data-ticket='{html.escape(self.current['view']['ticket_id'])}'>",
        )
        page = page.replace(
            "<button class='finish'>이 티켓 끝내기</button>",
            "<button class='quiet' onclick=\"return confirm('답장 없이 넘어가면 이 문의는 못 한 것으로 셉니다.')\">답장 없이 넘어가기</button>",
        )
        page = re.sub(r"<span>이 티켓 포함 \d+건 남음</span>", f"<span>{self._position()}</span>", page)
        ticket = self.current["view"]["ticket_id"]
        made = [q for q, d in self.desk.state["drafts"].items() if d["ticket_id"] == ticket]
        made = [q for q in made if q not in self.current["state"]["drafts"]]
        if made:  # the newest quote is already in the reply's attachment box; the person can clear it
            head, sep, tail = page.partition("<h2>손님에게 답장")
            tail = tail.replace("<input name='quote_id'", f"<input value='{html.escape(made[-1])}' name='quote_id'", 1)
            page = head + sep + tail
        if self.notice:
            lines = "<br>".join(html.escape(f"{c['name']}: {c['detail']}") for c in self.notice)
            page = page.replace(
                "<div class='grid'>",
                "<div class='grid'><div class='notice'><b>데스크 검사에 막혔습니다. 한 번 다시 쓸 수 있습니다(LoRA도 같습니다).</b>"
                f"<br>{lines}</div>",
                1,
            )
        return page.replace("</body></html>", KEEP_TYPED + "</body></html>")

    def _person_draft(self, row: dict) -> str:
        e = html.escape
        res = row.get("result") or {}
        out = []
        for d in res.get("drafts", []):
            attached = {d["quote_id"]} if d.get("quote_id") else set()
            out.append(
                f"<div class='en pre'>{e(d['message'])}</div>" + self._quotes_html(res.get("quotes", {}), attached)
            )
        for q in res.get("questions", []):
            out.append(f"<p><small>사장님께 올린 질문</small></p><div class='en pre'>{e(q['question'])}</div>")
        return "".join(out) or "<p>보낸 것이 없습니다.</p>"

    def _verdict(self, v: str) -> str:
        return f"<span class='{'good' if v in GOOD else 'bad'}'>{html.escape(VERDICT_KO.get(v, v))}</span>"

    def _reveal_page(self) -> str:
        e = html.escape
        row = self.last
        a = self.answers[row["task_id"]]
        lora = row["lora"]
        sheet = (
            f"<h2>이 문의의 답</h2><p><b>{e(a['title_ko'])}</b></p><p>해야 할 일: {e(a['need_ko'])}</p>"
            + (f"<p>이렇게 해도 맞습니다: {e(a['alt_ko'])}</p>" if a.get("alt_ko") else "")
            + (f"<p>틀린 처리: {e(a['wrong_ko'])}</p>" if a.get("wrong_ko") else "")
        )
        reasons = lambda rs: ("<br><small>" + e("; ".join(rs)) + "</small>") if rs else ""  # noqa: E731
        rows = []
        if row["mode"] == "hand":
            rows.append(
                f"<tr><td>직접 처리</td><td>{clock_text(row['seconds'])}</td><td>{self._verdict(row['verdict'])}{reasons(row['reasons'])}</td></tr>"
            )
        else:
            caught = (
                "승인했습니다"
                if row["decision"] == "approve"
                else "고쳐야 한다고 표시했습니다" + (f": {e(row['reason'])}" if row["reason"] else "")
            )
            rows.append(f"<tr><td>초안 검토</td><td>{clock_text(row['seconds'])}</td><td>{caught}</td></tr>")
        if lora:
            rows.append(
                f"<tr><td>LoRA가 쓴 시간</td><td>{clock_text(lora['seconds'])}</td><td>{self._verdict(lora['verdict'])}{reasons(lora['reasons'])}</td></tr>"
            )
        mine = (
            f"<div class='card'><h2>내가 처리한 것</h2>{self._person_draft(row)}</div>" if row["mode"] == "hand" else ""
        )
        theirs = f"<div class='card'>{self._lora_draft(row['task_id'])}</div>" if lora else ""
        return (
            self._head()
            + "<div class='page'><div class='card'><div class='wrap'><table><tr><th></th><th>시간</th><th>채점</th></tr>"
            + "".join(rows)
            + f"</table></div></div><div class='card'>{sheet}</div>{mine}{theirs}"
            + "<div>"
            + self._form("/next", "다음으로", "finish")
            + " "
            + self._form("/end", "여기서 끝내고 결과 보기", "quiet")
            + "</div>"
            + self._table()
            + "</div></body></html>"
        )

    def _table(self) -> str:
        e = html.escape
        if not self.rows:
            return ""
        body = []
        for r in self.rows:
            mine = (
                self._verdict(r["verdict"])
                if r["mode"] == "hand"
                else ("승인" if r["decision"] == "approve" else "고쳐야 함")
            )
            lora = r["lora"]
            body.append(
                f"<tr><td>{r['order']}</td><td>{'연습, ' if r.get('warmup') else ''}{'직접' if r['mode'] == 'hand' else '검토'}</td>"
                f"<td>{e(self.answers[r['task_id']]['title_ko'] or '연습 문의')}</td><td>{clock_text(r['seconds'])}</td><td>{mine}</td>"
                f"<td>{clock_text(lora['seconds']) if lora else ''}</td><td>{self._verdict(lora['verdict']) if lora else ''}</td></tr>"
            )
        return (
            "<div class='card'><h2>지금까지</h2><div class='wrap'><table><tr><th>순서</th><th>방식</th><th>문의</th><th>시간</th>"
            f"<th>나</th><th>LoRA 시간</th><th>LoRA 채점</th></tr>{''.join(body)}</table></div></div>"
        )

    def _summary_page(self) -> str:
        s = summary(self.rows, self.lora)
        h, r = s["hand"], s["review"]

        def t(v):
            return clock_text(v) if v is not None else "아직 없음"

        learning = (
            f"<p>손으로 한 시간, 앞 절반 중앙값 {t(h['first_half_median'])}, 뒤 절반 중앙값 {t(h['second_half_median'])}. 표본이 작아 경향만 봅니다.</p>"
            if h["tickets"] >= 4
            else ""
        )
        again = "<div>" + self._form("/continue", "더 하기", "finish") + "</div>" if self.current else ""
        return (
            self._head()
            + "<div class='page'><div class='card'><h2>결과</h2><div class='wrap'><table>"
            + "<tr><th></th><th>건수</th><th>사람이 쓴 시간(중앙값)</th><th>맞음</th><th>LoRA 시간(중앙값)</th></tr>"
            + f"<tr><td>직접 처리</td><td>{h['tickets']}</td><td class='big'>{t(h['median_seconds'])}</td><td>{h['right']} / {h['tickets']}</td>"
            + f"<td>{t(h['lora_median_seconds_same_tickets'])} (같은 문의에서 LoRA 맞음 {h['lora_right_same_tickets']})</td></tr>"
            + f"<tr><td>LoRA 초안 검토</td><td>{r['tickets']}</td><td class='big'>{t(r['median_seconds'])}</td><td>LoRA 맞음 {r['lora_right']} / {r['tickets']}</td>"
            + f"<td>{t(r['lora_median_seconds'])}</td></tr></table></div>"
            + learning
            + f"<p>LoRA 초안이 틀린 {r['lora_wrong']}건 가운데 고쳐야 한다고 잡아낸 것 {r['caught']}건, 놓치고 승인한 것 {r['missed']}건. "
            + f"맞는 초안을 고쳐야 한다고 표시한 것 {r['marked_right_draft']}건.</p></div>"
            + again
            + self._table()
            + "</div></body></html>"
        )

    def _rules_page(self) -> str:
        return (
            self._head()
            + f"<div class='page'><div class='card'><h2>규칙 장부</h2><div class='pre'>{html.escape(self.ko.get('rules', ''))}</div></div></div></body></html>"
        )

    def page_for(self, path: str) -> str:
        with self.lock:
            if path == "/rules":
                return self._rules_page()
            if self.ended or self.phase == "done":
                return self._summary_page()
            if self.phase == "ready":
                return self._ready_page()
            if self.phase == "reveal":
                return self._reveal_page()
            return self._review_page() if self.current["mode"] == "review" else self._hand_page()

    def page(self) -> str:
        return self.page_for("/")


def number_problems(ko: str, en: str) -> list[str]:
    """Every number in the English also in the Korean, by value: a date may lose its zero and its year in Korean
    (2026-12-07 reads 12월 7일)."""
    values = lambda text: Counter(n.replace(",", "").lstrip("0").rstrip(".") or "0" for n in NUMBER.findall(text))  # noqa: E731
    missing = values(en) - values(ko)
    gone = {n for n in missing if n not in values(ko) and not re.fullmatch(r"20\d\d", n)}  # a year may go with the date
    return [f"옮기며 빠진 숫자: {sorted(gone)}"] if gone else []


def translate_all(plan: dict, tasks: dict, lora: dict, command: str, rules: str, cache_path: Path) -> dict:
    """Korean for everything the person reads, made once before they start. Finished pieces are kept in cache_path, so a
    failed call costs a rerun of that piece only; a piece that keeps failing stays in English and is listed."""
    items = [i["task_id"] for i in plan["warmup"] + plan["tickets"]]
    rule_lines = [line for line in rules.splitlines() if line.strip()]
    jobs: list[tuple[tuple, str]] = [(("rules", n), line) for n, line in enumerate(rule_lines)]
    for task_id in items:
        v = tasks[task_id]["view"]
        jobs.append((("body", task_id), v["body"]))
        if v.get("sent_so_far"):
            jobs.append((("sent", task_id), v["sent_so_far"]))
        for n, m in enumerate(v.get("thread") or []):
            jobs.append((("thread", task_id, n), m["text"]))
        r = lora.get(task_id)
        for n, d in enumerate(r["drafts"] if r else []):
            jobs.append((("message", task_id, n), d["message"]))
        for n, q in enumerate(r["questions"] if r else []):
            jobs.append((("question", task_id, n), q["question"]))
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    lock = threading.Lock()

    def one(job):
        where, text = job
        key = hashlib.sha256(text.encode()).hexdigest()
        out = cache.get(key)
        problems: list = ["translator failed"]
        for _ in range(3):
            if out is None:
                try:
                    res = subprocess.run(
                        shlex.split(command) + ["--to", "ko"],
                        input=text,
                        capture_output=True,
                        text=True,
                        timeout=300,
                        check=True,
                    ).stdout
                    out = json.loads(res)["ko"]
                except (subprocess.SubprocessError, ValueError, KeyError):
                    continue
            problems = [] if where[0] == "rules" else number_problems(out, text)
            if not problems:
                with lock:
                    cache[key] = out
                    cache_path.write_text(json.dumps(cache, ensure_ascii=False))
                break
            out = None
        return where, out or text, problems

    with ThreadPoolExecutor(6) as pool:
        done = list(pool.map(one, jobs))
    ko: dict = {
        "tickets": {t: {"thread": [], "body": "", "sent_so_far": None} for t in items},
        "lora": {},
        "problems": {},
    }
    rule_parts = {}
    for where, text, problems in done:
        if problems:
            ko["problems"]["/".join(map(str, where))] = problems
        if where[0] == "rules":
            rule_parts[where[1]] = text
        elif where[0] == "body":
            ko["tickets"][where[1]]["body"] = text
        elif where[0] == "sent":
            ko["tickets"][where[1]]["sent_so_far"] = text
        elif where[0] == "thread":
            ko["tickets"][where[1]].setdefault("_t", {})[where[2]] = text
        else:
            kind = "messages" if where[0] == "message" else "questions"
            ko["lora"].setdefault(where[1], {}).setdefault(kind, {})[where[2]] = text
    ko["rules"] = "\n\n".join(rule_parts[n] for n in sorted(rule_parts))
    for t in ko["tickets"].values():
        parts = t.pop("_t", {})
        t["thread"] = [{"text": parts[n]} for n in sorted(parts)]
    for l in ko["lora"].values():
        for kind in list(l):
            l[kind] = [l[kind][n] for n in sorted(l[kind])]
    return ko


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--out", required=True)
    p = sub.add_parser("translate")
    p.add_argument("--plan", required=True)
    p.add_argument("--translator", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("serve")
    p.add_argument("--plan", required=True)
    p.add_argument("--ko", required=True)
    p.add_argument("--translator", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--port", type=int, default=8771)
    args = ap.parse_args()
    tasks, answers = load_sheet()
    if args.cmd == "plan":
        test = [t for t in read_jsonl(SHEET / "tickets.jsonl")]
        plan = {
            "seed": "person-time",
            "tickets_sha256": hashlib.sha256((SHEET / "tickets.jsonl").read_bytes()).hexdigest(),
            "lora": {"arm": LORA_ARM, "round": LORA_ROUND},
            "warmup": [{"task_id": "test24dev-00", "mode": "hand", "warmup": True}],
            "tickets": make_plan(test, answers),
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(plan, indent=1) + "\n")
        print(args.out)
        return
    plan = json.loads(Path(args.plan).read_text())
    if args.cmd == "translate":
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        cache = Path(args.out).with_suffix(".cache.json")
        ko = translate_all(plan, tasks, load_lora(), args.translator, RULES.read_text(), cache)
        Path(args.out).write_text(json.dumps(ko, ensure_ascii=False, indent=1))
        print(f"{args.out}; translation problems left: {len(ko['problems'])}")
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    translate = command_translator(args.translator)

    def term(ko: str) -> tuple[str, str]:
        res = subprocess.run(
            shlex.split(args.translator) + ["--term"], input=ko, capture_output=True, text=True, timeout=120, check=True
        ).stdout
        return json.loads(res)["en"].strip(), "term"

    desk = PersonTime(
        plan["warmup"] + plan["tickets"],
        tasks,
        answers,
        load_lora(),
        json.loads(Path(args.ko).read_text()),
        Recorder(out / "records.jsonl"),
        translate,
        term,
        out,
    )
    print(f"http://127.0.0.1:{args.port}", flush=True)
    make_server(desk, port=args.port).serve_forever()


if __name__ == "__main__":
    main()
