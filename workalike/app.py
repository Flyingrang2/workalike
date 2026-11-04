"""The product loop on one screen: the person works as usual on the left, and the agent pane on the right finds
the rules in the records, has the person confirm them, works new requests with them, and shows what a change does.

    python -m workalike ...   (wires the models, translators and files)

Korean by default; lang="en" sends every page through workalike.lang for people who read English.

/        both panes side by side
/work    the Korean quote desk, recording through a switch the person controls (it applies from the next ticket)
/agent   records (the person's, which they can delete, and a virtual worker's, read-only); rule cards (propose,
         confirm, ask, edit, preview the edit on the fixed cases, apply); new work (a fixed case worked with the
         confirmed rules: a draft waiting for approval, a question for the person, or stopped); approvals

Nothing is sent from here. An approved reply goes to sent.jsonl, where a mail connector would pick it up, and only
after guard() passed on it. A draft made under an older rule version must be re-checked before it can be approved.
"""

from __future__ import annotations

import html
import json
import queue
import re
import secrets
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import product
from .autolearn import active_adapter
from .desk2 import list_price
from .envs import V2
from .filter import clean_episode, episode_meta
from .lang import boundary
from .person_desk import HANGUL, describe_en, describe_ko, translation_problems
from .record import Recorder, group_episodes, read_events, wall_clock
from .rules import STATES, RuleBook, propose

e = html.escape
LOOKUPS = ("lookup_quote", "search_catalog", "get_cost", "check_stock", "calc_price", "create_quote")
OUTCOME = {
    "draft": "초안, 승인 대기",
    "asked": "담당자에게 질문",
    "stopped": "멈춤(초안도 질문도 없음)",
    "max_turns": "멈춤(너무 오래 걸림)",
    "error": "오류",
}
RULE_LINE = re.compile(r"^(R\d+)\b", re.MULTILINE)
WORD = re.compile(r"[a-z]{4,}")
COMMON = {
    "customer",
    "quote",
    "worker",
    "with",
    "from",
    "that",
    "this",
    "when",
    "before",
    "price",
    "within",
    "their",
    "they",
    "have",
    "only",
    "into",
    "then",
    "than",
    "item",
    "items",
}

CSS = """
*{box-sizing:border-box}body{margin:0;font:14px/1.45 system-ui,"Apple SD Gothic Neo","Malgun Gothic",sans-serif;color:#22211f;background:#f4f1e6}
header{height:44px;display:flex;align-items:center;gap:14px;padding:0 14px;background:#22211f;color:#fff}header b{font-size:16px}
.frames{display:grid;grid-template-columns:58fr 42fr;height:calc(100vh - 44px)}iframe{width:100%;height:100%;border:0;border-right:2px solid #22211f}
.pane{padding:10px;display:flex;flex-direction:column;gap:10px}
.card{background:#fbfaf4;border:1px solid #d9d5c8;border-radius:6px;padding:8px 10px}
h2{font-size:13px;margin:0 0 6px;color:#5e5b55}h3{font-size:13px;margin:0 0 4px}
button{font:inherit;padding:3px 10px;border-radius:4px;border:1px solid #22211f;background:#22211f;color:#fff;cursor:pointer}
button.quiet{background:#fbfaf4;color:#22211f}button.go{background:#ff6a13;border-color:#ff6a13;color:#22211f;font-weight:600}button.no{background:#fff;color:#a40000;border-color:#a40000}
form.inline{display:inline}textarea{width:100%;font:inherit;height:52px;padding:4px}
.chip{white-space:nowrap;font-size:12px;padding:0 7px;border-radius:9px;background:#eee9db;margin-left:6px}
.unconfirmed{background:#fff1c2}.applied,.confirmed{background:#d2f4dc}.settle{margin:4px 0 8px}.ask{background:#ffe1c2}.dropped{background:#eee;color:#5e5b55}.records{background:#eee9db}.person{background:#efe2fb}
.draft{background:#fff1c2}.asked{background:#ffe1c2}.stopped,.max_turns,.error{background:#ffd8d8}.approved{background:#d2f4dc}.rejected{background:#eee}
.rule{border-top:1px solid #eee;padding:6px 0}.rule p{margin:2px 0}.en{color:#5e5b55;font-size:12px;white-space:pre-wrap}
.ok{color:#1a7f37}.bad{color:#a40000}.msg{background:#fff1c2;border:1px solid #e0c060}.rec{color:#a40000;font-weight:600}
.reply{background:#fbfaf4;border:1px solid #d9d5c8;border-radius:4px;padding:6px;white-space:pre-wrap;margin:4px 0}
table{border-collapse:collapse;width:100%}td,th{border:1px solid #d9d5c8;padding:3px 6px;text-align:left;font-size:13px;vertical-align:top}
details summary{cursor:pointer;color:#22211f}small{color:#5e5b55}
"""


# The agent pane asks /status every 3 s and reloads only when something new is there and the person is not
# typing (a timed reload used to wipe a card being edited).
POLL = """<script>
const shown = "__STAMP__";
let typing = false;
document.addEventListener("input", e => { if (e.target.matches("textarea,input")) typing = true; });
document.addEventListener("submit", () => { typing = false; });
setInterval(async () => {
  if (typing || document.activeElement?.matches("textarea,input")) return;
  try { const s = await (await fetch("/status")).json(); if (s.stamp !== shown) location.reload(); } catch (e) {}
}, 3000);
</script>"""


def rule_ids(text: str) -> set[int]:
    return {int(r[1:]) for r in RULE_LINE.findall(text or "")}


class DeskApp:
    def __init__(
        self,
        out,
        make_model,
        cases: list[dict],
        desk=None,
        switch=None,
        person_records=None,
        examples: list[tuple[str, dict]] | None = None,
        examples_label: str = "",
        to_en=None,
        to_ko=None,
        cached_cards=None,
        workers: int = 6,
        rule_en=None,
        versions=None,
        adapter_model=None,
        lang: str = "ko",
    ):
        self.lang = lang
        self.keep, self.show = boundary(lang)
        self.describe = describe_en if lang == "en" else describe_ko
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.make_model, self.cases, self.desk, self.switch = make_model, cases, desk, switch
        self.person_records = Path(person_records) if person_records else None
        self.examples, self.examples_label = examples or [], examples_label
        self.to_en, self.to_ko = to_en, to_ko
        self.rule_en = rule_en or to_en  # a rule line is translated as an instruction, not as a letter
        self.cached_cards = Path(cached_cards) if cached_cards else None
        self.versions, self.adapter_model = versions, adapter_model  # the autolearn folder and how to reach its adapter
        self.token = desk.token if desk else secrets.token_urlsafe(16)
        self.book = RuleBook(self.out / "rules.json")
        self.lock = threading.Lock()
        self.jobs: dict[str, dict] = {}
        jobs_file = self.out / "jobs.json"
        if jobs_file.exists():
            self.jobs = {j["id"]: j for j in json.loads(jobs_file.read_text())}
            for j in self.jobs.values():  # a job cut off by a restart does not resume on its own
                if j["status"] in ("queued", "running"):
                    j.update(status="error", error="서버가 다시 시작되어 끊겼습니다")
        self.answers: dict[str, dict] = {}
        self.message = ""
        self.todo: queue.Queue = queue.Queue()
        for _ in range(workers):
            threading.Thread(target=self._worker, daemon=True).start()

    # ---------- jobs ----------
    def _save_jobs(self):
        tmp = self.out / "jobs.tmp"
        tmp.write_text(json.dumps(list(self.jobs.values()), ensure_ascii=False) + "\n")
        tmp.replace(self.out / "jobs.json")

    def _enqueue(self, kind: str, key: str, **fields) -> str | None:
        """One job per key while it is queued or running: a repeated button press adds nothing."""
        if any(j["key"] == key and j["status"] in ("queued", "running") for j in self.jobs.values()):
            self.message = "같은 일이 이미 처리 중입니다."
            return None
        jid = f"j{len(self.jobs) + 1:03d}"
        self.jobs[jid] = {
            "id": jid,
            "kind": kind,
            "key": key,
            "status": "queued",
            "created": wall_clock(),
            "decision": None,
            **fields,
        }
        self._save_jobs()
        self.todo.put(jid)
        return jid

    def _worker(self):
        while True:
            jid = self.todo.get()
            with self.lock:
                job = self.jobs[jid]
                job["status"] = "running"
            try:
                result = getattr(self, "_run_" + job["kind"])(job)
                with self.lock:
                    job.update(result, status="done")
            except Exception as err:  # the page shows it; the person can try again
                with self.lock:
                    job.update(status="error", error=f"{type(err).__name__}: {err}")
            with self.lock:
                self._save_jobs()
            self.todo.task_done()

    def _ko(self, text: str) -> str | None:
        if not self.to_ko or not text:
            return None
        try:
            return self.to_ko(text)
        except Exception:
            return None

    def _ko_cards(self, cards: list[dict]) -> dict:
        """Korean for every field of every card, several translator calls at once."""
        texts = [t for c in cards for t in (c["situation"], c["action"], *c["unknowns"])]
        with ThreadPoolExecutor(8) as pool:
            ko = iter(list(pool.map(self._ko, texts)))
        return {
            i: {"situation": next(ko), "action": next(ko), "unknowns": [next(ko) for _ in c["unknowns"]]}
            for i, c in enumerate(cards)
        }

    def _run_propose(self, job):
        records = self.examples if job["source"] == "examples" else self.my_records()
        if not records:
            raise ValueError("규칙을 찾을 기록이 없습니다")
        model = self.make_model()

        def chat(messages):
            text = (model(messages, None).get("content") or "").strip()
            if not text:
                raise ValueError("모델이 빈 답을 냈습니다")
            return text

        cards = propose(records, V2, chat)["cards"]
        ko = self._ko_cards(cards)
        with self.lock:
            ids = self.book.add_proposed(cards, job["source"], ko, status="applied")
        return {"cards": ids}

    def _run_load(self, job):
        data = json.loads(self.cached_cards.read_text())
        ko = self._ko_cards(data["cards"])
        with self.lock:
            ids = self.book.add_proposed(data["cards"], f"cached:{self.cached_cards.name}", ko, status="applied")
        return {"cards": ids}

    def _attempt(self, job, rules, adapter=None):
        case = self.cases[job["case"]]
        tid = case["view"]["ticket_id"]
        # Each attempt records with its own recorder (a recorder follows one episode); previews are not kept.
        recorder = None if job.get("preview") else Recorder(self.out / "agent.jsonl")
        model = self.adapter_model(adapter["name"]) if adapter else self.make_model()
        # An adapter works with no rules in its prompt: what it learned is in its weights, and it keeps the prompt it
        # was trained on. A rewrite note still reaches it.
        trained_on = product.PRODUCT_SYSTEM if adapter and adapter.get("prompt") == "product" else V2.system_prompt
        keep = {"base_system": trained_on, "ask_used": False} if adapter else {}
        done = product.work(
            case["view"],
            case["hidden"]["state"],
            model,
            rules,
            recorder,
            answers={tid: job["answer"]} if job.get("answer") else None,
            episode=f"e-{job['id']}-{secrets.token_hex(3)}",
            one_nudge=True,
            **keep,
        )
        job.setdefault("spent", []).extend(getattr(model, "calls", []))
        checks = product.guard(done["state"], tid, done["history"], inner_words=True) if done["drafts"] else []
        return done, checks

    def active_adapter(self):
        """Not when the rules have moved on since it was trained: a person's change counts from the next request,
        through the written rules, until the adapter has learned it."""
        a = active_adapter(self.versions) if self.versions and self.adapter_model else None
        return None if a and a.get("rule_version") not in (None, self.book.version) else a

    def _run_work(self, job):
        """Work the case; if the desk's checks hold the draft back, say why and let the model write it once more.
        Both drafts are kept, and a second draft that still fails stays unapprovable."""
        adapter = self.active_adapter()
        done, checks, first = product.with_one_rewrite(
            lambda rules: self._attempt(job, rules, adapter), "" if adapter else job["rules"]
        )
        spent = job.pop("spent", [])
        if first:
            first.pop("job")
        s = done["state"]
        quotes = {}
        for qid in sorted({d["quote_id"] for d in done["drafts"] if d.get("quote_id")}):
            lines = s["drafts"][qid]["lines"]
            quotes[qid] = [
                {**l, "off": round(max(0.0, 1 - l["unit_price"] / list_price(s, l["sku"], lines)), 4)} for l in lines
            ]
        used = done["used"]
        known = rule_ids(job["rules"])
        return {
            "outcome": done["outcome"],
            "error": done["error"],
            "used": used,
            "adapter": adapter and adapter["name"],
            "tokens": sum((c["usage"] or {}).get("completion_tokens", 0) for c in spent),
            "seconds": round(sum(c["seconds"] for c in spent)),
            "used_unknown": sorted(set(used or []) - known) if used is not None else [],
            "drafts": [{**d, "ko": self._ko(d["message"])} for d in done["drafts"]],
            "questions": [{**q, "ko": self._ko(q["question"])} for q in done["questions"]],
            "quotes": quotes,
            "checks": checks,
            "first": first,
            "info": [describe_ko(c["name"], c["args"], c["result"]) for c in done["history"] if c["name"] in LOOKUPS],
            "calls": [{k: c[k] for k in ("name", "args", "result")} for c in done["history"] if c["name"] in LOOKUPS],
        }

    # ---------- records ----------
    def my_records(self) -> list[tuple[str, dict]]:
        if not self.person_records or not self.person_records.exists():
            return []
        out = []
        for events in group_episodes(read_events(self.person_records)).values():
            if events[-1]["type"] == "episode_end":
                out.append((episode_meta(events)["ep"], clean_episode(events)))
        return out

    def _in_progress(self) -> str | None:
        if self.desk and self.desk.current and self.switch and self.switch.live:
            return self.desk.current["episode"]
        return None

    def delete_record(self, ep: str):
        if ep == self._in_progress():
            self.message = "지금 기록 중인 티켓은 지울 수 없습니다. 끝낸 뒤에 지우십시오."
            return
        lock = self.desk.lock if self.desk else threading.Lock()
        with lock:
            events = read_events(self.person_records)
            kept = [ev for ev in events if ev["ep"] != ep]
            tmp = self.person_records.with_suffix(".tmp")
            tmp.write_text("".join(json.dumps(ev, ensure_ascii=False) + "\n" for ev in kept))
            tmp.replace(self.person_records)
        self.book.forget_records({ep})
        self.message = f"기록 {ep}를 지웠습니다. 이 기록을 근거로 쓴 카드에는 표시가 남습니다."

    # ---------- posts ----------
    def post(self, path: str, form: dict[str, str]) -> None:
        if not path.startswith(("/agent/", "/work/")):
            if self.desk:
                self.desk.post(path, form)
            return
        with self.lock:
            self.message = ""
            getattr(self, "_post_" + path.rsplit("/", 1)[1], lambda f: None)(form)

    def after_post(self, path: str) -> str:
        return "/agent" if path.startswith("/agent/") else "/work"

    def _post_record(self, form):
        if self.switch:
            self.switch.on = form.get("on") == "1"

    def _post_delete(self, form):
        self.delete_record(form.get("ep", ""))

    def _post_propose(self, form):
        source = "examples" if form.get("source") == "examples" else "mine"
        self._enqueue("propose", f"propose:{source}", source=source)

    def _post_load(self, form):
        if self.cached_cards:
            self._enqueue("load", "load")

    def _post_status(self, form):
        if form.get("status") in STATES:
            self.book.set_status(form["card"], form["status"])

    def _post_edit(self, form):
        card = self.book.card(form["card"])
        fields = {}
        for f in ("situation", "action"):
            text = form.get(f, "").strip()
            if not text or text in (card[f]["ko"], card[f]["en"]):
                continue
            if HANGUL.search(text) and self.rule_en:
                en, _ = self.rule_en(text)
                problems = translation_problems(text, en)
                if problems:
                    self.message = "번역에서 숫자나 품번이 달라져 저장하지 않았습니다: " + "; ".join(problems)
                    return
                fields[f] = {"en": en, "ko": text}
            else:
                fields[f] = {"en": text, "ko": None}
        if fields:
            self.book.edit(form["card"], **fields)

    def _post_settle(self, form):
        """The person answers one thing the records could not settle."""
        cid, question, text = form["card"], form.get("question", ""), form.get("answer", "").strip()
        if not text or question not in {u["en"] for u in self.book.card(cid)["unknowns"]}:
            return
        ko, en = None, text
        if HANGUL.search(text) and self.rule_en:
            en, _ = self.rule_en(text)
            problems = translation_problems(text, en)
            if problems:
                self.message = "번역에서 숫자나 품번이 달라져 저장하지 않았습니다: " + "; ".join(problems)
                return
            ko = text
        self.book.settle(cid, question, ko, en)
        self.message = f"{cid}에 답을 적었습니다. 규칙 버전 v{self.book.version}."

    def _post_drop(self, form):
        self.book.drop_edit(form["card"])

    def _post_apply(self, form):
        self.book.apply_edit(form["card"])
        self.message = f"{form['card']}을 고친 대로 적용했습니다. 규칙 버전 v{self.book.version}."

    def _post_preview(self, form):
        cid = form["card"]
        before, after = self.book.as_procedure(), self.book.as_procedure(preview=cid)
        batch = f"b{len(self.jobs) + 1:03d}-{cid}"
        for i in range(len(self.cases)):
            self._enqueue(
                "work",
                f"{batch}:before:{i}",
                case=i,
                rules=before,
                version=self.book.version,
                batch=batch,
                side="before",
                preview=cid,
            )
            self._enqueue(
                "work",
                f"{batch}:after:{i}",
                case=i,
                rules=after,
                version=self.book.version,
                batch=batch,
                side="after",
                preview=cid,
            )

    def _post_work(self, form):
        which = range(len(self.cases)) if form.get("case") == "all" else [int(form["case"])]
        for i in which:
            self._enqueue(
                "work",
                f"work:{i}:v{self.book.version}",
                case=i,
                rules=self.book.as_procedure(),
                version=self.book.version,
            )

    def _post_answer(self, form):
        job = self.jobs[form["job"]]
        says = form.get("says", "").strip()
        if not says:
            return
        en = self.to_en(says)[0] if HANGUL.search(says) and self.to_en else says
        answer = {"says": en, "ko": says, "policy": "floor_ok" if form.get("floor_ok") == "1" else None}
        self._enqueue(
            "work",
            f"work:{job['case']}:v{self.book.version}:answered",
            case=job["case"],
            rules=self.book.as_procedure(),
            version=self.book.version,
            answer=answer,
            answers_job=job["id"],
        )

    def _post_approve(self, form):
        job = self.jobs[form["job"]]
        if job.get("decision") or job.get("preview") or job.get("outcome") != "draft":
            return
        if not all(c["ok"] for c in job["checks"]):
            self.message = "검사를 통과하지 못한 초안은 승인할 수 없습니다."
            return
        if job["version"] != self.book.version and form.get("recheck") != "1":
            self.message = "이 초안은 바뀌기 전 규칙으로 만들었습니다. 다시 확인했다고 표시한 뒤 승인하십시오."
            return
        job["decision"] = "approved"
        row = {
            "t": wall_clock(),
            "job": job["id"],
            "case": self.cases[job["case"]]["task_id"],
            "rule_version": job["version"],
            "now_version": self.book.version,
            "replies": job["drafts"],
            "quotes": job["quotes"],
        }
        with (self.out / "sent.jsonl").open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._decided(job, "approved")

    def _post_reject(self, form):
        job = self.jobs[form["job"]]
        if not job.get("decision"):
            job["decision"] = "rejected"
            self._decided(job, "rejected")

    def _decided(self, job, what):
        with (self.out / "decisions.jsonl").open("a") as f:
            f.write(json.dumps({"t": wall_clock(), "job": job["id"], "decision": what}) + "\n")
        self._save_jobs()

    # ---------- pages ----------
    def status(self) -> dict:
        with self.lock:
            busy = sum(j["status"] in ("queued", "running") for j in self.jobs.values())
            return {
                "version": self.book.version,
                "busy": busy,
                "jobs": len(self.jobs),
                "recording": bool(self.switch and self.switch.on),
                "stamp": self._stamp(),
            }

    def _stamp(self) -> str:
        """Changes whenever the agent pane would show something new (the page polls it instead of reloading)."""
        done = sum(j["status"] in ("done", "error") for j in self.jobs.values())
        decided = sum(bool(j.get("decision")) for j in self.jobs.values())
        return f"{self.book.version}.{len(self.book.cards)}.{len(self.jobs)}.{done}.{decided}"

    def page(self) -> str:
        return self.page_for("/")

    def page_for(self, path: str) -> str:
        if path.startswith("/work"):  # the desk's page has been through the boundary already
            return (
                self.desk.page()
                if self.desk
                else self.show(self._doc("업무", "<div class='pane'>업무 화면 없음</div>"))
            )
        if path.startswith("/agent"):
            with self.lock:
                return self.show(self._agent())
        return self.show(
            "<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>workalike</title>"
            f"<style>{CSS}</style></head><body><header><b>workalike</b><span>평소처럼 일하면, 적힌 적 없는 "
            "업무 규칙을 찾아 모르는 것만 묻고, 승인할 일을 준비해 둡니다</span></header><div class='frames'>"
            "<iframe src='/work' title='업무'></iframe><iframe src='/agent' title='대리인'></iframe></div></body></html>"
        )

    def badge(self) -> str:
        """The recording switch in the desk's header."""
        if not self.switch:
            return ""
        now = "<span class='rec'>● 기록 중</span>" if self.switch.live else "<span>○ 이 티켓은 기록하지 않음</span>"
        nxt = "다음 티켓부터 끄기" if self.switch.on else "다음 티켓부터 기록"
        return (
            f"{now}<form method='post' action='/work/record' class='inline'>{self._csrf()}"
            f"<input type='hidden' name='on' value='{'0' if self.switch.on else '1'}'>"
            f"<button class='quiet'>{nxt}</button></form>"
        )

    def _csrf(self) -> str:
        return f"<input type='hidden' name='csrf' value='{self.token}'>"

    def _button(self, action, label, cls="", **fields) -> str:
        hidden = "".join(f"<input type='hidden' name='{k}' value='{e(str(v))}'>" for k, v in fields.items())
        return (
            f"<form method='post' action='/agent/{action}' class='inline'>{self._csrf()}{hidden}"
            f"<button class='{cls}'>{label}</button></form>"
        )

    def _doc(self, title, body, stamp=None) -> str:
        return (
            f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>{title}</title>"
            f"<style>{CSS}</style></head><body>{body}"
            + (POLL.replace("__STAMP__", stamp) if stamp else "")
            + "</body></html>"
        )

    def _learn_card(self) -> str:
        path = Path(self.versions) / "status.json" if self.versions else None
        if not path or not path.exists():
            return ""
        s = json.loads(path.read_text())
        a = s.get("active")
        on = (
            (
                f"<p>켜진 어댑터 <b>{e(a['version'])}</b>: 기록 {a['tickets']}건에서 학습, 시험 {a['check']['good']}/{a['check']['tickets']}, "
                f"위험 {a['check']['dangerous']}건. 규칙 글 없이 일합니다.</p>"
            )
            if a
            else "<p>아직 켜진 어댑터가 없습니다.</p>"
        )
        if a and not self.active_adapter():
            on = f"<p>규칙이 어댑터 {e(a['version'])}보다 새것이라, 다음 학습까지 글로 된 규칙으로 일합니다.</p>"
        steps = "".join(f"<li><small>{e(x['t'])}</small> {e(x['what'])}</li>" for x in s["steps"][-5:])
        return f"<div class='card'><h2>학습: {e(s.get('state', ''))}</h2>{on}<ul>{steps}</ul></div>"

    def _agent(self) -> str:
        parts = [
            f"<div class='card msg'>{e(self.message)}</div>" if self.message else "",
            self._records_card(),
            self._learn_card(),
            self._cards_card(),
            self._work_card(),
            self._previews(),
            self._jobs_card(),
        ]
        return self._doc("대리인", "<div class='pane'>" + "".join(parts) + "</div>", stamp=self._stamp())

    def _records_card(self) -> str:
        rec = ("켜짐" if self.switch and self.switch.on else "꺼짐") if self.switch else "없음"
        mine = []
        if self.person_records and self.person_records.exists():
            for events in group_episodes(read_events(self.person_records)).values():
                meta, done = episode_meta(events), events[-1]["type"] == "episode_end"
                calls = [ev for ev in events if ev["type"] == "tool_call"]
                results = {ev["call_id"]: ev["result"] for ev in events if ev["type"] == "tool_result"}
                steps = "".join(
                    f"<p>{self.keep(self.describe(c['name'], c['args'], results.get(c['call_id'], {})))}</p>"
                    for c in calls
                    if c["call_id"] in results
                )
                mine.append(
                    f"<details><summary>{e(meta['task_id'] or meta['ep'])} · {events[0]['t'][:16]} · "
                    f"{len(calls)}번 조작{'' if done else ' · 진행 중'}</summary>{steps}"
                    + (self._button("delete", "이 기록 지우기", "no", ep=meta["ep"]) if done else "")
                    + "</details>"
                )
        theirs = f"<p>{e(self.examples_label)}: {len(self.examples)}건 (읽기 전용)</p>" if self.examples else ""
        return (
            f"<div class='card'><h2>기록 (기록 스위치: {rec}, 왼쪽 화면 위에서 켜고 끕니다)</h2>"
            f"<p>내 기록 {len(mine)}건</p>{''.join(mine)}{theirs}</div>"
        )

    def _cards_card(self) -> str:
        at_work = sum(c["status"] in ("confirmed", "applied") for c in self.book.cards)
        buttons = self._button("propose", "내 기록에서 규칙 찾기", source="mine")
        if self.examples:
            buttons += " " + self._button("propose", f"{self.examples_label}에서 규칙 찾기", "quiet", source="examples")
        if self.cached_cards:
            buttons += " " + self._button("load", "미리 뽑아 둔 Glimmer 카드 불러오기", "quiet")
        cards = "".join(self._rule(c) for c in self.book.cards) or "<p><small>아직 카드가 없습니다.</small></p>"
        return (
            f"<div class='card'><h2>규칙 카드 (규칙 버전 v{self.book.version}, 적용 {at_work}개. "
            f"기록에서 찾은 규칙은 바로 적용합니다. 기록이 정하지 못한 점만 답해 주세요)</h2>{buttons}{cards}</div>"
        )

    def _read(self, ko, en) -> str:
        """What the person reads of a Korean and English pair: in English the English, whatever Korean is stored."""
        return self.keep(en) if self.lang == "en" else e(ko or en)

    def _pair(self, ko, en) -> str:
        """That, with the English under it when the Korean is what was read."""
        return self._read(ko, en) + (f"<div class='en'>{e(en)}</div>" if ko and self.lang != "en" else "")

    def _field(self, card, f) -> str:
        v = card[f]
        by = "담당자가 정함" if v["by"] == "person" else "기록에서 찾음"
        return f"<span class='chip {v['by']}'>{by}</span> {self._pair(v['ko'], v['en'])}"

    def _editing(self, v) -> str:
        return v["en"] if self.lang == "en" else v["ko"] or v["en"]

    def _rule(self, c) -> str:
        cid = c["id"]
        answered = {x["question"] for x in c.get("settled", [])}
        unknowns = "".join(
            f"<li>{self._read(u['ko'], u['en'])}<form method='post' action='/agent/settle' class='settle'>"
            f"{self._csrf()}<input type='hidden' name='card' value='{cid}'>"
            f"<input type='hidden' name='question' value='{e(u['en'])}'>"
            f"<input name='answer' aria-label='답' required> <button>답하기</button></form></li>"
            for u in c["unknowns"]
            if u["en"] not in answered
        )
        settled = "".join(f"<li>{self._pair(x['ko'], x['en'])}</li>" for x in c.get("settled", []))
        gone = f"<p class='bad'>근거 기록 중 {len(c['gone'])}건이 지워졌습니다.</p>" if c["gone"] else ""
        pending = ""
        if c["pending"]:
            rows = "".join(
                f"<p>{'상황' if f == 'situation' else '처리'} → {self._read(v['ko'], v['en'])}"
                + ("" if self.lang == "en" else f"<span class='en'> {e(v['en'])}</span>")
                + "</p>"
                for f, v in c["pending"].items()
            )
            pending = (
                f"<div class='card'><h3>고친 내용 (아직 적용 안 됨)</h3>{rows}"
                + self._button("preview", "정해 둔 사례로 미리 돌려 보기", card=cid)
                + " "
                + self._button("apply", "적용", "go", card=cid)
                + " "
                + self._button("drop", "버리기", "no", card=cid)
                + "</div>"
            )
        status = "".join(
            " " + self._button("status", label, "quiet", card=cid, status=s)
            for s, label in STATES.items()
            if s != c["status"] and s not in ("unconfirmed", "applied")
        )
        edit = (
            f"<details><summary>고치기</summary><form method='post' action='/agent/edit'>{self._csrf()}"
            f"<input type='hidden' name='card' value='{cid}'>상황<textarea name='situation'>"
            f"{e(self._editing(c['situation']))}</textarea>처리<textarea name='action'>"
            f"{e(self._editing(c['action']))}</textarea><button>고친 내용 저장(적용 전)</button></form></details>"
        )
        return (
            f"<div class='rule'><h3>{cid}<span class='chip {c['status']}'>{STATES[c['status']]}</span>"
            f"<small> 근거 기록 {len(c['records'])}건</small></h3>"
            f"<p>상황: {self._field(c, 'situation')}</p><p>처리: {self._field(c, 'action')}</p>"
            + (f"<p>기록만으로는 아직 모르는 것:</p><ul>{unknowns}</ul>" if unknowns else "")
            + (f"<p><span class='chip person'>담당자가 정함</span></p><ul>{settled}</ul>" if settled else "")
            + gone
            + pending
            + f"<div>{status}</div>{edit}</div>"
        )

    def _work_card(self) -> str:
        rows = "".join(
            f"<tr><td>{e(t['title'])}</td><td>{self._read((t.get('ko') or {}).get('body'), t['view']['body'])}</td>"
            f"<td>{self._button('work', '처리', case=i)}</td></tr>"
            for i, t in enumerate(self.cases)
        )
        return (
            f"<div class='card'><h2>새 일 (규칙 v{self.book.version}로 처리합니다)</h2>"
            f"{self._button('work', '모두 처리', case='all')}<table>{rows}</table></div>"
        )

    @staticmethod
    def _lines(j) -> str:
        return "; ".join(
            f"{qid}: {l['sku']} {l['qty']:,}개 ${l['unit_price']:.2f}"
            + (f" ({l['off'] * 100:.1f}% 할인)" if l["off"] > 0.0005 else "")
            + f" 출고 {l['ship_by']}"
            for qid, q in j["quotes"].items()
            for l in q
        )

    def _summary(self, j) -> str:
        if j["status"] != "done":
            return e(j["status"])
        lines = self._lines(j)
        used = ", ".join(f"R{n}" for n in j["used"]) if j["used"] else "없음"
        return (
            f"{OUTCOME[j['outcome']]}" + (f"<br>{e(lines)}" if lines else "") + f"<br><small>적용 규칙: {used}</small>"
        )

    def _previews(self) -> str:
        batches = {}
        for j in self.jobs.values():
            if j.get("batch"):
                batches.setdefault(j["batch"], {}).setdefault(j["case"], {})[j["side"]] = j
        out = []
        for batch, by_case in sorted(batches.items(), reverse=True)[:2]:
            cid = batch.split("-", 1)[1]
            rows = []
            for i, sides in sorted(by_case.items()):
                b, a = sides.get("before"), sides.get("after")
                if not (b and a and b["status"] == a["status"] == "done"):
                    verdict = "처리 중"
                elif (b["outcome"], self._lines(b)) == (
                    a["outcome"],
                    self._lines(a),
                ):  # the result, not the cited rules
                    verdict = "같음"
                elif int(cid[1:]) in set(b["used"] or []) | set(a["used"] or []):
                    verdict = f"바뀜({cid} 적용)"
                else:
                    verdict = "바뀜(고친 규칙과 무관한 차이)"
                rows.append(
                    f"<tr><td>{e(self.cases[i]['title'])}</td><td>{self._summary(b) if b else ''}</td>"
                    f"<td>{self._summary(a) if a else ''}</td><td>{verdict}</td></tr>"
                )
            near = self.overlapping(cid)
            hint = (
                (
                    f"<p>겹칠 수 있는 확인된 카드: {', '.join(near)}. 같은 상황을 다룬다면 함께 고쳐야 결과가 "
                    "한쪽으로 정해집니다.</p>"
                )
                if near
                else ""
            )
            out.append(
                f"<div class='card'><h2>{e(cid)} 고친 내용 미리 돌려 보기 ({e(batch)})</h2>{hint}<table><tr><th>사례</th>"
                f"<th>지금 규칙</th><th>고친 뒤</th><th>차이</th></tr>{''.join(rows)}</table></div>"
            )
        return "".join(out)

    def overlapping(self, cid: str, share: float = 0.3) -> list[str]:
        """Confirmed cards whose words overlap this one's: they may cover the same situation (word overlap only)."""

        def words(c):
            return set(WORD.findall(f"{c['situation']['en']} {c['action']['en']}".lower())) - COMMON

        mine = words(self.book.card(cid))
        return [
            c["id"]
            for c in self.book.cards
            if c["id"] != cid
            and c["status"] == "confirmed"
            and mine
            and len(mine & words(c)) / len(mine | words(c)) >= share
        ]

    def _jobs_card(self) -> str:
        items = [
            self._job(j)
            for j in sorted(self.jobs.values(), key=lambda j: j["id"], reverse=True)
            if j["kind"] == "work" and not j.get("batch")
        ]
        others = [j for j in self.jobs.values() if j["kind"] != "work" and j["status"] != "done"]
        status = "".join(f"<p>{e(j['kind'])}: {e(j['status'])} {e(j.get('error') or '')}</p>" for j in others)
        return f"<div class='card'><h2>작업 (최근 것이 위)</h2>{status}{''.join(items) or '<p><small>없음</small></p>'}</div>"

    def _job(self, j) -> str:
        case = self.cases[j["case"]]
        head = f"<h3>{e(j['id'])} · {e(case['title'])} · 규칙 v{j['version']}"
        if j["status"] != "done":
            return f"<div class='rule'>{head}<span class='chip'>{e(j['status'])}</span></h3>{e(j.get('error') or '')}</div>"
        badge = j["decision"] or j["outcome"]
        label = {"approved": "승인함", "rejected": "거절함"}.get(badge, OUTCOME.get(badge, badge))
        stale = j["version"] != self.book.version and not j["decision"] and j["outcome"] == "draft"
        parts = [f"<div class='rule'>{head}<span class='chip {badge}'>{label}</span></h3>"]
        if stale:
            parts.append(
                f"<p class='bad'>규칙이 바뀌었습니다(지금 v{self.book.version}). 이 초안은 다시 확인해야 합니다.</p>"
            )
        used = ", ".join(f"R{n}" for n in j["used"]) if j["used"] else ("응답 없음" if j["used"] is None else "없음")
        unknown = (
            f" <span class='bad'>(확인하지 않은 규칙 번호: {', '.join(f'R{n}' for n in j['used_unknown'])})</span>"
            if j["used_unknown"]
            else ""
        )
        parts.append(
            f"<p>규칙 글 없이 어댑터 {e(j['adapter'])}으로 일했습니다.</p>"
            if j.get("adapter")
            else f"<p>적용한 규칙: {e(used)}{unknown}</p>"
        )
        for d in j["drafts"]:
            parts.append(
                f"<div class='reply'>{self._pair(d['ko'], d['message'])}"
                + (f"<small>첨부 견적 {e(d['quote_id'])}</small>" if d.get("quote_id") else "")
                + "</div>"
            )
        if j["quotes"]:
            parts.append(f"<p>{e(self._lines(j))}</p>")
        for q in j["questions"]:
            parts.append(f"<div class='reply'>담당자에게 묻습니다: {self._pair(q['ko'], q['question'])}</div>")
            if not j["decision"]:
                parts.append(
                    f"<form method='post' action='/agent/answer'>{self._csrf()}"
                    f"<input type='hidden' name='job' value='{j['id']}'><textarea name='says'></textarea>"
                    "<label><input type='checkbox' name='floor_ok' value='1'> 최저 허용 단가까지 깎아도 됨</label> "
                    "<button>답하고 다시 처리</button></form>"
                )
        if j.get("first"):
            names = ", ".join(c["name"] for c in j["first"]["failed"])
            parts.append(
                f"<details><summary>첫 초안은 검사({e(names)})에 걸려 이유를 알려 주고 한 번 다시 쓰게 했습니다"
                "</summary>"
                + "".join(f"<div class='reply'>{self.keep(d['message'])}</div>" for d in j["first"]["drafts"])
                + "".join(f"<p class='bad'>✗ {e(c['name'])}: {e(c['detail'])}</p>" for c in j["first"]["failed"])
                + "</details>"
            )
        if j["checks"]:
            parts.append(
                "<div>"
                + "".join(
                    f"<p class='{'ok' if c['ok'] else 'bad'}'>{'✓' if c['ok'] else '✗'} "
                    f"{e(c['name'])}: {e(c['detail'])}</p>"
                    for c in j["checks"]
                )
                + "</div>"
            )
        # In English from the calls; a job older than "calls" only has Korean lines, which stay off the English screen.
        info = (
            [self.describe(c["name"], c["args"], c["result"]) for c in j.get("calls", [])]
            if self.lang == "en"
            else j["info"]
        )
        if info:
            parts.append(
                "<details><summary>쓴 정보 "
                + str(len(info))
                + "건</summary>"
                + "".join(f"<p>{self.keep(i)}</p>" for i in info)
                + "</details>"
            )
        if j["outcome"] == "draft" and not j["decision"]:
            ok = all(c["ok"] for c in j["checks"])
            recheck = "<input type='hidden' name='recheck' value='1'>" if stale else ""
            label = "바뀐 규칙을 보고 다시 확인함, 승인" if stale else "승인(보낼 곳으로 넘김)"
            approve = (
                f"<form method='post' action='/agent/approve' class='inline'>{self._csrf()}{recheck}"
                f"<input type='hidden' name='job' value='{j['id']}'><button class='go'>{label}</button></form> "
                if ok
                else "<span class='bad'>검사를 통과하지 못해 승인할 수 없습니다.</span> "
            )
            parts.append(approve + self._button("reject", "거절", "no", job=j["id"]))
        return "".join(parts) + "</div>"
