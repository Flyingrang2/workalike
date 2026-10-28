"""Web entrance to quote-desk/2 for a person who reads and writes Korean.

The desk, the customer, the records and what a model learns from stay in English. The page shows every
English text the person needs with a Korean translation made beforehand (the ticket, the earlier thread,
the customer's and the owner's fixed answers) and Korean labels. The person writes a reply, or a question
to the owner, in Korean; a translator (any command that reads Korean on stdin and prints English) turns
it into English, shown side by side. The English goes out only when its item codes, quote ids and numbers
match the Korean, and only when the person presses send. The record keeps the English call, exactly what
the desk saw, and as a private event the Korean original, the English and the translator's name, which
never reach training data. Text with no Hangul in it is sent as written, without the translator.
"""
from __future__ import annotations

import html
import json
import re
import secrets
import shlex
import subprocess
import threading

from .desk import Desk
from .envs import V2
from .record import Recorder
from .web import _num

FINAL = "Done."
HANGUL = re.compile(r"[가-힣]")
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december"]
# Item codes as people write them (MUG-15W, mug-15w, MUG15W, TSH-STD) and quote ids; never plain words.
CODE = re.compile(r"(?<![A-Za-z0-9-])(?:Q-\d+(?:-\d+)?|[A-Z]{2,4}-[A-Z0-9]{2,5}|[a-z]{2,4}-[a-z0-9]*\d[a-z0-9]*|[A-Z]{2,4}\d[A-Z0-9]{0,4})(?![A-Za-z0-9])")


def _numbers(text: str, english: bool) -> list[str]:
    t = text
    if english:
        t = re.sub(r"\b(" + "|".join(MONTHS + [m[:3] for m in MONTHS]) + r")\b\.?(?=\s*\d)",  # a date, not "may"
                   lambda m: str(next(i for i, k in enumerate(MONTHS, 1) if k.startswith(m.group(1).lower()[:3]))),
                   t, flags=re.IGNORECASE)
    t = CODE.sub(" ", t).replace(",", "")
    return sorted(re.findall(r"\d+(?:\.\d+)?", t))


def translation_problems(ko: str, en: str) -> list[str]:
    """Why the English cannot go out for the Korean: a code, quote id or number that differs."""
    problems = []
    a, b = sorted(c.upper() for c in CODE.findall(ko)), sorted(c.upper() for c in CODE.findall(en))
    if a != b:
        problems.append(f"품번이나 견적 번호가 다릅니다: 한국어 {a}, 영어 {b}")
    x, y = _numbers(ko, False), _numbers(en, True)
    if x != y:
        problems.append(f"숫자가 다릅니다: 한국어 {x}, 영어 {y}")
    if not en.strip():
        problems.append("번역이 비어 있습니다")
    return problems


def command_translator(cmd: str):
    """A translator from a shell command: Korean on stdin, English on stdout (or JSON {"en", "translator"})."""
    def translate(ko: str) -> tuple[str, str]:
        out = subprocess.run(shlex.split(cmd), input=ko, capture_output=True, text=True, timeout=180, check=True).stdout
        try:
            data = json.loads(out)
            return data["en"].strip(), data.get("translator", cmd)
        except (json.JSONDecodeError, KeyError, TypeError):
            return out.strip(), cmd
    return translate


class Korean:
    """Translations made before the person starts: per ticket, and the fixed answers every ticket shares."""

    def __init__(self, data: dict):
        self.tickets = data["tickets"]
        fixed = data["fixed"]
        self.owner = dict(zip(fixed["en"]["owner"], fixed["ko"]["owner"]))
        self.defaults = list(zip(fixed["en"]["customer_defaults"], fixed["ko"]["customer_defaults"]))

    def answer(self, task_id: str, english: str, state: dict) -> str | None:
        """The Korean of a customer's or the owner's fixed answer, or None if it was not translated."""
        if english in self.owner:
            return self.owner[english]
        ko = self.tickets.get(task_id, {}).get("customer", {})
        en = next((c["says"] for c in state["customers"].values()), {})
        pairs = {en[k]: ko[k] for k in en if k in ko}
        if english in pairs:
            return pairs[english]
        (again_en, again_ko) = self.defaults[2]
        if english.startswith(again_en) and english[len(again_en):] in pairs:
            return again_ko + pairs[english[len(again_en):]]
        for en_t, ko_t in self.defaults[:2]:
            m = re.fullmatch(re.escape(en_t).replace(re.escape("{qid}"), r"(Q-[\w-]+)"), english)
            if m:
                return ko_t.replace("{qid}", m.group(1))
        return None


def _money(v) -> str:
    return f"${v:,.2f}" if isinstance(v, (int, float)) else str(v)


NOTICE = re.compile(r"(\S+) at ([\d.]+) is ([\d.]+)% under list \(([\d.]+)\); (.*)")
NOTICE_KO = {"within your discount authority.": "할인 권한 안입니다.",
             "beyond your discount authority; the owner agreed.": "할인 권한을 넘지만 사장님이 동의했습니다.",
             "beyond your discount authority, owner approval needed.": "할인 권한을 넘습니다. 사장님 승인이 필요합니다."}
ERRORS = {"unknown_quote": "그런 견적 번호가 없습니다", "unknown_ticket": "티켓 번호가 다릅니다",
          "below_margin_floor": "최저 허용 단가보다 낮습니다", "invalid_date": "날짜 형식이 틀렸습니다(YYYY-MM-DD)",
          "ship_by_in_past": "출고일이 오늘보다 앞입니다", "invalid_value": "수량이나 가격이 0 이하입니다",
          "no_lines": "견적 줄이 없습니다", "empty_message": "내용이 비어 있습니다",
          "too_many_lines": "견적 줄이 너무 많습니다", "too_many_lines_for_item": "한 품목은 두 줄까지만 나눌 수 있습니다"}


def describe_ko(name: str, args: dict, r: dict, ko_answer=None) -> str:
    """A result in the words a Korean-speaking clerk would read."""
    if "error" in r:
        if r["error"] == "unknown_sku":
            similar = ", ".join(f"{s['sku']}({s['name']}, 재고 {s['on_hand']:,}개)" for s in r.get("similar", []))
            return f"{r['sku']}는 카탈로그에 없습니다." + (f" 비슷한 품번: {similar}." if similar else "")
        if r["error"] == "stock_short_for_ship_date":
            return f"{r['sku']}는 {r['ship_by']}까지 {r['available_by_then']:,}개만 출고할 수 있습니다."
        if r["error"] == "invalid_arguments":
            return f"입력이 올바르지 않습니다({r['detail']})."
        return f"실패: {ERRORS.get(r['error'], r['error'])}" + (f" ({r['sku']})" if r.get("sku") else "") + "."
    if name == "lookup_quote":
        status = "아직 유효합니다" if r["status"] == "approved" else "만료됐습니다"
        lines = "; ".join(f"{l['sku']}({l['name']}) {l['qty']:,}개, 승인 단가 {_money(l['unit_price'])}" for l in r["lines"])
        return f"견적 {r['quote_id']}: 유효기한 {r['valid_until']}, {status}. {lines}."
    if name == "get_cost":
        return f"{r['sku']}({r['name']}) 원가 {_money(r['unit_cost'])} ({r['as_of']} 기준)."
    if name == "check_stock":
        text = f"{r['sku']}({r['name']}) {r['qty']:,}개 필요, 재고 {r['on_hand']:,}개: "
        if r["enough"]:
            return text + "충분합니다."
        restock = (f"재입고 {r['restock_date']}에 {r['restock_qty']:,}개" if r.get("restock_date") else "재입고 예정 없음")
        subs = ", ".join(f"{s['sku']}({s['name']}, 재고 {s['on_hand']:,}개)" for s in r["substitutes"]) or "없음"
        return text + f"부족합니다. {restock}. 재고 있는 대체품: {subs}."
    if name == "calc_price":
        return (f"{r['sku']} {r['qty']:,}개, 원가 {_money(r['unit_cost'])}: 가격표 단가 {_money(r['unit_price'])}, "
                f"합계 {_money(r['line_total'])}, 최저 허용 단가 {_money(r['floor_unit_price'])}.")
    if name == "create_quote":
        lines = "; ".join(f"{l['sku']} {l['qty']:,}개 단가 {_money(l['unit_price'])} 출고 {l['ship_by']}" for l in r["lines"])
        notes = []
        for n in r.get("notices", []):
            m = NOTICE.fullmatch(n)
            notes.append(f"{m.group(1)} 단가 {m.group(2)}는 가격표({m.group(4)})보다 {m.group(3)}% 낮습니다. "
                         f"{NOTICE_KO.get(m.group(5), m.group(5))}" if m else n)
        return f"견적 초안 {r['quote_id']}: {lines}. 합계 {_money(r['total'])}." + "".join(" " + n for n in notes)
    if name == "ask_owner":
        return f"사장님 답: {ko_answer or r['owner_reply']}"
    if name == "send_reply":
        attached = f"(견적 {args['quote_id']} 첨부)" if args.get("quote_id") else ""
        return f"답장 #{r['reply_no']}{attached}을 보냈습니다. 손님 답: {ko_answer or r['customer_reply']}"
    return json.dumps(r, ensure_ascii=False)


def form_to_call(form: dict[str, str], ticket_id: str) -> tuple[str, dict]:
    """One submitted lookup or quote form to (tool, args). Replies and owner questions go through a draft."""
    tool = form.get("tool", "")
    f = {k: v.strip() for k, v in form.items()}
    if tool == "lookup_quote":
        return tool, {"quote_id": f.get("quote_id", "")}
    if tool == "get_cost":
        return tool, {"sku": f.get("sku", "")}
    if tool == "check_stock":
        return tool, {"sku": f.get("sku", ""), "qty": _num(f.get("qty", ""), int)}
    if tool == "calc_price":
        return tool, {"sku": f.get("sku", ""), "qty": _num(f.get("qty", ""), int),
                      "unit_cost": _num(f.get("unit_cost", ""), float)}
    if tool == "create_quote":
        lines = [{"sku": f[f"sku{i}"], "qty": _num(f.get(f"qty{i}", ""), int),
                  "unit_price": _num(f.get(f"price{i}", ""), float), "ship_by": f.get(f"ship{i}", "")}
                 for i in (1, 2, 3, 4) if f.get(f"sku{i}")]
        return tool, {"ticket_id": ticket_id, "lines": lines}
    return tool, {}


CSS = """
*{box-sizing:border-box}body{margin:0;font:15px/1.45 system-ui,"Apple SD Gothic Neo","Malgun Gothic",sans-serif;color:#15181d;background:#eef0f3}
header{height:48px;display:flex;align-items:center;gap:20px;padding:0 16px;background:#1f2a37;color:#fff}
header b{font-size:17px}header form{margin-left:auto}header span{opacity:.85}
main{display:grid;grid-template-columns:480px 1fr;gap:12px;padding:12px;height:calc(100vh - 48px)}
.col{display:flex;flex-direction:column;gap:12px;min-height:0}
.card{background:#fff;border:1px solid #cfd5dd;border-radius:6px;padding:10px 12px}
h2{font-size:13px;margin:0 0 6px;letter-spacing:.02em;color:#44505e}
.ticket{flex:0 1 auto;max-height:52vh;overflow:auto}.ticket .body{white-space:pre-wrap;margin:4px 0 8px}
.facts{background:#f6f7f9;border-radius:4px;padding:6px 8px;margin:0 0 8px;font-size:14px}
.en{color:#5a6573;font-size:13px;white-space:pre-wrap}details{margin-top:6px}summary{cursor:pointer;color:#2f4a6d;font-size:13px}
.log{flex:1;overflow:auto;min-height:140px}.log p{margin:0 0 8px;padding-bottom:6px;border-bottom:1px solid #eee}
.log small{color:#5a6573}.cust{background:#f3f7ff;border-left:3px solid #2f4a6d;padding:4px 8px;display:block;margin-top:2px}
.err{color:#a40000}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;overflow:auto;align-content:start}.wide{grid-column:1/3}
label{display:inline-block;margin:2px 8px 2px 0;font-size:14px}input,textarea{font:inherit;padding:4px 6px;border:1px solid #9aa5b1;border-radius:4px}
input{width:120px}input.short{width:84px}input.date{width:118px}textarea{width:100%;height:70px}
button{font:inherit;padding:5px 12px;border-radius:4px;border:1px solid #1f2a37;background:#2f4a6d;color:#fff;cursor:pointer}
button.quiet{background:#fff;color:#2f4a6d}.finish{background:#1a7f37;border-color:#1a7f37}.row{display:flex;gap:6px;margin:3px 0}
.draft{border-color:#2f4a6d;background:#fbfcff}.problem{color:#a40000;font-size:14px}
"""


class KoreanDesk:
    """Serves one quote-desk/2 ticket at a time to a Korean-speaking person; Finish moves to the next."""

    def __init__(self, sessions: list[dict], recorder: Recorder, korean: Korean, translate):
        self.queue = list(sessions)
        self.recorder = recorder
        self.korean = korean
        self.translate = translate
        self.token = secrets.token_urlsafe(16)
        self.lock = threading.Lock()
        self.current, self.desk, self.log, self.finished, self.draft = None, None, [], [], None
        self._next()

    def _next(self) -> None:
        self.current = self.queue.pop(0) if self.queue else None
        self.log, self.draft = [], None
        if self.current:
            s = self.current
            self.recorder.start(s["episode"], s["view"], task_id=s.get("task_id"), actor=s.get("actor"),
                                entrance="web-ko", env=V2.version)
            self.desk = Desk(s["state"], self.recorder, env=V2)

    def act(self, form: dict[str, str]) -> None:
        name, args = form_to_call(form, self.current["view"]["ticket_id"])
        self.log.insert(0, self.desk.call_many([(name, args)])[0])

    def make_draft(self, form: dict[str, str]) -> None:
        ko = form.get("text", "").strip()
        kind = "owner" if form.get("kind") == "owner" else "reply"
        if HANGUL.search(ko):
            try:
                en, who = self.translate(ko)
            except Exception as e:  # the person sees it and can try again
                en, who = "", f"translator failed: {type(e).__name__}"
        else:
            en, who = ko, "none (written in English)"
        self.draft = {"kind": kind, "ko": ko, "en": en, "translator": who, "quote_id": form.get("quote_id", "").strip(),
                      "problems": translation_problems(ko, en) if ko else ["내용이 비어 있습니다"]}

    def send(self) -> None:
        d = self.draft
        if not d or d["problems"]:
            return
        ticket = self.current["view"]["ticket_id"]
        self.recorder.event("translation", "private", kind=d["kind"], ko=d["ko"], en=d["en"], translator=d["translator"])
        if d["kind"] == "owner":
            name, args = "ask_owner", {"ticket_id": ticket, "question": d["en"]}
        else:
            name, args = "send_reply", {"ticket_id": ticket, "message": d["en"]}
            if d["quote_id"]:
                args["quote_id"] = d["quote_id"]
        self.log.insert(0, self.desk.call_many([(name, args)])[0])
        self.draft = None

    def finish(self, status: str = "done", final: str | None = FINAL) -> None:
        if final is not None:
            self.recorder.final(final)
        self.recorder.end(status)
        self.finished.append(self.current.get("task_id"))
        self._next()

    def post(self, path: str, form: dict[str, str]) -> None:
        with self.lock:
            if not self.current:
                return
            if path == "/act":
                self.act(form)
            elif path == "/draft":
                self.make_draft(form)
            elif path == "/send":
                self.send()
            elif path == "/discard":
                self.draft = None
            elif path == "/finish":
                self.finish()

    def status(self) -> dict:
        return {"task_id": self.current.get("task_id") if self.current else None,
                "episode": self.current["episode"] if self.current else None, "finished": self.finished,
                "left": len(self.queue) + (1 if self.current else 0)}

    def _ticket(self) -> str:
        e = html.escape
        s, v = self.current, self.current["view"]
        ko = self.korean.tickets.get(s.get("task_id"), {})
        pct = v.get("authority", 0.0)
        authority = ("할인 권한 없음. 가격표보다 낮게 주려면 먼저 사장님께 여쭤야 합니다." if pct <= 0 else
                     f"가격표에서 {pct * 100:.0f}%까지는 직접 깎아 줄 수 있습니다. 그 이상은 사장님께 여쭙니다. "
                     "최저 허용 단가 아래로는 안 됩니다.")
        sent = ko.get("sent_so_far") or (v.get("sent_so_far") and e(v["sent_so_far"])) or "아직 없음"
        thread_ko = ko.get("thread", [])
        thread = "".join(
            f"<p><small>{e(m['date'])}, {'손님' if m['who'] == 'customer' else '가게'}:</small><br>"
            f"{e(thread_ko[i]['text'] if i < len(thread_ko) else m['text'])}</p>" for i, m in enumerate(v.get("thread", [])))
        english = (f"From: {v['from']}\n\n" + "".join(f"{m['date']}, {m['who']}: {m['text']}\n" for m in v.get("thread", []))
                   + f"\n{v['body']}")
        return (f"<div class='card ticket'><h2>티켓 {e(v['ticket_id'])} · {e(ko.get('who', v['from']))}</h2>"
                f"<div class='facts'>{e(authority)}<br>지금까지 이 손님에게 보낸 것: {e(sent)}</div>"
                f"<h2>새 메일</h2><div class='body'>{e(ko.get('body', v['body']))}</div>"
                + (f"<h2>앞서 오간 메일</h2>{thread}" if thread else "")
                + f"<details><summary>영어 원문 보기</summary><div class='en'>{e(english)}</div></details></div>")

    def _log(self) -> str:
        e = html.escape
        items = []
        state, task_id = self.current["state"], self.current.get("task_id")
        for c in self.log:
            r = c["result"]
            said = r.get("customer_reply") or r.get("owner_reply")
            ko = self.korean.answer(task_id, said, state) if said else None
            text = describe_ko(c["name"], c["args"], r, ko)
            if said and "error" not in r:
                head, _, _ = text.partition(": " if c["name"] == "ask_owner" else "손님 답: ")
                lead = "사장님 답" if c["name"] == "ask_owner" else head + "손님 답"
                text_html = (f"{e(lead)}:<span class='cust'>{e(ko or said)}"
                             + (f"<br><span class='en'>{e(said)}</span>" if ko else "") + "</span>")
            else:
                text_html = e(text)
            sent = c["args"].get("message") or c["args"].get("question")
            extra = f"<br><span class='en'>보낸 영어: {e(sent)}</span>" if sent else ""
            items.append(f"<p class='{'err' if 'error' in r else ''}'><small>{e(c['name'])}</small><br>{text_html}{extra}</p>")
        return "".join(items) or "<p><small>아직 한 일이 없습니다.</small></p>"

    def page(self) -> str:
        e = html.escape
        head = (f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>견적 데스크</title>"
                f"<style>{CSS}</style></head><body>")
        if not self.current:
            return head + "<header><b>견적 데스크</b></header><main><div class='card'>남은 티켓이 없습니다. 수고하셨습니다.</div></main>"
        v = self.current["view"]
        token = f"<input type='hidden' name='csrf' value='{self.token}'>"

        def inp(name, label, cls=""):
            return f"<label>{label} <input name='{name}' class='{cls}' autocomplete='off'></label>"

        def form(action, buttons, body, cls="card"):
            b = "".join(f"<button name='tool' value='{t}'>{lab}</button> " for t, lab in buttons)
            return f"<form method='post' action='{action}' class='{cls}'>{token}{body}<div>{b}</div></form>"

        rows = "".join(f"<div class='row'>{inp(f'sku{i}', f'품번 {i}')}{inp(f'qty{i}', '수량', 'short')}"
                       f"{inp(f'price{i}', '단가', 'short')}{inp(f'ship{i}', '출고일', 'date')}</div>" for i in (1, 2, 3, 4))
        draft = ""
        if self.draft:
            d = self.draft
            what = "사장님께 보낼 질문" if d["kind"] == "owner" else "손님에게 보낼 답장" + (
                f" (견적 {d['quote_id']} 첨부)" if d["quote_id"] else "")
            problems = "".join(f"<p class='problem'>{e(p)}</p>" for p in d["problems"])
            send = "" if d["problems"] else f"<form method='post' action='/send' style='display:inline'>{token}<button>이대로 보내기</button></form> "
            draft = (f"<div class='card draft wide'><h2>{e(what)}: 영어로 옮긴 것을 확인하십시오</h2>"
                     f"<div>{e(d['ko'])}</div><div class='en' style='margin:6px 0'>{e(d['en'])}</div>{problems}"
                     f"{send}<form method='post' action='/discard' style='display:inline'>{token}"
                     f"<button class='quiet'>다시 쓰기</button></form></div>")
        forms = (
            draft
            + form("/act", [("lookup_quote", "견적 찾기")], "<h2>지난 견적</h2>" + inp("quote_id", "견적 번호"))
            + form("/act", [("get_cost", "원가 보기"), ("check_stock", "재고 보기")],
                   "<h2>품목</h2>" + inp("sku", "품번") + inp("qty", "수량", "short"))
            + form("/act", [("calc_price", "가격 계산")],
                   "<h2>가격 계산</h2>" + inp("sku", "품번") + inp("qty", "수량", "short") + inp("unit_cost", "원가", "short"),
                   "card wide")
            + form("/act", [("create_quote", "견적 만들기")],
                   "<h2>견적 만들기 (줄마다 출고일, YYYY-MM-DD)</h2>" + rows, "card wide")
            + form("/draft", [("reply", "영어로 옮기기")],
                   "<h2>손님에게 답장 (한국어로 쓰십시오)</h2><input type='hidden' name='kind' value='reply'>"
                   "<textarea name='text'></textarea>" + inp("quote_id", "첨부할 견적 번호"), "card wide")
            + form("/draft", [("owner", "영어로 옮기기")],
                   "<h2>사장님께 여쭙기 (손님은 보지 않습니다)</h2><input type='hidden' name='kind' value='owner'>"
                   "<textarea name='text' style='height:48px'></textarea>", "card wide")
        )
        left = len(self.queue) + 1
        return (head + f"<header><b>견적 데스크</b><span>오늘 {e(v['today'])}</span><span>이 티켓 포함 {left}건 남음</span>"
                f"<form method='post' action='/finish'>{token}<button class='finish'>이 티켓 끝내기</button></form></header>"
                f"<main><div class='col'>{self._ticket()}<div class='card log'><h2>한 일 (최근 것이 위)</h2>{self._log()}</div></div>"
                f"<div class='grid'>{forms}</div></main></body></html>")
