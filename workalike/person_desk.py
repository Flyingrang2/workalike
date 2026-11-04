"""Web entrance to quote-desk/2 for a person who reads and writes Korean.

The desk, the customer, the records and what a model learns from stay in English. The page shows every
English text the person needs with a Korean translation made beforehand (the ticket, the earlier thread,
the customer's and the owner's fixed answers) and Korean labels. The person writes a reply, or a question
to the owner, in Korean; a translator (any command that reads Korean on stdin and prints English) turns
it into English, shown side by side. The English goes out only when its item codes, quote ids and numbers
match the Korean, and only when the person presses send. The record keeps the English call, exactly what
the desk saw, and as a private event the Korean original, the English and the translator's name, which
never reach training data. Text with no Hangul in it is sent as written, without the translator.

With lang="en" (a demo for people who read English) the same desk has no translation at all: the person reads the mail
as it came and what they type goes out as written (AsWritten, as_written), and every page passes workalike.lang.
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
from .lang import boundary
from .record import Recorder
from .web import _num

FINAL = "Done."
HANGUL = re.compile(r"[가-힣]")
MONTHS = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]
# Item codes as people write them (MUG-15W, mug-15w, MUG15W, TSH-STD) and quote ids; never plain words.
CODE = re.compile(
    r"(?<![A-Za-z0-9-])(?:Q-\d+(?:-\d+)?|[A-Z]{2,4}-[A-Z0-9]{2,5}|[a-z]{2,4}-[a-z0-9]*\d[a-z0-9]*|[A-Z]{2,4}\d[A-Z0-9]{0,4})(?![A-Za-z0-9])"
)


def _numbers(text: str, english: bool) -> list[str]:
    t = text
    if english:
        t = re.sub(
            r"\b(" + "|".join(MONTHS + [m[:3] for m in MONTHS]) + r")\b\.?(?=\s*\d)",  # a date, not "may"
            lambda m: str(next(i for i, k in enumerate(MONTHS, 1) if k.startswith(m.group(1).lower()[:3]))),
            t,
            flags=re.IGNORECASE,
        )
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
        if english.startswith(again_en) and english[len(again_en) :] in pairs:
            return again_ko + pairs[english[len(again_en) :]]
        for en_t, ko_t in self.defaults[:2]:
            m = re.fullmatch(re.escape(en_t).replace(re.escape("{qid}"), r"(Q-[\w-]+)"), english)
            if m:
                return ko_t.replace("{qid}", m.group(1))
        return None


class AsWritten:
    """In place of Korean for a person who reads the English mail as it came: nothing has a translation."""

    tickets: dict = {}

    def answer(self, task_id: str, english: str, state: dict) -> None:
        return None


def as_written(text: str) -> tuple[str, str]:
    """In place of the translator, for the same person."""
    return text, "none (written in English)"


def _money(v) -> str:
    return f"${v:,.2f}" if isinstance(v, (int, float)) else str(v)


NOTICE = re.compile(r"(\S+) at ([\d.]+) is ([\d.]+)% under list \(([\d.]+)\); (.*)")
NOTICE_KO = {
    "within your discount authority.": "할인 권한 안입니다.",
    "beyond your discount authority; the owner agreed.": "할인 권한을 넘지만 사장님이 동의했습니다.",
    "beyond your discount authority, owner approval needed.": "할인 권한을 넘습니다. 사장님 승인이 필요합니다.",
}
ERRORS = {
    "unknown_quote": "그런 견적 번호가 없습니다",
    "unknown_ticket": "티켓 번호가 다릅니다",
    "below_margin_floor": "최저 허용 단가보다 낮습니다",
    "invalid_date": "날짜 형식이 틀렸습니다(YYYY-MM-DD)",
    "ship_by_in_past": "출고일이 오늘보다 앞입니다",
    "invalid_value": "수량이나 가격이 0 이하입니다",
    "no_lines": "견적 줄이 없습니다",
    "empty_message": "내용이 비어 있습니다",
    "too_many_lines": "견적 줄이 너무 많습니다",
    "too_many_lines_for_item": "한 품목은 두 줄까지만 나눌 수 있습니다",
}


def ship_when_ko(r: dict) -> str:
    """When a short item can ship, the line a person asked for (10-29: a clerk stopped not knowing the date)."""
    now, short = r["on_hand"], r["qty"] - r["on_hand"]
    if not r.get("restock_date"):
        return f"출고 가능: 지금 {now:,}개뿐이고, 나머지 {short:,}개는 재입고 예정이 없어 날짜를 약속할 수 없습니다."
    if r["restock_qty"] >= short:
        return f"출고 가능: 지금 {now:,}개, 나머지 {short:,}개는 재입고일 {r['restock_date']}부터."
    return (
        f"출고 가능: 지금 {now:,}개, 재입고일 {r['restock_date']}부터 {r['restock_qty']:,}개 더. "
        f"재입고 뒤에도 {short - r['restock_qty']:,}개가 모자랍니다."
    )


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
    if name == "lookup_quote" and r["status"] in ("issued", "draft"):
        said = "이 창구에서 낸 견적입니다" if r["status"] == "issued" else "아직 보내지 않은 초안입니다"
        lines = "; ".join(f"{l['sku']}({l['name']}) {l['qty']:,}개, 단가 {_money(l['unit_price'])}" for l in r["lines"])
        return f"견적 {r['quote_id']}: {said}. {lines}."
    if name == "lookup_quote":
        status = "아직 유효합니다" if r["status"] == "approved" else "만료됐습니다"
        lines = "; ".join(
            f"{l['sku']}({l['name']}) {l['qty']:,}개, 승인 단가 {_money(l['unit_price'])}" for l in r["lines"]
        )
        return f"견적 {r['quote_id']}: 유효기한 {r['valid_until']}, {status}. {lines}."
    if name == "search_catalog":
        found = ", ".join(f"{i['sku']}({i['name']})" for i in r["items"]) or "없음"
        return f"'{r['query']}' 검색: {found}." + (f" 그 밖에 {r['more']}개." if r["more"] else "")
    if name == "get_cost":
        return f"{r['sku']}({r['name']}) 원가 {_money(r['unit_cost'])} ({r['as_of']} 기준)."
    if name == "check_stock":
        text = f"{r['sku']}({r['name']}) {r['qty']:,}개 필요, 재고 {r['on_hand']:,}개: "
        if r["enough"]:
            return text + "충분합니다. 지금 재고로 바로 출고할 수 있습니다."
        restock = (
            f"재입고 {r['restock_date']}에 {r['restock_qty']:,}개" if r.get("restock_date") else "재입고 예정 없음"
        )
        subs = ", ".join(f"{s['sku']}({s['name']}, 재고 {s['on_hand']:,}개)" for s in r["substitutes"]) or "없음"
        return text + f"부족합니다. {restock}. {ship_when_ko(r)} 재고 있는 대체품: {subs}."
    if name == "calc_price":
        return (
            f"{r['sku']} {r['qty']:,}개, 원가 {_money(r['unit_cost'])}: 가격표 단가 {_money(r['unit_price'])}, "
            f"합계 {_money(r['line_total'])}, 최저 허용 단가 {_money(r['floor_unit_price'])}."
        )
    if name == "create_quote":
        lines = "; ".join(
            f"{l['sku']} {l['qty']:,}개 단가 {_money(l['unit_price'])} 출고 {l['ship_by']}" for l in r["lines"]
        )
        notes = []
        for n in r.get("notices", []):
            m = NOTICE.fullmatch(n)
            notes.append(
                f"{m.group(1)} 단가 {m.group(2)}는 가격표({m.group(4)})보다 {m.group(3)}% 낮습니다. "
                f"{NOTICE_KO.get(m.group(5), m.group(5))}"
                if m
                else n
            )
        return f"견적 초안 {r['quote_id']}: {lines}. 합계 {_money(r['total'])}." + "".join(" " + n for n in notes)
    if name == "ask_owner":
        return f"사장님 답: {ko_answer or r['owner_reply']}"
    if name == "send_reply":
        attached = f"(견적 {args['quote_id']} 첨부)" if args.get("quote_id") else ""
        return f"답장 #{r['reply_no']}{attached}을 보냈습니다. 손님 답: {ko_answer or r['customer_reply']}"
    return json.dumps(r, ensure_ascii=False)


ERRORS_EN = {
    "unknown_quote": "no such quote id",
    "unknown_ticket": "wrong ticket id",
    "below_margin_floor": "under the floor price",
    "invalid_date": "bad date format (YYYY-MM-DD)",
    "ship_by_in_past": "the ship date is before today",
    "invalid_value": "quantity or price is 0 or less",
    "no_lines": "the quote has no lines",
    "empty_message": "the message is empty",
    "too_many_lines": "too many quote lines",
    "too_many_lines_for_item": "one item can take two lines at most",
}


def ship_when_en(r: dict) -> str:
    """ship_when_ko in English."""
    now, short = r["on_hand"], r["qty"] - r["on_hand"]
    if not r.get("restock_date"):
        return f"Can ship: {now:,} now; the other {short:,} have no restock planned, so no date can be promised."
    if r["restock_qty"] >= short:
        return f"Can ship: {now:,} now, the other {short:,} from the restock on {r['restock_date']}."
    return (
        f"Can ship: {now:,} now, {r['restock_qty']:,} more from the restock on {r['restock_date']}. "
        f"After the restock {short - r['restock_qty']:,} are still missing."
    )


def describe_en(name: str, args: dict, r: dict, answer=None) -> str:
    """describe_ko in English, written here because these lines are mostly other people's data: item names, search
    terms and replies that a replacement over the page could not tell from the screen's own words."""
    if "error" in r:
        if r["error"] == "unknown_sku":
            similar = ", ".join(f"{s['sku']} ({s['name']}, {s['on_hand']:,} on hand)" for s in r.get("similar", []))
            return f"{r['sku']} is not in the catalog." + (f" Similar: {similar}." if similar else "")
        if r["error"] == "stock_short_for_ship_date":
            return f"{r['sku']}: only {r['available_by_then']:,} can ship by {r['ship_by']}."
        if r["error"] == "invalid_arguments":
            return f"Invalid input ({r['detail']})."
        return f"Failed: {ERRORS_EN.get(r['error'], r['error'])}" + (f" ({r['sku']})" if r.get("sku") else "") + "."
    if name == "lookup_quote" and r["status"] in ("issued", "draft"):
        said = "issued from this desk" if r["status"] == "issued" else "a draft, not sent yet"
        lines = "; ".join(
            f"{l['sku']} ({l['name']}) x {l['qty']:,}, unit price {_money(l['unit_price'])}" for l in r["lines"]
        )
        return f"Quote {r['quote_id']}: {said}. {lines}."
    if name == "lookup_quote":
        status = "still valid" if r["status"] == "approved" else "expired"
        lines = "; ".join(
            f"{l['sku']} ({l['name']}) x {l['qty']:,}, approved unit price {_money(l['unit_price'])}"
            for l in r["lines"]
        )
        return f"Quote {r['quote_id']}: valid until {r['valid_until']}, {status}. {lines}."
    if name == "search_catalog":
        found = ", ".join(f"{i['sku']} ({i['name']})" for i in r["items"]) or "nothing"
        return f"Search '{r['query']}': {found}." + (f" {r['more']} more." if r["more"] else "")
    if name == "get_cost":
        return f"{r['sku']} ({r['name']}): unit cost {_money(r['unit_cost'])} as of {r['as_of']}."
    if name == "check_stock":
        text = f"{r['sku']} ({r['name']}), {r['qty']:,} needed, {r['on_hand']:,} on hand: "
        if r["enough"]:
            return text + "enough. It can ship from stock right away."
        restock = (
            f"Restock on {r['restock_date']}: {r['restock_qty']:,}" if r.get("restock_date") else "No restock planned"
        )
        subs = ", ".join(f"{s['sku']} ({s['name']}, {s['on_hand']:,} on hand)" for s in r["substitutes"]) or "none"
        return text + f"short. {restock}. {ship_when_en(r)} In-stock substitutes: {subs}."
    if name == "calc_price":
        return (
            f"{r['sku']} x {r['qty']:,} at cost {_money(r['unit_cost'])}: list unit price {_money(r['unit_price'])}, "
            f"line total {_money(r['line_total'])}, floor unit price {_money(r['floor_unit_price'])}."
        )
    if name == "create_quote":
        lines = "; ".join(
            f"{l['sku']} x {l['qty']:,} at {_money(l['unit_price'])}, ships {l['ship_by']}" for l in r["lines"]
        )
        return f"Draft quote {r['quote_id']}: {lines}. Total {_money(r['total'])}." + "".join(
            " " + n for n in r.get("notices", [])
        )
    if name == "ask_owner":
        return f"Owner replied: {answer or r['owner_reply']}"
    if name == "send_reply":
        attached = f" with quote {args['quote_id']}" if args.get("quote_id") else ""
        return f"Reply #{r['reply_no']}{attached} sent. Customer replied: {answer or r['customer_reply']}"
    return json.dumps(r, ensure_ascii=False)


def form_to_call(form: dict[str, str], ticket_id: str) -> tuple[str, dict]:
    """One submitted lookup or quote form to (tool, args). Replies and owner questions go through a draft."""
    tool = form.get("tool", "")
    f = {k: v.strip() for k, v in form.items()}
    if tool == "lookup_quote":
        return tool, {"quote_id": f.get("quote_id", "")}
    if tool == "search_catalog":
        return tool, {"query": f.get("query", "")}
    if tool == "get_cost":
        return tool, {"sku": f.get("sku", "")}
    if tool == "check_stock":
        return tool, {"sku": f.get("sku", ""), "qty": _num(f.get("qty", ""), int)}
    if tool == "calc_price":
        return tool, {
            "sku": f.get("sku", ""),
            "qty": _num(f.get("qty", ""), int),
            "unit_cost": _num(f.get("unit_cost", ""), float),
        }
    if tool == "create_quote":
        lines = [
            {
                "sku": f[f"sku{i}"],
                "qty": _num(f.get(f"qty{i}", ""), int),
                "unit_price": _num(f.get(f"price{i}", ""), float),
                "ship_by": f.get(f"ship{i}", ""),
            }
            for i in (1, 2, 3, 4)
            if f.get(f"sku{i}")
        ]
        return tool, {"ticket_id": ticket_id, "lines": lines}
    return tool, {}


CSS = """
*{box-sizing:border-box}body{margin:0;font:15px/1.45 system-ui,"Apple SD Gothic Neo","Malgun Gothic",sans-serif;color:#22211f;background:#f4f1e6}
header{height:48px;display:flex;align-items:center;gap:14px;padding:0 12px;font-size:13px;white-space:nowrap;background:#22211f;color:#fff}
header b{font-size:15px}header button{padding:3px 9px;font-size:13px}header form{margin-left:auto}header span{opacity:.85}
main{display:grid;grid-template-columns:480px 1fr;gap:12px;padding:12px;height:calc(100vh - 48px)}
.col{display:flex;flex-direction:column;gap:12px;min-height:0}
.card{background:#fbfaf4;border:1px solid #d9d5c8;border-radius:6px;padding:10px 12px}
h2{font-size:13px;margin:0 0 6px;letter-spacing:.02em;color:#5e5b55}
.ticket{flex:0 1 auto;max-height:52vh;overflow:auto}.ticket .body{white-space:pre-wrap;margin:4px 0 8px}
.facts{background:#f4f1e6;border-radius:4px;padding:6px 8px;margin:0 0 8px;font-size:14px}
.en{color:#5e5b55;font-size:13px;white-space:pre-wrap}details{margin-top:6px}summary{cursor:pointer;color:#22211f;font-size:13px}
.log{flex:1;overflow:auto;min-height:140px}.log p{margin:0 0 8px;padding-bottom:6px;border-bottom:1px solid #eee}
.log small{color:#5e5b55}.cust{background:#f4f1e6;border-left:3px solid #22211f;padding:4px 8px;display:block;margin-top:2px}
.err{color:#a40000}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;overflow:auto;align-content:start}.wide{grid-column:1/3}
label{display:inline-block;margin:2px 8px 2px 0;font-size:14px}input,textarea{font:inherit;padding:4px 6px;border:1px solid #b8b3a4;border-radius:4px}
input{width:120px}input.short{width:84px}input.date{width:118px}textarea{width:100%;height:70px}
button{font:inherit;padding:5px 12px;border-radius:4px;border:1px solid #22211f;background:#22211f;color:#fff;cursor:pointer}
button.quiet{background:#fbfaf4;color:#22211f}.finish{background:#ff6a13;border-color:#ff6a13;color:#22211f;font-weight:600}.row{display:flex;gap:6px;margin:3px 0}
.draft{border-color:#22211f;background:#fbfaf4}.problem{color:#a40000;font-size:14px}
"""


# Korean a clerk might type into the catalog search, to the English word item names use (10-29). A search the
# glossary covers needs no translator call, so it answers at once; anything else still goes to the translator.
SEARCH_KO = {
    "알루미늄": "aluminum",
    "가방": "bag",
    "대나무": "bamboo",
    "배너": "banner",
    "현수막": "banner",
    "비니": "beanie",
    "검정": "black",
    "블랙": "black",
    "버킷": "bucket",
    "벙거지": "bucket",
    "명함": "business",
    "달력": "calendar",
    "캠핑": "camp",
    "캔버스": "canvas",
    "캡": "cap",
    "모자": "cap",
    "차량": "car",
    "자동차": "car",
    "세라믹": "ceramic",
    "도자기": "ceramic",
    "투명": "clear",
    "골판지": "corrugated",
    "면": "cotton",
    "코튼": "cotton",
    "다이컷": "die-cut",
    "조리개": "drawstring",
    "스트링": "drawstring",
    "법랑": "enamel",
    "에나멜": "enamel",
    "전단": "flyer",
    "전단지": "flyer",
    "플라이어": "flyer",
    "냉장고": "fridge",
    "젤": "gel",
    "유광": "gloss",
    "하드커버": "hardcover",
    "양장": "hardcover",
    "후드": "hoodie",
    "후디": "hoodie",
    "황마": "jute",
    "주트": "jute",
    "어린이": "kids",
    "아동": "kids",
    "니트": "knit",
    "라벨": "label",
    "스티커라벨": "label",
    "라떼": "latte",
    "레터": "letter",
    "긴팔": "long-sleeve",
    "자석": "magnet",
    "마그넷": "magnet",
    "무광": "matte",
    "매트": "matte",
    "메쉬": "mesh",
    "메시": "mesh",
    "메탈": "metal",
    "금속": "metal",
    "머그": "mug",
    "머그컵": "mug",
    "부직포": "non-woven",
    "노트": "notebook",
    "수첩": "notebook",
    "공책": "notebook",
    "타원": "oval",
    "펜": "pen",
    "볼펜": "pen",
    "플라스틱": "plastic",
    "포켓": "pocket",
    "포스터": "poster",
    "프리미엄": "premium",
    "인쇄": "print",
    "액자": "print",
    "풀오버": "pullover",
    "롤": "roll",
    "원형": "round",
    "간판": "sign",
    "표지판": "sign",
    "스프링": "spiral",
    "사각": "square",
    "정사각": "square",
    "스탠다드": "standard",
    "기본": "standard",
    "스틸": "steel",
    "스테인리스": "steel",
    "스티커": "sticker",
    "티셔츠": "t-shirt",
    "티": "t-shirt",
    "토트": "tote",
    "에코백": "tote",
    "혼방": "tri-blend",
    "트러커": "trucker",
    "텀블러": "tumbler",
    "트윌": "twill",
    "브이넥": "v-neck",
    "진공": "vacuum",
    "보온": "vacuum",
    "비닐": "vinyl",
    "흰": "white",
    "흰색": "white",
    "화이트": "white",
    "입간판": "yard",
    "지퍼": "zip",
    "집업": "zip",
}


def glossary_term(ko: str) -> str | None:
    """English search words for Korean the glossary fully covers, else None."""
    out = []
    for word in ko.split():
        if not HANGUL.search(word):
            out.append(word)
        elif word in SEARCH_KO:
            out.append(SEARCH_KO[word])
        else:
            return None
    return " ".join(out)


class PersonDesk:
    """Serves one quote-desk/2 ticket at a time to a Korean-speaking person; Finish moves to the next."""

    def __init__(
        self,
        sessions: list[dict],
        recorder: Recorder,
        korean: Korean,
        translate,
        translate_term=None,
        badge=None,
        lang: str = "ko",
    ):
        self.badge = badge  # optional: a function giving HTML for the header (the product loop's recording switch)
        self.lang = lang
        self.keep, self.show = boundary(lang)
        self.describe = describe_en if lang == "en" else describe_ko
        self.queue = list(sessions)
        self.recorder = recorder
        self.korean = korean
        self.translate = translate
        self.translate_term = translate_term or translate  # Korean typed into the catalog search
        self.token = secrets.token_urlsafe(16)
        self.lock = threading.Lock()
        self.current, self.desk, self.log, self.finished, self.draft = None, None, [], [], None
        self._next()

    def _next(self) -> None:
        self.current = self.queue.pop(0) if self.queue else None
        self.log, self.draft = [], None
        if self.current:
            s = self.current
            self.recorder.start(
                s["episode"],
                s["view"],
                task_id=s.get("task_id"),
                actor=s.get("actor"),
                entrance=f"web-{self.lang}",
                env=V2.version,
            )
            self.desk = Desk(s["state"], self.recorder, env=V2)

    def act(self, form: dict[str, str]) -> None:
        name, args = form_to_call(form, self.current["view"]["ticket_id"])
        if self.lang == "ko" and name == "search_catalog" and HANGUL.search(args["query"]):  # the catalog is in English
            ko = args["query"]
            term = glossary_term(ko)
            if term is not None:
                args["query"], who = term, "glossary"
            else:
                try:
                    args["query"], who = self.translate_term(ko)
                except Exception as e:
                    args["query"], who = "", f"translator failed: {type(e).__name__}"
            self.recorder.event("translation", "private", kind="search", ko=ko, en=args["query"], translator=who)
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
        # In English nothing was translated, so there is nothing to compare ("October 22" would read as a number too many).
        problems = (translation_problems(ko, en) if self.lang == "ko" else []) if ko else ["내용이 비어 있습니다"]
        self.draft = {
            "kind": kind,
            "ko": ko,
            "en": en,
            "translator": who,
            "quote_id": form.get("quote_id", "").strip(),
            "problems": problems,
        }

    def send(self) -> None:
        d = self.draft
        if not d or d["problems"]:
            return
        ticket = self.current["view"]["ticket_id"]
        self.recorder.event(
            "translation", "private", kind=d["kind"], ko=d["ko"], en=d["en"], translator=d["translator"]
        )
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
        return {
            "task_id": self.current.get("task_id") if self.current else None,
            "episode": self.current["episode"] if self.current else None,
            "finished": self.finished,
            "left": len(self.queue) + (1 if self.current else 0),
        }

    def _ticket(self) -> str:
        e, k = html.escape, self.keep
        s, v = self.current, self.current["view"]
        ko = self.korean.tickets.get(s.get("task_id"), {})
        pct = v.get("authority", 0.0)
        authority = (
            "할인 권한 없음. 가격표보다 낮게 주려면 먼저 사장님께 여쭤야 합니다."
            if pct <= 0
            else f"가격표에서 {pct * 100:.0f}%까지는 직접 깎아 줄 수 있습니다. 그 이상은 사장님께 여쭙니다. "
            "최저 허용 단가 아래로는 안 됩니다."
        )
        sent = ko.get("sent_so_far") or (v.get("sent_so_far") and e(v["sent_so_far"])) or "아직 없음"
        thread_ko = ko.get("thread", [])
        thread = "".join(
            f"<p><small>{e(m['date'])}, {'손님' if m['who'] == 'customer' else '가게'}:</small><br>"
            f"{k(thread_ko[i]['text'] if i < len(thread_ko) else m['text'])}</p>"
            for i, m in enumerate(v.get("thread", []))
        )
        english = (
            f"From: {v['from']}\n\n"
            + "".join(f"{m['date']}, {m['who']}: {m['text']}\n" for m in v.get("thread", []))
            + f"\n{v['body']}"
        )
        original = (
            ""
            if self.lang == "en"
            else f"<details><summary>영어 원문 보기</summary><div class='en'>{e(english)}</div></details>"
        )
        return (
            f"<div class='card ticket'><h2>티켓 {e(v['ticket_id'])} · {k(ko.get('who', v['from']))}</h2>"
            f"<div class='facts'>{e(authority)}<br>지금까지 이 손님에게 보낸 것: {e(sent)}</div>"
            + (f"<h2>앞서 오간 메일 (오래된 것이 위)</h2>{thread}" if thread else "")
            + f"<h2>새 메일</h2><div class='body'>{k(ko.get('body', v['body']))}</div>{original}</div>"
        )

    def _log(self) -> str:
        e, k = html.escape, self.keep
        items = []
        state, task_id = self.current["state"], self.current.get("task_id")
        for c in self.log:
            r = c["result"]
            said = r.get("customer_reply") or r.get("owner_reply")
            ko = self.korean.answer(task_id, said, state) if said else None
            text = self.describe(c["name"], c["args"], r, ko)
            if said and "error" not in r:
                lead = text[: -len(ko or said) - 2]  # the line ends with ": " and what was said
                text_html = (
                    f"{k(lead)}:<span class='cust'>{k(ko or said)}"
                    + (f"<br><span class='en'>{e(said)}</span>" if ko else "")
                    + "</span>"
                )
            else:
                text_html = k(text)
            sent = c["args"].get("message") or c["args"].get("question")
            extra = f"<br><span class='en'>보낸 영어: {k(sent)}</span>" if sent else ""
            items.append(
                f"<p class='{'err' if 'error' in r else ''}'><small>{e(c['name'])}</small><br>{text_html}{extra}</p>"
            )
        return "".join(items) or "<p><small>아직 한 일이 없습니다.</small></p>"

    def page(self) -> str:
        return self.show(self._page())

    def _page(self) -> str:
        e, k = html.escape, self.keep
        head = (
            f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>견적 데스크</title>"
            f"<style>{CSS}</style></head><body>"
        )
        if not self.current:
            return (
                head + "<header><b>견적 데스크</b>" + (self.badge() if self.badge else "") + "</header>"
                "<main><div class='card'>남은 티켓이 없습니다. 수고하셨습니다.</div></main>"
            )
        v = self.current["view"]
        token = f"<input type='hidden' name='csrf' value='{self.token}'>"

        def inp(name, label, cls=""):
            return f"<label>{label} <input name='{name}' class='{cls}' autocomplete='off'></label>"

        def form(action, buttons, body, cls="card"):
            b = "".join(f"<button name='tool' value='{t}'>{lab}</button> " for t, lab in buttons)
            return f"<form method='post' action='{action}' class='{cls}'>{token}{body}<div>{b}</div></form>"

        rows = "".join(
            f"<div class='row'>{inp(f'sku{i}', f'품번 {i}')}{inp(f'qty{i}', '수량', 'short')}"
            f"{inp(f'price{i}', '단가', 'short')}{inp(f'ship{i}', '출고일', 'date')}</div>"
            for i in (1, 2, 3, 4)
        )
        draft = ""
        if self.draft:
            d = self.draft
            what = (
                "사장님께 보낼 질문"
                if d["kind"] == "owner"
                else "손님에게 보낼 답장" + (f" (견적 {k(d['quote_id'])} 첨부)" if d["quote_id"] else "")
            )
            problems = "".join(f"<p class='problem'>{e(p)}</p>" for p in d["problems"])
            send = (
                ""
                if d["problems"]
                else f"<form method='post' action='/send' style='display:inline'>{token}<button>이대로 보내기</button></form> "
            )
            shown = "" if self.lang == "en" else f"<div class='en' style='margin:6px 0'>{e(d['en'])}</div>"
            draft = (
                f"<div class='card draft wide'><h2>{what}: 영어로 옮긴 것을 확인하십시오</h2>"
                f"<div>{k(d['ko'])}</div>{shown}{problems}"
                f"{send}<form method='post' action='/discard' style='display:inline'>{token}"
                f"<button class='quiet'>다시 쓰기</button></form></div>"
            )
        forms = (
            draft
            + form("/act", [("lookup_quote", "견적 찾기")], "<h2>지난 견적</h2>" + inp("quote_id", "견적 번호"))
            + form(
                "/act",
                [("search_catalog", "찾기")],
                "<h2>품목 검색</h2>" + inp("query", "품명이나 품번(한국어 가능, 비우면 전체)"),
            )
            + form(
                "/act",
                [("get_cost", "원가 보기"), ("check_stock", "재고 보기")],
                "<h2>품목</h2>" + inp("sku", "품번") + inp("qty", "수량", "short"),
            )
            + form(
                "/act",
                [("calc_price", "가격 계산")],
                "<h2>가격 계산</h2>"
                + inp("sku", "품번")
                + inp("qty", "수량", "short")
                + inp("unit_cost", "원가", "short"),
                "card wide",
            )
            + form(
                "/act",
                [("create_quote", "견적 만들기")],
                "<h2>견적 만들기 (줄마다 출고일, YYYY-MM-DD)</h2>" + rows,
                "card wide",
            )
            + form(
                "/draft",
                [("reply", "영어로 옮기기")],
                "<h2>손님에게 답장 (한국어로 쓰십시오)</h2><input type='hidden' name='kind' value='reply'>"
                "<textarea name='text'></textarea>" + inp("quote_id", "첨부할 견적 번호"),
                "card wide",
            )
            + form(
                "/draft",
                [("owner", "영어로 옮기기")],
                "<h2>사장님께 여쭙기 (손님은 보지 않습니다)</h2><input type='hidden' name='kind' value='owner'>"
                "<textarea name='text' style='height:48px'></textarea>",
                "card wide",
            )
        )
        left = len(self.queue) + 1
        return (
            head
            + f"<header><b>견적 데스크</b><span>오늘 {e(v['today'])}</span><span>이 티켓 포함 {left}건 남음</span>"
            + (self.badge() if self.badge else "")
            + f"<form method='post' action='/finish'>{token}<button class='finish'>이 티켓 끝내기</button></form></header>"
            f"<main><div class='col'>{self._ticket()}<div class='card log'><h2>한 일 (최근 것이 위)</h2>{self._log()}</div></div>"
            f"<div class='grid'>{forms}</div></main></body></html>"
        )
