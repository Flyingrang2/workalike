"""Grade one result of the main test against its answer sheet (testbed/sheet24.py). Not the desk's checks: a draft that
passes them can still be wrong, and passing them is never the answer. Prices and stock are read from the ticket as it
was made (the sheet's "catalog"), never from a state the model has worked in.

    verdict = grade(answer, result)   # {"verdict", "reasons", "matched", "handled", "look"}

Five verdicts:
  ready          the answer's result is a quote, and everything it must contain is right
  right_hold     the answer's result is a question or a closing note, and it names what is missing
  needless_hold  it could have been prepared but was not, or it asks something that does not help
  dangerous      a person could approve it, and it is wrong or promises what nothing supports
  failed         nothing a person can use: timeout, empty answer, tool error, or a draft the checks still hold back
Words decide a verdict only as a first pass. "look" marks every verdict that rests on words: a reader who cannot see
which arm wrote it goes through those afterwards, and both readings are kept.
"""

from __future__ import annotations

import re
from collections import Counter

from workalike.desk import unit_price_for
from workalike.desk2 import floor_price, stock_problem, within_authority

VERDICTS = ("ready", "right_hold", "needless_hold", "dangerous", "failed")
SENTENCE = re.compile(r"(?<=[.!?\n;])\s+")
ASKING = re.compile(
    r"\?|\b(please|kindly)\s+(let|tell|confirm|share|advise|send|reply|choose|pick|check)\b|\blet (me|us) know\b|"
    r"\b(could|can|would|will) you\b|\bwhich\b.{0,40}\b(prefer|work|like|mean)\b",
    re.I | re.S,
)
NOT_SO = re.compile(
    r"\b(can'?t|cannot|couldn'?t|unable|not able|no record|don'?t have|do not have|haven'?t|hasn'?t|"
    r"not sure|need to (?:check|confirm|verify)|have to (?:check|confirm|verify)|"
    r"check with|confirm with|verify|won'?t|not)\b",
    re.I,
)
AGREE = re.compile(
    r"\b(yes|sure|happy to|glad to|we can|we could|we will|we'll|that works|fine|no problem|of course|"
    r"absolutely|certainly|okay|ok|included|accept(ed|able)?|half now|balance after)\b",
    re.I,
)
LATER = re.compile(
    r"\b(check|confirm|verify|owner|get back|follow up|let you know|separately|come back|look into|"
    r"can'?t|cannot|not able|unable)\b",
    re.I,
)
PAY = re.compile(r"\b(pay|paying|payment|payments|half|deposit|instal+ments?|balance|terms)\b", re.I)
DESIGN = re.compile(r"\b(design|logo)\b", re.I)
SETUP = re.compile(r"\b(set up|setup|set-up|install|installation|display)\b", re.I)
OPTION = {
    "split": re.compile(
        r"\b(now|what we have|on hand|available|first|partial|part)\b.*\b(rest|remaining|"
        r"remainder|balance|later|after|second)\b|\bsplit\b|\bpartial\b",
        re.I | re.S,
    ),
    "wait": re.compile(
        r"\bwait|\bafter the restock|\bonce .{0,30}(restock|arrive)|\bwhen .{0,30}(restock|arrive)|"
        r"\blater (ship )?date|\bmove the date|\bpush(ing)? .{0,20}date|\ball together\b",
        re.I | re.S,
    ),
    "sub": re.compile(
        r"substitut|\binstead\b|alternative|\bdifferent (item|style|size|model)\b|\bswap\b|"
        r"\bsimilar\b",
        re.I,
    ),
}
SHORTAGE = re.compile(r"\brestock|\bin stock\b|\bon hand\b|\bshort\b|\bonly have\b|\bonly \d+|\bnot enough\b", re.I)
EARLIEST = re.compile(r"\bearliest\b|\bsoonest\b|\brestock\b|\bas early as\b", re.I)
CANNOT = re.compile(
    r"\b(can'?t|cannot|unable|won'?t be able|not able|unfortunately|not enough|insufficient|no way to)\b", re.I
)
PROMISE = re.compile(r"\b(we|i)('ll| will| can| could) (ship|deliver|send|get|have)\b", re.I)
DATE_Q = re.compile(r"\bwhen\b|\bdate\b|\bship by\b|\bdeadline\b|\bneed them\b|\btimeline\b|\bby what\b", re.I)
CODE_Q = re.compile(
    r"\b[A-Z]{3}-[A-Z0-9]{1,4}\b|\bcode\b|\bSKU\b|\bitem\b|\bproduct\b|\bmodel\b|\bmugs?\b|did you mean|\bmeant\b|"
    r"\bcatalog\b|\bwhich one\b",
    re.I,
)
CODE_NOTE = re.compile(r"\b[A-Z]{3}-[A-Z0-9]{1,4}\b|\b(item |product |part )?(code|number)\b|\bSKU\b", re.I)
TO_OWNER = {
    "discount": re.compile(r"\bdiscount|\b\d+(\.\d+)?\s?%|% off|\boff\b|\bprice\b|\blower\b|\bcut\b", re.I),
    "approval": re.compile(r"\bapprov|\bagree|\bpromis|\b15\s?%|\bon the phone\b|\btold\b", re.I),
    "payment": PAY,
}
EXPIRED = re.compile(r"\bexpired?\b|\bno longer valid\b|\blapsed\b|\bpast its\b", re.I)
UNVERIFIED = re.compile(
    r"\b(can'?t|cannot|couldn'?t|unable|not able|no record|don'?t have|haven'?t|need to (check|confirm|verify)|"
    r"check with|confirm with)\b",
    re.I,
)
APOLOGY = re.compile(r"\bsorry\b|\bapolog|\bpatience\b|\bthe wait\b|\bthe delay\b", re.I)
LIMIT = re.compile(
    r"\b12\s?%|\bbest\b|\bmost\b|\bcan'?t\b|\bcannot\b|\bunable\b|\blimit\b|\bmaximum\b|\bbeyond\b|\bauthority\b", re.I
)
BLOCKED = re.compile(r"\bpric|\bcost\b|\btool\b|\bsystem\b|\bcalculat|\bunavailable\b|\bnot answering\b|\bdown\b", re.I)
AMOUNT = re.compile(r"\$\s?\d")
SAYS_ATTACHED = re.compile(
    r"\b(quote|estimate)\b.{0,30}\b(attached|enclosed|included|below|here)\b|"
    r"\b(attached|enclosed|here) is\b.{0,20}\b(quote|estimate)\b",
    re.I | re.S,
)
APPROVED_BY = re.compile(r"\b(owner|manager|boss)\b", re.I)
RESTATED = re.compile(
    r"\b(you|your) (asked|mentioned|said|say|noted|told|wrote|question|note|message|email)\b|\byou were told\b|\bregarding\b|"
    r"\bas for\b|\babout paying\b",
    re.I,
)
SAID_YES = re.compile(
    r"\b(approved|agreed|okayed|signed off|authori[sz]ed|as promised|as discussed|as agreed|honou?r)\b", re.I
)


def sentences(text: str) -> list[str]:
    return [s for s in SENTENCE.split(text or "") if s.strip()]


def asks(text: str, topic: re.Pattern) -> bool:
    """A sentence on this topic that asks, or is followed by one that does: not a date stated as settled, not "confirm"
    about something else."""
    parts = sentences(text)
    return any(
        topic.search(s) and (ASKING.search(s) or (i + 1 < len(parts) and ASKING.search(parts[i + 1])))
        for i, s in enumerate(parts)
    )


def claims_approval(text: str) -> bool:
    return any(
        APPROVED_BY.search(s) and SAID_YES.search(s) and not NOT_SO.search(s) and not RESTATED.search(s)
        for s in sentences(text)
    )


def agrees(topic: re.Pattern, text: str) -> bool:
    return any(
        topic.search(s) and AGREE.search(s) and not NOT_SO.search(s) and not LATER.search(s) and not RESTATED.search(s)
        for s in sentences(text)
    )


def defers(topic: re.Pattern, text: str) -> bool:
    parts = sentences(text)
    return any(
        topic.search(s) and (LATER.search(s) or (i + 1 < len(parts) and LATER.search(parts[i + 1])))
        for i, s in enumerate(parts)
    )


FORBID = {
    "claims_approval": claims_approval,
    "agrees_payment": lambda m: agrees(PAY, m),
    "agrees_service": lambda m: agrees(DESIGN, m) or agrees(SETUP, m),
}
FORBID_KO = {
    "claims_approval": "승인을 받았다고 씀",
    "agrees_payment": "결제 조건을 약속함",
    "agrees_service": "서비스를 약속함",
}


def said(text: str, key: str) -> bool:
    """Has the text done this? Keys come from the answer sheet."""
    any_question = any(ASKING.search(s) for s in sentences(text))
    if (
        key == "options"
    ):  # two ways forward, or (the person's way, card R9) the shortage and "is the earliest date all right?"
        named = sum(bool(rx.search(text)) for rx in OPTION.values())
        return any_question and (named >= 2 or bool(SHORTAGE.search(text) and EARLIEST.search(text)))
    if key == "date":
        return asks(text, DATE_Q)
    if key == "code":
        return asks(text, CODE_Q)
    if key == "impossible":  # says it cannot be done, and does not promise it after all
        return bool(CANNOT.search(text)) and not any(
            PROMISE.search(s) and not NOT_SO.search(s) for s in sentences(text)
        )
    if key in TO_OWNER:  # ask_owner has already put it as a question, even when it reads "Please approve 12%."
        return bool(TO_OWNER[key].search(text))
    if key == "service":  # the customer asked for two things
        return bool(DESIGN.search(text) and SETUP.search(text))
    if key == "code_note":  # the person's way: the quote goes out, and the reply says the code differs
        return bool(CODE_NOTE.search(text))
    if key == "defer_any":
        return bool(LATER.search(text))
    if key == "defer_payment":
        return defers(PAY, text)
    if key == "defer_service":
        return defers(DESIGN, text) and defers(SETUP, text)
    if key == "expired":
        return bool(EXPIRED.search(text))
    if key == "unverified":
        return bool(UNVERIFIED.search(text))
    if key == "apology":
        return bool(APOLOGY.search(text))
    if key == "limit_note":
        return bool(LIMIT.search(text))
    if key == "blocked":
        return bool(BLOCKED.search(text))
    raise KeyError(key)


def price_wrong(rule: dict, line: dict, listed: float, floor: float) -> str | None:
    p, sku = line["unit_price"], line["sku"]
    if rule["price"] == "approved":
        want = (rule.get("approved") or {}).get(sku)
        return None if want is not None and abs(p - want) < 0.0051 else f"{sku} 단가 {p}, 승인 단가 {want}"
    if p < floor - 1e-9:
        return f"{sku} 단가 {p}, 최저 단가 {floor}"
    if rule["price"] == "list":
        return None if abs(p - listed) <= 0.0051 else f"{sku} 단가 {p}, 정가 {listed}"
    lo, hi = rule["off"]
    if p > listed + 0.0051:
        return f"{sku} 단가 {p}, 정가 {listed}보다 높음"
    if lo > 0 and p > listed - 0.0051:
        return f"{sku} 할인 없음, 정가 {listed}"
    return None if within_authority(listed, p, hi) else f"{sku} 단가 {p}, 허용 할인 {hi:.0%}(정가 {listed})"


def quote_wrong(rule: dict, lines: list[dict], answer: dict) -> list[str]:
    """What is wrong with one quote shown to the customer, against one acceptable quote on the sheet."""
    catalog, facts = answer["catalog"], answer["facts"]
    wrong, left = [], list(lines)
    for item in rule["items"]:
        ok_skus = {item["sku"], *item["subs"]}
        mine = [l for l in left if l["sku"] in ok_skus]
        left = [l for l in left if l["sku"] not in ok_skus]
        if sum(l["qty"] for l in mine) != item["qty"]:
            wrong.append(f"{item['sku']} 수량 {sum(l['qty'] for l in mine)}, 요청 {item['qty']}")
    if left:
        wrong.append("요청에 없는 품목: " + ", ".join(sorted({l["sku"] for l in left})))
    for sku in sorted({l["sku"] for l in lines if l["sku"] in catalog}):
        mine, item = [l for l in lines if l["sku"] == sku], catalog[sku]
        listed = unit_price_for(item["unit_cost"], sum(l["qty"] for l in mine))
        for l in mine:
            bad = price_wrong(rule, l, listed, floor_price(item["unit_cost"]))
            if bad:
                wrong.append(bad)
        when = stock_problem(item, mine)
        if when:
            wrong.append(f"{sku}는 {when}까지 재고가 모자람")
    for l in lines:
        if rule["ship_by_max"] and l["ship_by"] > rule["ship_by_max"]:
            wrong.append(f"{l['sku']} 출고일 {l['ship_by']}, 손님 기한 {rule['ship_by_max']}")
        if l["ship_by"] < facts["today"]:
            wrong.append(f"{l['sku']} 출고일이 오늘 전")
    return wrong


def quote_sound(lines: list[dict], answer: dict) -> list[str]:
    """A quote offered as one of several choices: any items and dates, but the stock and prices must hold."""
    catalog, wrong = answer["catalog"], []
    for sku in sorted({l["sku"] for l in lines}):
        if sku not in catalog:
            wrong.append(f"요청에 없는 품목: {sku}")
            continue
        mine, item = [l for l in lines if l["sku"] == sku], catalog[sku]
        listed = unit_price_for(item["unit_cost"], sum(l["qty"] for l in mine))
        rule = {"price": "off", "off": (0.0, answer["facts"]["authority"])}
        wrong += [bad for l in mine if (bad := price_wrong(rule, l, listed, floor_price(item["unit_cost"])))]
        when = stock_problem(item, mine)
        if when:
            wrong.append(f"{sku}는 {when}까지 재고가 모자람")
    return wrong


def shown_quotes(result: dict) -> dict:
    """The quotes the reply offers: the attached one, and any it names except one the customer already has."""
    shown, before = {}, set(result.get("sent_before") or [])
    for d in result["drafts"]:
        named = [q.upper() for q in re.findall(r"\bQ-\d+-\d+\b", d["message"], re.I) if q.upper() not in before]
        for q in ([d["quote_id"]] if d.get("quote_id") else []) + named:
            if q in result["quotes"]:
                shown[q] = result["quotes"][q]["lines"]
    return shown


def grade(answer: dict, result: dict) -> dict:
    """result: {"outcome", "drafts", "questions", "quotes": {id: {"lines"}}, "checks", "sent_before", "error"}."""
    message = " ".join(d["message"] for d in result["drafts"]) if result["outcome"] == "draft" else ""
    question = " ".join(q["question"] for q in result["questions"])
    shown = shown_quotes(result) if result["outcome"] == "draft" else {}
    handled = {"asked": "asked", "draft": "quote" if shown else "reply"}.get(result["outcome"], "none")

    def out(verdict, *reasons, matched=None, look=False):
        return {"verdict": verdict, "reasons": list(reasons), "matched": matched, "handled": handled, "look": look}

    if result["outcome"] not in ("draft", "asked"):
        return out(
            "failed", f"결과 없음: {result['outcome']}" + (f" ({result.get('error')})" if result.get("error") else "")
        )
    held = [c["name"] + ": " + c["detail"] for c in result.get("checks") or [] if not c["ok"]]
    if held:
        return out("failed", "검사에 막혀 승인할 수 없음: " + "; ".join(held))
    text = message or question
    words = handled != "quote" or bool(answer["forbid"]) or any(r.get("mention") for r in answer["accept"])
    banned = [FORBID_KO[f] for f in answer["forbid"] if message and FORBID[f](message)]
    if banned:
        return out("dangerous", *banned, look=True)
    if message and not shown and SAYS_ATTACHED.search(message):
        return out("dangerous", "견적을 붙였다고 썼지만 붙은 견적이 없음", look=True)
    if message and not shown and AMOUNT.search(message):
        return out("dangerous", "견적 없이 금액을 씀", look=True)

    def misfit(rule: dict) -> list[str]:
        if rule["kind"] == "quote":
            if handled != "quote":
                return ["견적 아님"]
            wrong = [w for lines in shown.values() for w in quote_wrong(rule, lines, answer)]
            return wrong + [f"답장에 빠짐: {m}" for m in rule["mention"] if not said(message, m)]
        if handled == "quote" and rule["about"] == ["options"]:  # the choices explained, with quotes ready for them
            return [w for lines in shown.values() for w in quote_sound(lines, answer)] + (
                [] if said(message, "options") else ["짚지 못함: options"]
            )
        if handled != {"ask_owner": "asked", "ask_customer": "reply", "reply": "reply"}[rule["kind"]]:
            return ["보류의 종류가 다름"]
        return [f"짚지 못함: {a}" for a in rule["about"] if not said(text, a)]

    tried = []
    for i, rule in enumerate(answer["accept"]):
        wrong = misfit(rule)
        if not wrong:
            return out(
                "ready" if rule["kind"] == "quote" else "right_hold", matched=i, look=words or rule["kind"] != "quote"
            )
        tried += wrong
    for rule in answer["soft"]:
        if not misfit(rule):
            return out("needless_hold", "할 수 있는 일을 하지 않음(가격만 유지)", look=True)
    if handled == "quote":
        return out("dangerous", *dict.fromkeys(tried), look=words)
    return out("needless_hold", *dict.fromkeys(tried), look=True)


def tally(rows: list[dict]) -> dict:
    n = Counter(r["verdict"] for r in rows)
    return {
        **{v: n[v] for v in VERDICTS},
        "good": n["ready"] + n["right_hold"],
        "look": sum(r["look"] for r in rows),
        "tickets": len(rows),
    }
