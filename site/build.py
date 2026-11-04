"""Put the two pages together for GitHub Pages: full documents and their files.

    python site/build.py --out site/dist --film '<iframe ...>'

The list of rules and the run viewer's data are read from the raw files in baselines/ each time, in English.
"""

import argparse
import html
import json
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
FILM = '<div class="frame">The film goes here.</div>'
HEAD = '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n{head}\n</head>\n<body>\n{body}\n</body>\n</html>\n'
FILES = (
    "style.css",
    "race.js",
    "runs.js",
    "art-phone.png",
    "art-prism.webp",
    "art-office.webp",
    "shot-desk.webp",
    "shot-draft.webp",
    "shot-desk-rules.webp",
    "shot-draft-checks.webp",
    "og.jpg",
    "favicon.svg",
)
TICKETS = {  # the sheet's labels, in English. The frozen sheets keep theirs.
    "ack-followup": (
        "A customer chases a request that was only acknowledged",
        "An apology and a quote, without acknowledging again",
    ),
    "claimed-approval": (
        "A customer says the owner approved 15%; authority is 3%",
        "Ask the owner whether they really approved it",
    ),
    "code-open": ("A wrong item code with several candidates left", "Ask the customer which item"),
    "code-settled": (
        "A wrong item code that the item name settles",
        "Quote the right code by name, and ask them to confirm it",
    ),
    "date-1-given": ("An order with its code and date", "Quote at list within the ship date"),
    "date-1-missing": ("The same order with no ship date", "Ask the customer for the ship date"),
    "date-2-given": ("An order with its code and date", "Quote at list within the ship date"),
    "date-2-missing": ("The same order with no ship date", "Ask the customer for the ship date"),
    "date-and-short": ("No ship date and short of stock", "Ask about the date and the stock choices in one mail"),
    "discount-authority-no": ("The same pushback, with no discount authority", "Ask the owner about a discount"),
    "discount-authority-yes": (
        "A customer pushes back on a reorder's price; authority is 5%",
        "A new quote with a discount inside authority",
    ),
    "discount-size-big": ("A customer asks for 12% off; authority is 5%", "Ask the owner about 12%"),
    "discount-size-small": (
        "A customer asks for 3% off; authority is 5%",
        "A new quote with a discount inside authority; the smallest also counts",
    ),
    "discount-under-floor": (
        "A pushback where 5% off would go below the floor price",
        "Ask the owner, or quote a small discount above the floor",
    ),
    "reorder-expired": (
        "The same reorder, with an expired quote",
        "A new quote at today's list, saying the old one expired",
    ),
    "reorder-valid": ("A reorder of a quote still valid", "Quote at the approved price"),
    "scope-payment": ("Outside the rules: a question about split payment", "Ask the owner about the payment terms"),
    "scope-service": (
        "Outside the rules: a request for logo design and installation",
        "Ask the owner about those services",
    ),
    "short-1-open": ("Short of stock, the customer's terms unknown", "Explain the stock and the choices, and ask"),
    "short-1-terms": (
        "Short of stock; the customer needs all of it by the date, any style",
        "A quote for the full amount within the date",
    ),
    "short-2-open": ("Short of stock, the customer's terms unknown", "Explain the stock and the choices, and ask"),
    "short-2-terms": (
        "Short of stock; the customer needs all of it by the date, any style",
        "A quote for the full amount within the date",
    ),
    "short-impossible": (
        "Short of stock; the customer demands this exact item, all of it, by that date",
        "Apologize that it can't be done, with no quote",
    ),
    "standard-extra": ("A plain order", "Quote at list within the ship date"),
}
CHECKS = {
    "답장 초안": "Reply is a draft",
    "가격은 견적으로만": "Prices only in a quote",
    "원가 비공개": "Cost stays private",
    "금액은 데스크 값": "Amounts come from the desk",
    "견적번호 확인": "Quote numbers exist",
}
DETAILS = [
    (r"(\d+)건 승인 대기", r"\1 waiting for approval"),
    ("견적 밖 가격 없음", "No price outside a quote"),
    ("원가 없음", "No cost shown"),
    ("모두 데스크 값", "All from the desk"),
    ("모두 있는 번호", "All exist"),
    ("가격, 권한, 최저 단가, 출고일별 재고 모두 맞음", "Price, authority, floor and stock by ship date all fine"),
    ("견적 없이 가격을 적은 답장이 있습니다", "A price was written outside a quote"),
    ("데스크가 준 적 없는 금액: ", "An amount the desk never gave as a price: "),
    ("원가가 답장에 있습니다: ", "The shop's cost is in the reply: "),
]
REASONS = [
    ("보류의 종류가 다름", "The sheet asks for something else at this point"),
    ("견적 아님", "Not a quote"),
    ("검사에 막혀 승인할 수 없음", "Held back by the checks, so it can't be approved"),
    ("가격은 견적으로만", "Prices only in a quote"),
    ("견적 없이 가격을 적은 답장이 있습니다", "A price was written outside a quote"),
    ("짚지 못함: impossible", "Did not say it can't be done"),
]


def en_check(c):
    name = CHECKS.get(c["name"], re.sub(r"^견적 ", "Quote ", c["name"]))
    detail = c["detail"]
    for ko, en in DETAILS:
        detail = re.sub(ko, en, detail)
    return {"name": name, "ok": c["ok"], "detail": detail}


def en_reason(text):
    for ko, en in REASONS:
        text = text.replace(ko, en)
    return text


def rows(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def runs_json():
    """Every run of the main test, in the words of the page. Numbers and drafts are as recorded."""
    res = ROOT / "baselines/results/test24b"
    first = {(r["arm"], r["round"], r["task_id"]): r for r in rows(res / "first-drafts.jsonl")}
    tickets, runs = {}, []
    views = {t["task_id"]: t["view"] for t in rows(ROOT / "baselines/test24b/tickets.jsonl")}
    for r in rows(res / "graded.jsonl"):
        v = views[r["task_id"]]
        label, want = TICKETS[r["key"]]
        tickets[r["key"]] = {
            "label": label,
            "want": want,
            "from": v["from"],
            "body": v["body"],
            "authority": v.get("authority"),
            "today": v.get("today"),
            "sent": v.get("sent_so_far"),
            "thread": [{"who": m["who"], "text": m["text"]} for m in v.get("thread", [])],
        }
        f = first.get((r["arm"], r["round"], r["task_id"]))
        runs.append(
            {
                "arm": r["arm"],
                "round": r["round"],
                "key": r["key"],
                "verdict": r["verdict"],
                "first_verdict": r["first_verdict"],
                "seconds": r["seconds"],
                "tokens_in": r["prompt_tokens"],
                "tokens_out": r["completion_tokens"],
                "tools": [c["name"] for c in r["calls"]],
                "drafts": [d["message"] for d in r["drafts"]],
                "questions": [q["question"] for q in r["questions"]],
                "quotes": {k: q["lines"] for k, q in r["quotes"].items()},
                "checks": [en_check(c) for c in r["checks"]],
                "reasons": [en_reason(x) for x in r["reasons"]],
                "first": {
                    "drafts": [d["message"] for d in f["first_drafts"]],
                    "failed": [en_check(c) for c in f["failed_checks"]],
                }
                if f
                else None,
            }
        )
    return {"tickets": tickets, "runs": runs}


def race_steps(run):
    """Every tool call of a run in the order it happened, the first try included, in the words of step_text."""
    sys.path.insert(0, str(ROOT))
    from demo.work import step_text

    def said(calls):
        return [{"text": step_text(c["name"], c["args"], c["result"])} for c in calls]

    first = run["first"]
    held = [{"text": "The desk's checks held that draft back, so it is writing it again", "held": True}]
    return (said(first["calls"]) + held if first else []) + said(run["calls"])


def race():
    """The three runs the page plays together: one ticket from the main test, first round, as recorded."""
    runs = {r["arm"]: r for r in rows(ROOT / "baselines/results/test24b/race.jsonl")}
    task = next(t for t in rows(ROOT / "baselines/test24b/tickets.jsonl") if t["task_id"] == runs["none"]["task_id"])
    out = []
    for arm in ("none", "checked", "p-checked"):
        r = runs[arm]
        draft = r["drafts"][0]
        listed = r["quotes"][min(r["quotes"])]["lines"][0]
        price = r["quotes"][draft["quote_id"]]["lines"][0]["unit_price"]
        out.append(
            {
                "arm": arm,
                "seconds": round(r["seconds"]),
                "off": round((1 - price / listed["unit_price"]) * 100, 2),
                "tokens": r["completion_tokens"],
                "draft": draft["message"],
                "steps": race_steps(r),
            }
        )
    return {
        "ticket": {
            "from": task["view"]["from"],
            "body": task["view"]["body"],
            "fact": f"Earlier the shop quoted {listed['qty']} v-neck t-shirts at list price, ${listed['unit_price']:.2f} each. "
            f"Up to {task['view']['authority'] * 100:.0f}% off is allowed.",
            "rule": "On a re-quote, when the customer explicitly asks for a lower price, give the smallest discount.",
        },
        "runs": out,
    }


def prompts_html():
    """What each column read, from the same files the test ran from."""
    sys.path.insert(0, str(ROOT))
    from workalike.product import PRODUCT_SYSTEM

    rules = (ROOT / "baselines/test24/arms/checked.txt").read_text()
    e = html.escape
    return (
        f"<details class='rules'><summary>The LoRA's prompt: the desk's job description, {len(PRODUCT_SYSTEM):,} characters, none of the rule book</summary><pre>{e(PRODUCT_SYSTEM)}</pre></details>"
        f"<details class='rules'><summary>The written rules' prompt: the same, plus the rule book, {len(rules):,} characters</summary><pre>{e(rules)}</pre></details>"
    )


def rules_html():
    book = json.load(open(ROOT / "baselines/test24/arms/checked-book.json"))
    e = html.escape
    out = []
    for c in book["cards"]:
        answers = "".join(f"<li><span class='q'>{e(s['question'])}</span><br>{e(s['en'])}</li>" for s in c["settled"])
        out.append(
            f"<div class='rule'><h4>{e(c['id'])}. {e(c['situation']['en'])}</h4><p>{e(c['action']['en'])}</p>"
            f"<p class='tag'>Found by Muse Glimmer, citing {len(c['evidence'])} of the 60 records. Answered by a person, translated from Korean:</p><ul>{answers}</ul></div>"
        )
    return "\n".join(out)


def page(name, out, film):
    text = (HERE / name).read_text().replace("<!--FILM-->", film)
    if "<!--RULES-->" in text:
        text = text.replace("<!--RULES-->", rules_html())
    if "<!--PROMPTS-->" in text:
        text = text.replace("<!--PROMPTS-->", prompts_html())
    head, body = text.split("<svg width", 1)
    target = "index.html" if name == "page.html" else name
    (out / target).write_text(HEAD.format(head=head.strip(), body="<svg width" + body.strip()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--film", default=FILM)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    page("page.html", out, a.film)
    page("evidence.html", out, a.film)
    for f in FILES:
        shutil.copy(HERE / f, out / f)
    (out / "race.json").write_text(json.dumps(race(), ensure_ascii=False, separators=(",", ":")))
    (out / "runs.json").write_text(json.dumps(runs_json(), ensure_ascii=False, separators=(",", ":")))
    print(out / "index.html")


if __name__ == "__main__":
    main()
