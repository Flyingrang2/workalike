"""English for the product screen, for a demo; Korean stays the default and its code path does not change.

The screen's own words are written in Korean all through the code, and some are data (guard()'s check names are also
keys of HINTS), so they are translated once, when a finished page leaves: english(page).

A replacement cannot reach what is not the screen's own. Every entry below holds Hangul, so English text never
matches. Tags (form values, the CSRF token, ids) and textareas are skipped whole, and so are the customer's, the
model's and the person's words, which the pages mark with keep() because they may be in Korean.
"""

from __future__ import annotations

import html
import re

OPEN, CLOSE = "", ""  # private-use characters: they mark kept text and never reach the browser

# Inside tags, or told apart by the tag beside them. All of these hold a ' or a <, which escaped text cannot.
MARKUP = {
    "aria-label='답'": "aria-label='Answer'",
    "<html lang='ko'>": "<html lang='en'>",
    "title='업무'": "title='Work'",
    "title='대리인'": "title='Agent'",
    "처리<textarea": "Action<textarea",
}

# Strings holding numbers or ids, most specific first. Run before TABLE, over one piece of text between tags.
PATTERNS = [
    (re.compile(ko), en)
    for ko, en in {
        r"^(.+)에서 규칙 찾기$": r"Find rules in \1",
        r"내 기록 (\d+)건": r"My records: \1",
        r"(\d+)번 조작": r"actions: \1",
        r": (\d+)건 \(읽기 전용\)": r": \1 (read-only)",
        r"기록 (\S+)를 지웠습니다\. 이 기록을 근거로 쓴 카드에는 표시가 남습니다\.": r"Record \1 deleted. Cards that cited it keep a note of that.",
        r"(R\d+)을 고친 대로 적용했습니다\. 규칙 버전 v(\d+)\.": r"\1 applied as edited. Rule version v\2.",
        r"규칙 카드 \(규칙 버전 v(\d+), 적용 (\d+)개\. 기록에서 찾은 규칙은 바로 적용합니다\. 기록이 정하지 못한 점만 답해 주세요\)": r"Rules (version v\1, \2 at work. Rules found in your records apply right away. Answer only what the records could not settle)",
        r"규칙 글 없이 어댑터 (\S+)으로 일했습니다\.": r"Worked with adapter \1. No written rules in its prompt.",
        r"(R\d+)에 답을 적었습니다\. 규칙 버전 v(\d+)\.": r"\1: your answer is in. Rule version v\2.",
        r"근거 기록 중 (\d+)건이 지워졌습니다\.": r"Records behind this card that were deleted: \1.",
        r"근거 기록 (\d+)건": r"records behind it: \1",
        r"새 일 \(규칙 v(\d+)로 처리합니다\)": r"New work (handled with rules v\1)",
        r"([\d,]+)개 \$": r"\1 at $",
        r"· 규칙 v(\d+)": r"· rules v\1",
        r"규칙이 바뀌었습니다\(지금 v(\d+)\)\. 이 초안은 다시 확인해야 합니다\.": r"The rules changed (now v\1). This draft must be re-checked.",
        r"쓴 정보 (\d+)건": r"Information used: \1",
        r"바뀜\((R\d+) 적용\)": r"changed (\1 applied)",
        r"(R\d+) 고친 내용 미리 돌려 보기": r"Preview of the \1 edit",
        r"겹칠 수 있는 확인된 카드: (.*?)\. 같은 상황을 다룬다면 함께 고쳐야 결과가 한쪽으로 정해집니다\.": r"Confirmed cards that may overlap: \1. If they cover the same situation, edit them together so the result is settled one way.",
        # the learning card and autolearn's steps
        r": 기록 (\d+)건에서 학습, 시험 ([\d, ]+)/(\d+), 위험 ([\d, ]+)건\. 규칙 글 없이 일합니다\.": r": learned from \1 tickets, test \2 of \3 right, \4 dangerous. It works with no written rules.",
        r"규칙이 어댑터 (\S+)보다 새것이라, 다음 학습까지 글로 된 규칙으로 일합니다\.": r"The rules are newer than adapter \1, so work follows the written rules until the next training.",
        r"기록 중, 다음 학습까지 (\d+)건": r"recording, \1 more until the next training",
        r"(\S+)는 켜지 않았습니다\(전보다 나쁨\)": r"\1 not turned on (it did not pass its check)",
        r"(\S+) 켰습니다": r"\1 turned on",
        # guard()'s details
        r"(\d+)건 승인 대기": r"\1 waiting for approval",
        r"(\S+) 단가가 가격표보다 높습니다": r"\1 is priced above list",
        r"(\S+) 단가가 최저 허용 단가보다 낮습니다": r"\1 is priced under the floor",
        r"(\S+) 할인이 권한을 넘습니다\(사장님 동의 없음\)": r"\1 discount is beyond authority (no okay from the owner)",
        r"(\S+)는 (\S+)까지 재고가 모자랍니다": r"\1 is short of stock for \2",
        r"첨부 견적 (\S+)": r"Attached quote \1",
        r"견적 (Q-[\w-]+)": r"Quote \1",
        # the desk
        r"가격표에서 (\d+)%까지는 직접 깎아 줄 수 있습니다\. 그 이상은 사장님께 여쭙니다\. 최저 허용 단가 아래로는 안 됩니다\.": r"You can take up to \1% off list yourself. Beyond that, ask the owner. Never under the floor price.",
        r"이 티켓 포함 (\d+)건 남음": r"\1 left, this one included",
    }.items()
]

TABLE = {
    # both panes
    "평소처럼 일하면, 적힌 적 없는 업무 규칙을 찾아 모르는 것만 묻고, 승인할 일을 준비해 둡니다": "Work as usual: it finds the rules nobody wrote down, asks only what it can't tell, and prepares the work for your approval",
    "업무 화면 없음": "No work screen",
    "업무": "Work",
    "대리인": "Agent",
    "● 기록 중": "● Recording",
    "○ 이 티켓은 기록하지 않음": "○ This ticket is not recorded",
    "다음 티켓부터 끄기": "Stop recording from the next ticket",
    "다음 티켓부터 기록": "Record from the next ticket",
    # messages and job errors
    "서버가 다시 시작되어 끊겼습니다": "Cut off when the server restarted",
    "같은 일이 이미 처리 중입니다.": "The same job is already running.",
    "규칙을 찾을 기록이 없습니다": "No records to find rules in",
    "모델이 빈 답을 냈습니다": "The model gave an empty answer",
    "지금 기록 중인 티켓은 지울 수 없습니다. 끝낸 뒤에 지우십시오.": "The ticket being recorded cannot be deleted. Finish it first.",
    "검사를 통과하지 못한 초안은 승인할 수 없습니다.": "A draft that did not pass the checks cannot be approved.",
    "이 초안은 바뀌기 전 규칙으로 만들었습니다. 다시 확인했다고 표시한 뒤 승인하십시오.": "This draft was made under the earlier rules. Mark it as re-checked, then approve it.",
    # the learning card
    "켜진 어댑터 ": "Active adapter ",
    "아직 켜진 어댑터가 없습니다.": "No adapter is active yet.",
    "학습: ": "Learning: ",
    "데이터 만드는 중": "building the data",
    "학습 중": "training",
    "시험 중": "checking",
    # records
    "기록 (기록 스위치: ": "Records (recording switch: ",
    ", 왼쪽 화면 위에서 켜고 끕니다)": ", set at the top of the left pane)",
    "켜짐": "on",
    "꺼짐": "off",
    "진행 중": "in progress",
    "이 기록 지우기": "Delete this record",
    "가상 직원 C의 기록": "Virtual worker C's records",
    # rule cards
    "내 기록": "my records",
    "미리 뽑아 둔 Glimmer 카드 불러오기": "Load the cards Glimmer proposed earlier",
    "아직 카드가 없습니다.": "No cards yet.",
    "담당자가 정함": "set by the person",
    "기록에서 찾음": "found in the records",
    "상황: ": "Situation: ",
    "처리: ": "Action: ",
    "상황 → ": "Situation → ",
    "처리 → ": "Action → ",
    "상황": "Situation",
    "기록만으로는 아직 모르는 것:": "What the records alone do not settle yet:",
    "고친 내용 (아직 적용 안 됨)": "Edit (not applied yet)",
    "정해 둔 사례로 미리 돌려 보기": "Preview on the fixed cases",
    "적용": "Apply",
    "버리기": "Discard",
    "고치기": "Edit",
    "고친 내용 저장(적용 전)": "Save the edit (not applied yet)",
    "확인 전": "Not confirmed",
    "바로 적용": "Applied",
    "답하기": "Answer",
    "확인됨": "Confirmed",
    "더 물어봐야 함": "Ask first",
    "뺌": "Dropped",
    # new work: the cases' titles, then what a job shows
    "만료된 견적의 재주문": "Reorder of an expired quote",
    "재주문 견적에 가격 반발": "Pushback on a reorder quote's price",
    "재고 부족, 날짜를 꼭 맞춰야 함": "Short of stock, the date must be met",
    "재고 부족": "Short of stock",
    "출고일이 없는 요청": "Request with no ship date",
    "규칙 밖 질문이 붙은 주문": "Order with a question no rule covers",
    "모두 처리": "Work all",
    "처리 중": "working",
    "처리": "Work",
    "% 할인)": "% off)",
    " 출고 ": ", ships ",
    "초안, 승인 대기": "Draft, waiting for approval",
    "담당자에게 질문": "Question for the person",
    "멈춤(초안도 질문도 없음)": "Stopped (no draft, no question)",
    "멈춤(너무 오래 걸림)": "Stopped (took too long)",
    "오류": "Error",
    "적용 규칙: ": "Rules used: ",
    "적용한 규칙: ": "Rules used: ",
    "응답 없음": "no answer",
    "없음": "none",
    "작업 (최근 것이 위)": "Jobs (newest first)",
    "승인함": "Approved",
    "거절함": "Rejected",
    "(확인하지 않은 규칙 번호: ": "(rule numbers not confirmed: ",
    "담당자에게 묻습니다: ": "Question for the person: ",
    "최저 허용 단가까지 깎아도 됨": "OK to go down to the floor price",
    "답하고 다시 처리": "Answer and work it again",
    "첫 초안은 검사(": "The first draft failed the checks (",
    ")에 걸려 이유를 알려 주고 한 번 다시 쓰게 했습니다": "), so the model was told why and wrote it once more",
    "바뀐 규칙을 보고 다시 확인함, 승인": "Re-checked against the changed rules, approve",
    "승인(보낼 곳으로 넘김)": "Approve (pass on for sending)",
    "검사를 통과하지 못해 승인할 수 없습니다.": "It did not pass the checks, so it cannot be approved.",
    "거절": "Reject",
    # a preview
    "같음": "same",
    "바뀜(고친 규칙과 무관한 차이)": "changed (not because of the edited rule)",
    "사례": "Case",
    "지금 규칙": "Rules now",
    "고친 뒤": "After the edit",
    "차이": "Difference",
    # guard()'s checks
    "답장 초안": "Reply draft",
    "가격은 견적으로만": "Prices only on a quote",
    "견적 없이 가격을 적은 답장이 있습니다": "A reply states a price without a quote",
    "견적 밖 가격 없음": "No price outside a quote",
    "원가 비공개": "Cost stays private",
    "내부 기준 비공개": "Inner limits stay private",
    "답장에 내부 기준이 있습니다": "The reply mentions the shop's inner limits",
    "내부 기준 없음": "No inner limits mentioned",
    "원가가 답장에 있습니다: ": "The reply shows a cost: ",
    "원가 없음": "No cost shown",
    "금액은 데스크 값": "Amounts come from the desk",
    "데스크가 준 적 없는 금액: ": "Amounts the desk never gave: ",
    "모두 데스크 값": "All from the desk",
    "견적번호 확인": "Quote ids exist",
    "이 티켓에 없는 견적번호: ": "Quote ids not on this ticket: ",
    "모두 있는 번호": "All exist",
    "가격, 권한, 최저 단가, 출고일별 재고 모두 맞음": "Price, authority, floor price and stock by ship date all fine",
    # the desk
    "견적 데스크": "Quote desk",
    "남은 티켓이 없습니다. 수고하셨습니다.": "No tickets left. Thank you.",
    "티켓 ": "Ticket ",
    "할인 권한 없음. 가격표보다 낮게 주려면 먼저 사장님께 여쭤야 합니다.": "No discount authority. To go under list, ask the owner first.",
    "지금까지 이 손님에게 보낸 것: ": "Sent to this customer so far: ",
    "아직 없음": "nothing yet",
    ", 손님:": ", customer:",
    ", 가게:": ", shop:",
    "앞서 오간 메일 (오래된 것이 위)": "Earlier mail (oldest first)",
    "새 메일": "New mail",
    "보낸 영어: ": "Sent: ",
    "아직 한 일이 없습니다.": "Nothing done yet.",
    "한 일 (최근 것이 위)": "Done so far (newest first)",
    "오늘 ": "Today ",
    "이 티켓 끝내기": "Finish this ticket",
    "사장님께 보낼 질문": "Question for the owner",
    "손님에게 보낼 답장": "Reply to the customer",
    " (견적 ": " (quote ",
    " 첨부)": " attached)",
    ": 영어로 옮긴 것을 확인하십시오": ": read it once more before it goes out",
    "내용이 비어 있습니다": "The message is empty",
    "이대로 보내기": "Send as written",
    "다시 쓰기": "Rewrite",
    "견적 찾기": "Look up quote",
    "지난 견적": "Earlier quote",
    "견적 번호": "Quote id",
    "찾기": "Search",
    "품목 검색": "Catalog search",
    "품명이나 품번(한국어 가능, 비우면 전체)": "Item name or SKU (empty for all)",
    "원가 보기": "See cost",
    "재고 보기": "See stock",
    "품목": "Item",
    "품번": "SKU",
    "수량": "Qty",
    "단가": "Unit price",
    "원가": "Unit cost",
    "출고일": "Ship by",
    "가격 계산": "Calculate price",
    "견적 만들기 (줄마다 출고일, YYYY-MM-DD)": "Create quote (each line has its ship date, YYYY-MM-DD)",
    "견적 만들기": "Create quote",
    "영어로 옮기기": "Review before sending",
    "손님에게 답장 (한국어로 쓰십시오)": "Reply to the customer",
    "첨부할 견적 번호": "Quote id to attach",
    "사장님께 여쭙기 (손님은 보지 않습니다)": "Ask the owner (the customer does not see this)",
}

_SKIP = re.compile(f"({OPEN}.*?{CLOSE}|<textarea\\b.*?</textarea>|<[^>]*>)", re.DOTALL)
_PHRASE = re.compile("|".join(re.escape(ko) for ko in sorted(TABLE, key=len, reverse=True)))


def keep(text) -> str:
    """Escape someone else's words and mark them, so english() leaves them exactly as they were written."""
    return OPEN + html.escape(str(text).replace(OPEN, "").replace(CLOSE, "")) + CLOSE


def _words(text: str) -> str:
    """One piece of the screen's own text. One pass, longest phrase first, so no English is ever replaced again."""
    for pattern, en in PATTERNS:
        text = pattern.sub(en, text)
    return _PHRASE.sub(lambda m: TABLE[m.group()], text)


def english(page: str) -> str:
    """A finished page in English. Run it once per page: the marks are gone afterwards."""
    for ko, en in MARKUP.items():
        page = page.replace(ko, en)
    parts = _SKIP.split(page)  # the pieces at odd places are the ones to skip
    parts[::2] = [_words(p) for p in parts[::2]]
    return "".join(parts).replace(OPEN, "").replace(CLOSE, "")


def boundary(lang: str):
    """(how someone else's words are escaped, what a finished page goes through). Korean leaves both as they were."""
    return (keep, english) if lang == "en" else (html.escape, str)
