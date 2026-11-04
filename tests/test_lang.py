"""The screen in English: no Korean left in any state, and nothing that is not the screen's own words changed."""

import json
import re
import urllib.request

from test_app import app_for, asking_model, owner_model, post
from test_autolearn import TALLY, record_tickets

from testbed.loop_cases import TITLES, make_cases
from testbed.tasks2 import make_tasks
from workalike import autolearn, lang
from workalike.app import OUTCOME
from workalike.desk import unit_price_for
from workalike.person_desk import ERRORS, ERRORS_EN, AsWritten, Korean, PersonDesk, as_written, describe_en
from workalike.product import HINTS
from workalike.record import Recorder, RecordSwitch, read_events
from workalike.rules import STATES
from workalike.web import serve_in_background

KOREAN_WORDS = "승인 거절 처리 새 메일"  # phrases of the table, written by someone else: they must stay


def no_korean(page):
    assert re.findall(r".{0,40}[가-힣]+.{0,40}", page) == []
    assert lang.OPEN not in page and lang.CLOSE not in page
    return page


def ticket(messages):
    return re.search(r"Ticket (T-\d+)", messages[1]["content"]).group(1)


def quoting_model(messages, tools):
    """Looks things up with every tool, quotes 1% under list, and attaches the quote to a reply."""
    if not tools:
        return {"content": "Rules used: R1, R9"}
    t = ticket(messages)
    results = [m["content"] for m in messages if m.get("role") == "tool"]
    made = [json.loads(r)["quote_id"] for r in results if '"lines"' in r and '"total"' in r]
    if made:
        return {
            "tool_calls": [
                {
                    "name": "send_reply",
                    "arguments": {"ticket_id": t, "message": "Here is your quote.", "quote_id": made[0]},
                }
            ]
        }
    if results:
        cost = next(json.loads(r)["unit_cost"] for r in results if '"as_of"' in r)
        line = {
            "sku": "MUG-11W",
            "qty": 10,
            "unit_price": round(unit_price_for(cost, 10) * 0.99, 2),
            "ship_by": "2027-01-10",
        }
        return {"tool_calls": [{"name": "create_quote", "arguments": {"ticket_id": t, "lines": [line]}}]}
    return {
        "tool_calls": [
            {"name": n, "arguments": a}
            for n, a in (
                ("lookup_quote", {"quote_id": "Q-4232"}),
                ("search_catalog", {"query": "mug"}),
                ("get_cost", {"sku": "MUG-11W"}),
                ("get_cost", {"sku": "NOPE"}),
                ("check_stock", {"sku": "MUG-11W", "qty": 10}),
                ("check_stock", {"sku": "MUG-11W", "qty": 99999}),
                ("calc_price", {"sku": "MUG-11W", "qty": 10, "unit_cost": 2.39}),
            )
        ]
    }


def stray_model(messages, tools):
    """A reply with a price nobody gave, every time: held back, written again, and still not approvable."""
    if not tools:
        return {"content": "Rules used: none"}
    return {
        "tool_calls": [
            {"name": "send_reply", "arguments": {"ticket_id": ticket(messages), "message": "That is $123.45."}}
        ]
    }


def en_desk(tmp_path, n=2, keep=lambda t: t["family"] == "standard"):
    tasks = [t for t in make_tasks(11, "train") if keep(t)][:n]
    sessions = [
        {"episode": f"e-{i}", "view": t["view"], "state": t["hidden"]["state"], "task_id": t["task_id"], "actor": "me"}
        for i, t in enumerate(tasks)
    ]
    switch = RecordSwitch(Recorder(tmp_path / "records.jsonl"))
    return tasks, switch, PersonDesk(sessions, switch, AsWritten(), as_written, lang="en")


def worked(app, model_case="0"):
    app.post("/agent/work", {"case": model_case})
    app.todo.join()
    return max(app.jobs.values(), key=lambda j: j["id"])


def test_cards_loaded_confirmed_and_settled(tmp_path):
    app = app_for(tmp_path, lang="en")
    assert "Rules (version v0, 0 at work" in no_korean(app.page_for("/agent"))
    assert "<html lang='en'>" in no_korean(app.page_for("/")) and "title='Agent'" in app.page_for("/")
    assert "No work screen" in no_korean(app.page_for("/work"))
    app.post("/agent/load", {})
    app.todo.join()
    assert "found in the records" in no_korean(app.page_for("/agent"))
    app.post("/agent/status", {"card": "R1", "status": "confirmed"})
    no_korean(app.page_for("/agent"))
    app.book.settle("R1", "how soon is too soon", "이틀이면 충분합니다", "Two days is enough.")
    page = no_korean(app.page_for("/agent"))
    assert "Two days is enough." in page and "set by the person" in page and "how soon is too soon</li>" not in page


def test_a_pair_shows_its_english_and_a_card_edited_in_english_stays_unsaved_until_changed(tmp_path):
    app = app_for(tmp_path, lang="en")
    app.book.add_proposed(
        [{"situation": "the date is missing", "action": "ask for it", "unknowns": ["how soon"]}],
        "test",
        {0: {"situation": "날짜가 없음", "action": "물어봄", "unknowns": ["얼마나 빨리"]}},
    )
    page = no_korean(app.page_for("/agent"))
    assert (
        "<textarea name='situation'>the date is missing</textarea>" in page
        and "how soon<form method='post' action='/agent/settle'" in page
    )
    app.post("/agent/edit", {"card": "R1", "situation": "the date is missing", "action": "ask for it"})
    assert app.book.card("R1")["pending"] is None  # what the textarea held is not an edit


def test_a_work_job_with_a_draft_quote_and_what_it_looked_up(tmp_path):
    app = app_for(tmp_path, quoting_model, lang="en")
    app.post("/agent/load", {})
    app.todo.join()
    app.post("/agent/status", {"card": "R1", "status": "confirmed"})
    job = worked(app)
    assert job["outcome"] == "draft" and job["quotes"]
    assert next(iter(job["quotes"].values()))[0]["off"] > 0
    page = no_korean(app.page_for("/agent"))
    assert (
        "% off), ships 2027-01-10" in page and "Draft, waiting for approval" in page and "Here is your quote." in page
    )
    assert "Attached quote Q-" in page and "Information used: 8" in page and "NOPE is not in the catalog." in page
    assert "✓ Reply draft: 1 waiting for approval" in page and "✓ Quote Q-" in page
    assert "Rules used: R1, R9 <span class='bad'>(rule numbers not confirmed: R9)</span>" in page
    app.post("/agent/approve", {"job": job["id"]})
    assert "Approved" in no_korean(app.page_for("/agent")) and job["decision"] == "approved"
    other = worked(app, "1")
    app.post("/agent/reject", {"job": other["id"]})
    assert "Rejected" in no_korean(app.page_for("/agent"))


def test_a_held_back_draft_and_a_stale_one(tmp_path):
    app = app_for(tmp_path, stray_model, lang="en")
    job = worked(app)
    assert job["first"] and not all(c["ok"] for c in job["checks"])
    page = no_korean(app.page_for("/agent"))
    assert (
        "The first draft failed the checks (Prices only on a quote, Amounts come from the desk), so the model" in page
    )
    assert "✗ Amounts come from the desk: Amounts the desk never gave: $123.45" in page
    assert "It did not pass the checks, so it cannot be approved." in page
    app.post("/agent/approve", {"job": job["id"]})
    assert "A draft that did not pass" in no_korean(app.page_for("/agent"))
    assert {c["name"] for c in job["checks"]} >= {"가격은 견적으로만"}  # the data stays as guard() wrote it


def test_a_draft_made_under_older_rules_asks_for_a_recheck(tmp_path):
    app = app_for(tmp_path, lang="en")
    app.post("/agent/load", {})
    app.todo.join()
    job = worked(app)
    app.post("/agent/status", {"card": "R1", "status": "confirmed"})
    page = no_korean(app.page_for("/agent"))
    assert (
        "The rules changed (now v2). This draft must be re-checked." in page
        and "Re-checked against the changed" in page
    )
    app.post("/agent/approve", {"job": job["id"]})
    assert "made under the earlier rules" in no_korean(app.page_for("/agent"))


def test_a_question_waiting_for_the_person(tmp_path):
    app = app_for(tmp_path, owner_model, lang="en")
    asked = worked(app, "1")
    page = no_korean(app.page_for("/agent"))
    assert "Question for the person: May I discount?" in page and "Answer and work it again" in page
    app.post("/agent/answer", {"job": asked["id"], "says": "Yes, down to the floor.", "floor_ok": "1"})
    app.todo.join()
    assert no_korean(app.page_for("/agent")) and max(app.jobs.values(), key=lambda j: j["id"])["outcome"] == "draft"


def test_a_preview_of_an_edit(tmp_path):
    app = app_for(tmp_path, lang="en")
    cards = [
        {
            "situation": "a customer pushes back on a reorder price",
            "action": "offer a reorder discount within authority",
            "unknowns": [],
        },
        {
            "situation": "a customer asks for a lower price on a reorder",
            "action": "give a reorder discount within authority",
            "unknowns": [],
        },
    ]
    app.book.add_proposed(cards, "test")
    app.book.set_status("R2", "confirmed")
    app.post("/agent/edit", {"card": "R1", "action": "offer 5% off at once"})
    page = no_korean(app.page_for("/agent"))
    assert "Edit (not applied yet)" in page and "Action → offer 5% off at once" in page
    assert "<textarea name='action'>offer a reorder discount within authority</textarea>" in page
    app.post("/agent/preview", {"card": "R1"})
    app.todo.join()
    page = no_korean(app.page_for("/agent"))
    assert "Preview of the R1 edit" in page and "Confirmed cards that may overlap: R2." in page and "same" in page
    app.post("/agent/apply", {"card": "R1"})
    assert "R1 applied as edited. Rule version v2." in no_korean(app.page_for("/agent"))


def learned(tmp_path):
    """A real learning round's status file, with the steps autolearn writes, and a worse round after it."""
    records, versions = tmp_path / "r.jsonl", tmp_path / "me"
    record_tickets(records, 3)
    autolearn.learn_once(records, versions, lambda s, p: None, lambda n, p: TALLY, target=6)
    autolearn.learn_once(records, versions, lambda s, p: None, lambda n, p: {**TALLY, "good": 2}, target=6)
    return versions


def test_the_learning_card_with_an_active_adapter_and_with_a_resting_one(tmp_path):
    versions = learned(tmp_path)
    app = app_for(tmp_path, lang="en", versions=versions, adapter_model=lambda name: asking_model)
    page = no_korean(app.page_for("/agent"))
    assert "Active adapter <b>v1</b>: learned from 3 tickets, test 3 of 4 right, 0 dangerous." in page
    assert "Learning: v2 not turned on (it did not pass its check)" in page
    assert "building the data" in page and "training" in page and "checking" in page
    status = json.loads((versions / "status.json").read_text())
    status["active"]["rule_version"] = 99  # the rules moved on: the adapter rests
    status["state"] = "기록 중, 다음 학습까지 4건"
    (versions / "status.json").write_text(json.dumps(status, ensure_ascii=False))
    page = no_korean(app.page_for("/agent"))
    assert "The rules are newer than adapter v1" in page and "recording, 4 more until the next training" in page
    status["active"], status["state"] = None, "v1 켰습니다"
    (versions / "status.json").write_text(json.dumps(status, ensure_ascii=False))
    page = no_korean(app.page_for("/agent"))
    assert "No adapter is active yet." in page and "Learning: v1 turned on" in page


def test_the_messages_and_a_job_cut_off_by_a_restart(tmp_path):
    app = app_for(tmp_path, workers=0, lang="en")
    app.post("/agent/work", {"case": "2"})
    app.post("/agent/work", {"case": "2"})
    assert "already running" in no_korean(app.page_for("/agent"))
    again = app_for(tmp_path, workers=0, lang="en")
    assert "Cut off when the server restarted" in no_korean(again.page_for("/agent"))


def test_the_desk_with_a_ticket_under_way_needs_no_korean_file_and_no_translator(tmp_path):
    tasks, switch, desk = en_desk(tmp_path)
    app = app_for(tmp_path, desk=desk, switch=switch, lang="en")
    desk.badge = app.badge
    state, want = tasks[0]["hidden"]["state"], tasks[0]["grader"]["wants"][0]
    page = no_korean(app.page_for("/work"))
    assert tasks[0]["view"]["body"].splitlines()[0] in page and "Quote desk" in page and "● Recording" in page
    assert (
        "Stop recording from the next ticket" in page
        and "Finish this ticket" in page
        and "2 left, this one included" in page
    )
    for form in (
        {"tool": "lookup_quote", "quote_id": "Q-1"},
        {"tool": "search_catalog", "query": "mug"},
        {"tool": "get_cost", "sku": want["sku"]},
        {"tool": "get_cost", "sku": "NOPE"},
        {"tool": "check_stock", "sku": want["sku"], "qty": "1"},
        {"tool": "check_stock", "sku": want["sku"], "qty": "9999999"},
        {
            "tool": "calc_price",
            "sku": want["sku"],
            "qty": "5",
            "unit_cost": str(state["catalog"][want["sku"]]["unit_cost"]),
        },
        {"tool": "create_quote", "sku1": want["sku"], "qty1": "5", "price1": "0.01", "ship1": "2027-01-01"},
        {"tool": "create_quote", "sku1": want["sku"], "qty1": "5", "price1": "99", "ship1": "2027-01-01"},
    ):
        app.post("/act", form)
        no_korean(app.page_for("/work"))
    qid = desk.log[0]["result"]["quote_id"]
    page = app.page_for("/work")
    assert (
        "Search &#x27;mug&#x27;:" in page and "Failed: under the floor price" in page and f"Draft quote {qid}" in page
    )
    app.post("/draft", {"kind": "reply", "text": "It ships by October 22, 2026.", "quote_id": qid})
    page = no_korean(app.page_for("/work"))  # a month is not a number the "Korean" lacks
    assert (
        desk.draft["problems"] == []
        and "Send as written" in page
        and f"Reply to the customer (quote {qid} attached)" in page
    )
    assert page.count("It ships by October 22, 2026.") == 1
    app.post("/send", {})
    page = no_korean(app.page_for("/work"))
    assert (
        desk.log[0]["args"]["message"] == "It ships by October 22, 2026."
        and "Customer replied:" in page
        and "Sent: It ships" in page
    )
    app.post("/draft", {"kind": "owner", "text": "May I take 3% off?"})
    app.post("/send", {})
    assert "Owner replied:" in no_korean(app.page_for("/work")) and desk.log[0]["name"] == "ask_owner"
    app.post("/draft", {"kind": "reply", "text": ""})
    assert "The message is empty" in no_korean(app.page_for("/work"))
    app.post("/discard", {})
    assert "Delete this record" not in no_korean(app.page_for("/agent")) and "in progress" in app.page_for("/agent")
    app.post("/work/record", {"on": "0"})
    assert "Record from the next ticket" in no_korean(app.page_for("/work"))
    app.post("/finish", {})
    assert "This ticket is not recorded" in no_korean(app.page_for("/work"))
    page = no_korean(app.page_for("/agent"))
    assert "My records: 1" in page and "Delete this record" in page and "Owner replied:" in page
    app.post("/finish", {})
    assert "No tickets left." in no_korean(app.page_for("/work"))
    assert read_events(tmp_path / "records.jsonl")[0]["entrance"] == "web-en"


def test_a_ticket_with_an_earlier_thread_reads_in_english(tmp_path):
    tasks, switch, desk = en_desk(
        tmp_path, n=1, keep=lambda t: t["view"].get("thread") and t["view"].get("sent_so_far")
    )
    page = no_korean(desk.page())
    assert (
        tasks[0]["view"]["sent_so_far"] in page
        and "Earlier mail (oldest first)" in page
        and "customer:" in page
        and "shop:" in page
    )
    assert "New mail" in page and "<details>" not in page  # no second copy of a mail that is already English


def test_search_terms_work_as_typed(tmp_path):
    def no_translator(text):
        raise AssertionError("nothing is translated in English")

    _, switch, desk = en_desk(tmp_path, n=1)
    desk.translate_term = no_translator
    desk.post("/act", {"tool": "search_catalog", "query": "mug"})
    assert desk.log[0]["args"]["query"] == "mug" and desk.log[0]["result"]["items"]
    desk.post("/act", {"tool": "search_catalog", "query": "머그"})  # "머그" is in the Korean desk's glossary
    assert desk.log[0]["args"]["query"] == "머그" and not desk.log[0]["result"]["items"]


def test_someone_elses_korean_stays_exactly_as_written(tmp_path):
    def model(messages, tools):
        if not tools:
            return {"content": "Rules used: none"}
        return {
            "tool_calls": [
                {"name": "send_reply", "arguments": {"ticket_id": ticket(messages), "message": KOREAN_WORDS}}
            ]
        }

    app = app_for(tmp_path, model, lang="en")
    app.cases[1]["view"]["body"] = "고객: " + KOREAN_WORDS
    worked(app)
    page = app.page_for("/agent")
    assert f"<div class='reply'>{KOREAN_WORDS}<" in page and f"<td>고객: {KOREAN_WORDS}</td>" in page

    tasks, switch, desk = en_desk(tmp_path, n=1)
    desk.current["view"]["body"] = KOREAN_WORDS
    desk.post("/draft", {"kind": "reply", "text": KOREAN_WORDS})
    assert desk.draft["problems"] == [] and desk.draft["en"] == KOREAN_WORDS
    page = desk.page()
    assert f"<div class='body'>{KOREAN_WORDS}</div>" in page and f"<div>{KOREAN_WORDS}</div>" in page
    desk.post("/send", {})
    assert desk.log[0]["args"]["message"] == KOREAN_WORDS and f"Sent: {KOREAN_WORDS}" in desk.page()
    desk.post("/act", {"tool": "search_catalog", "query": "처리"})
    assert "Search &#x27;처리&#x27;" in desk.page()


def test_no_replacement_reaches_a_tag_a_textarea_or_kept_text():
    tag = "<input type='hidden' name='승인' value='처리'>"
    area = "<textarea name='says'>승인 거절</textarea>"
    assert (
        lang.english(f"{tag}{area}{lang.keep('거절 <b>')}<button>거절</button>")
        == f"{tag}{area}거절 &lt;b&gt;<button>Reject</button>"
    )
    assert lang.english(lang.keep("거절거절")) == "거절거절"  # a mark inside the text cannot end the protection
    assert all(re.search("[가-힣]", k) for k in lang.TABLE)  # so English can never match
    assert all(re.search("[가-힣]", p.pattern) for p, _ in lang.PATTERNS)


def test_every_label_the_data_can_hold_has_english():
    details = [
        "1건 승인 대기",
        "없음",
        "견적 없이 가격을 적은 답장이 있습니다",
        "견적 밖 가격 없음",
        "원가가 답장에 있습니다: $1.23",
        "원가 없음",
        "데스크가 준 적 없는 금액: $9.99",
        "모두 데스크 값",
        "이 티켓에 없는 견적번호: Q-1-1",
        "모두 있는 번호",
        "MUG-11W 단가가 가격표보다 높습니다; MUG-11W 단가가 최저 허용 단가보다 낮습니다",
        "MUG-11W 할인이 권한을 넘습니다(사장님 동의 없음); MUG-11W는 2026-10-10까지 재고가 모자랍니다",
        "가격, 권한, 최저 단가, 출고일별 재고 모두 맞음",
        "견적 Q-1-1",
        "ValueError: 규칙을 찾을 기록이 없습니다",
        "ValueError: 모델이 빈 답을 냈습니다",
        "가상 직원 C의 기록",
        "가상 직원 C의 기록에서 규칙 찾기",
    ]
    labels = [*OUTCOME.values(), *STATES.values(), *TITLES.values(), *(k for k in HINTS if k != "quote"), *details]
    for text in labels:
        no_korean(lang.english(text))
    assert lang.english("MUG-11W는 2026-10-10까지 재고가 모자랍니다") == "MUG-11W is short of stock for 2026-10-10"
    assert set(ERRORS_EN) == set(ERRORS)


def test_describe_en_says_when_a_short_item_can_ship():
    base = {
        "sku": "TSH-PRM",
        "name": "premium cotton t-shirt",
        "qty": 225,
        "on_hand": 113,
        "enough": False,
        "substitutes": [],
    }
    assert "113 now, the other 112 from the restock on 2026-10-03" in describe_en(
        "check_stock", {}, {**base, "restock_date": "2026-10-03", "restock_qty": 450}
    )
    assert "62 are still missing" in describe_en(
        "check_stock", {}, {**base, "restock_date": "2026-10-03", "restock_qty": 50}
    )
    assert "no restock planned" in describe_en("check_stock", {}, {**base, "restock_date": None, "restock_qty": 0})


def test_a_button_post_still_works_in_english(tmp_path):
    app = app_for(tmp_path, workers=1, lang="en")
    server = serve_in_background(app, port=0)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        page = urllib.request.urlopen(url + "/agent").read().decode()
        no_korean(page)
        form = re.search(r"<form method='post' action='/agent/load' class='inline'>(.*?)</form>", page).group(1)
        fields = dict(re.findall(r"name='(\w+)' value='([^']*)'", form))
        assert fields["csrf"] == app.token and "Load the cards Glimmer proposed earlier" in form
        assert post(url + "/agent/load", fields).endswith("/agent")
        app.todo.join()
        page = urllib.request.urlopen(url + "/agent").read().decode()
        form = re.search(r"<form method='post' action='/agent/status' class='inline'>(.*?)</form>", page).group(1)
        fields = dict(re.findall(r"name='(\w+)' value='([^']*)'", form))
        assert fields["card"] == "R1" and fields["status"] == "confirmed" and ">Confirmed</button>" in form
        assert post(url + "/agent/status", fields).endswith("/agent") and app.book.card("R1")["status"] == "confirmed"
    finally:
        server.shutdown()


def test_korean_stays_the_default_and_keeps_its_words(tmp_path):
    tasks = [t for t in make_tasks(11, "train") if t["family"] == "standard"][:1]
    sessions = [
        {
            "episode": "e-0",
            "view": tasks[0]["view"],
            "state": tasks[0]["hidden"]["state"],
            "task_id": tasks[0]["task_id"],
            "actor": "me",
        }
    ]
    switch = RecordSwitch(Recorder(tmp_path / "records.jsonl"))
    fixed = {"owner": [], "customer_defaults": ["", "", ""]}
    desk = PersonDesk(
        sessions, switch, Korean({"tickets": {}, "fixed": {"en": fixed, "ko": fixed}}), lambda ko: ("x", "fake")
    )
    app = app_for(tmp_path, desk=desk, switch=switch)
    desk.badge = app.badge
    app.post("/agent/load", {})
    app.todo.join()
    agent, work = app.page_for("/agent"), app.page_for("/work")
    assert (
        "<html lang='ko'>" in agent
        and "규칙 카드 (규칙 버전 v1, 적용 2개." in agent
        and "<summary>고치기</summary>" in agent
    )
    assert "상황<textarea name='situation'>the date is missing</textarea>처리<textarea name='action'>" in agent
    assert (
        "<b>견적 데스크</b>" in work
        and "영어 원문 보기" in work
        and "<button class='finish'>이 티켓 끝내기</button>" in work
    )
    assert lang.OPEN not in agent + work and lang.boundary("ko")[1](agent) == agent
    assert read_events(tmp_path / "records.jsonl")[0]["entrance"] == "web-ko"
    assert [c["title"] for c in make_cases()][0] == "만료된 견적의 재주문"
