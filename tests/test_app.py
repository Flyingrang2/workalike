"""The product loop end to end with fake models: record, cards, confirm, work, approve, edit, preview, apply."""

import json
import re
import urllib.error
import urllib.parse
import urllib.request

import pytest

from testbed.loop_cases import make_cases
from testbed.tasks2 import make_tasks
from workalike import product
from workalike.app import DeskApp
from workalike.person_desk import Korean, PersonDesk
from workalike.record import Recorder, RecordSwitch, read_events
from workalike.web import serve_in_background

CARDS = {
    "cards": [
        {
            "situation": "the date is missing",
            "action": "ask for it before pricing",
            "evidence": [1],
            "records": ["e-c1"],
            "unknowns": ["how soon is too soon"],
        },
        {
            "situation": "a reorder price went up",
            "action": "offer up to 3% off",
            "evidence": [2],
            "records": ["e-c2"],
            "unknowns": [],
        },
    ]
}


def asking_model(messages, tools):
    """Asks the customer one question about the date and stops; says which rules it followed when asked."""
    if not tools:
        return {"content": "Rules used: R1, R9"}
    ticket = re.search(r"Ticket (T-\d+)", messages[1]["content"]).group(1)
    return {
        "tool_calls": [{"name": "send_reply", "arguments": {"ticket_id": ticket, "message": "When do you need them?"}}]
    }


def owner_model(messages, tools):
    """Asks the owner first; once answered, writes a plain question to the customer."""
    if not tools:
        return {"content": "Rules used: R1"}
    ticket = re.search(r"Ticket (T-\d+)", messages[1]["content"]).group(1)
    if any(m.get("role") == "tool" and "owner_reply" in (m.get("content") or "") for m in messages):
        return {
            "tool_calls": [{"name": "send_reply", "arguments": {"ticket_id": ticket, "message": "Could you confirm?"}}]
        }
    return {"tool_calls": [{"name": "ask_owner", "arguments": {"ticket_id": ticket, "question": "May I discount?"}}]}


def app_for(tmp_path, model=asking_model, workers=2, desk=None, switch=None, **kw):
    cards = tmp_path / "cards.json"
    cards.write_text(json.dumps(CARDS))
    return DeskApp(
        tmp_path / "loop",
        lambda: model,
        make_cases(),
        desk,
        switch,
        tmp_path / "records.jsonl" if desk else None,
        cached_cards=cards,
        workers=workers,
        **kw,
    )


def test_the_whole_loop_with_a_fake_model(tmp_path):
    app = app_for(tmp_path)
    app.post("/agent/load", {})
    app.todo.join()
    assert [c["id"] for c in app.book.cards] == ["R1", "R2"] and "R1." in app.book.as_procedure()
    app.post("/agent/work", {"case": "0"})
    app.todo.join()
    [job] = [j for j in app.jobs.values() if j["kind"] == "work"]
    assert job["outcome"] == "draft" and job["version"] == 1 and "R1. When the date is missing" in job["rules"]
    assert job["used"] == [1, 9] and job["used_unknown"] == [9]  # R9 was never confirmed: shown as such
    assert all(c["ok"] for c in job["checks"])
    app.post("/agent/approve", {"job": job["id"]})
    app.post("/agent/approve", {"job": job["id"]})
    sent = (tmp_path / "loop" / "sent.jsonl").read_text().splitlines()
    assert len(sent) == 1 and json.loads(sent[0])["replies"][0]["message"] == "When do you need them?"

    # a second draft under v1, then an edit: pending until applied, previewed on both sides, then applied
    app.post("/agent/work", {"case": "1"})
    app.todo.join()
    waiting = max(app.jobs.values(), key=lambda j: j["id"])
    app.post("/agent/edit", {"card": "R2", "action": "offer up to 5% off at once"})
    assert app.book.card("R2")["pending"] and app.book.version == 1
    app.post("/agent/preview", {"card": "R2"})
    app.todo.join()
    preview = [j for j in app.jobs.values() if j.get("batch")]
    assert len(preview) == 2 * len(app.cases) and {j["side"] for j in preview} == {"before", "after"}
    assert "offer up to 5% off" in next(j for j in preview if j["side"] == "after")["rules"]
    assert "R2 (preview)" not in app.page_for("/agent") and "미리 돌려 보기" in app.page_for("/agent")
    app.post("/agent/apply", {"card": "R2"})
    assert app.book.version == 2 and app.book.card("R2")["action"]["by"] == "person"
    app.post("/agent/approve", {"job": waiting["id"]})  # made under v1: needs a re-check
    assert waiting["decision"] is None and "다시 확인" in app.message
    assert "이 초안은 다시 확인해야 합니다" in app.page_for("/agent")
    app.post("/agent/approve", {"job": waiting["id"], "recheck": "1"})
    assert waiting["decision"] == "approved"


def test_a_question_waits_for_the_person_and_the_answer_reruns_the_case(tmp_path):
    app = app_for(tmp_path, owner_model)
    app.post("/agent/work", {"case": "1"})
    app.todo.join()
    [asked] = app.jobs.values()
    assert asked["outcome"] == "asked" and asked["questions"][0]["question"] == "May I discount?"
    assert "답하고 다시 처리" in app.page_for("/agent")
    app.post("/agent/answer", {"job": asked["id"], "says": "Yes, down to the floor.", "floor_ok": "1"})
    app.todo.join()
    again = max(app.jobs.values(), key=lambda j: j["id"])
    assert again["outcome"] == "draft" and again["answer"]["policy"] == "floor_ok"


def test_a_repeated_press_makes_one_job(tmp_path):
    app = app_for(tmp_path, workers=0)
    app.post("/agent/work", {"case": "2"})
    app.post("/agent/work", {"case": "2"})
    assert len(app.jobs) == 1 and "이미 처리 중" in app.message


def test_recording_switch_applies_from_the_next_ticket_and_records_can_be_deleted(tmp_path):
    tasks = [t for t in make_tasks(11, "train") if t["family"] == "standard"][:3]
    sessions = [
        {"episode": f"e-{i}", "view": t["view"], "state": t["hidden"]["state"], "task_id": t["task_id"], "actor": "me"}
        for i, t in enumerate(tasks)
    ]
    switch = RecordSwitch(Recorder(tmp_path / "records.jsonl"))
    fixed = {"owner": [], "customer_defaults": ["", "", ""]}
    desk = PersonDesk(
        sessions, switch, Korean({"tickets": {}, "fixed": {"en": fixed, "ko": fixed}}), lambda ko: ("x", "fake")
    )
    app = app_for(tmp_path, desk=desk, switch=switch)
    desk.badge = app.badge
    assert "● 기록 중" in app.page_for("/work")
    app.post("/act", {"tool": "get_cost", "sku": tasks[0]["grader"]["wants"][0]["sku"]})
    app.post("/work/record", {"on": "0"})  # off from the next ticket
    assert "● 기록 중" in app.page_for("/work")
    app.post("/finish", {})
    assert "기록하지 않음" in app.page_for("/work")
    app.post("/act", {"tool": "get_cost", "sku": tasks[1]["grader"]["wants"][0]["sku"]})
    app.post("/work/record", {"on": "1"})
    app.post("/finish", {})  # ticket 3 is recorded again
    assert {e["ep"] for e in read_events(tmp_path / "records.jsonl")} == {"e-0", "e-2"}
    app.post("/agent/delete", {"ep": "e-2"})
    assert "지금 기록 중인 티켓은 지울 수 없습니다" in app.message
    app.post("/agent/delete", {"ep": "e-0"})
    assert {e["ep"] for e in read_events(tmp_path / "records.jsonl")} == {"e-2"}


def post(url, form):
    data = urllib.parse.urlencode(form).encode()
    with urllib.request.urlopen(urllib.request.Request(url, data=data, method="POST")) as r:
        return r.geturl()


def test_both_panes_are_served_and_posts_return_to_their_pane(tmp_path):
    app = app_for(tmp_path, workers=0)
    server = serve_in_background(app, port=0)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        root = urllib.request.urlopen(url + "/").read().decode()
        assert "src='/work'" in root and "src='/agent'" in root
        assert "규칙 카드" in urllib.request.urlopen(url + "/agent").read().decode()
        with pytest.raises(urllib.error.HTTPError) as err:
            post(url + "/agent/work", {"case": "0"})
        assert err.value.code == 403 and not app.jobs
        assert post(url + "/agent/work", {"case": "0", "csrf": app.token}).endswith("/agent")
        assert post(url + "/work/record", {"on": "0", "csrf": app.token}).endswith("/work")
    finally:
        server.shutdown()


def test_a_held_back_draft_is_told_why_and_written_once_more(tmp_path):
    tries = []

    def model(messages, tools):
        if not tools:
            return {"content": "Rules used: none"}
        ticket = re.search(r"Ticket (T-\d+)", messages[1]["content"]).group(1)
        held = "held back by the desk's checks" in messages[0]["content"]
        tries.append(held)
        text = "Could you confirm the date?" if held else "Our cost is $1.23, so it comes to $9.99."
        return {"tool_calls": [{"name": "send_reply", "arguments": {"ticket_id": ticket, "message": text}}]}

    app = app_for(tmp_path, model)
    app.post("/agent/work", {"case": "4"})
    app.todo.join()
    [job] = app.jobs.values()
    assert tries == [False, True] and job["first"]["drafts"][0]["message"].startswith("Our cost")
    assert {c["name"] for c in job["first"]["failed"]} == {"가격은 견적으로만", "금액은 데스크 값"}
    assert all(c["ok"] for c in job["checks"]) and "한 번 다시 쓰게 했습니다" in app.page_for("/agent")


def test_cards_that_share_words_with_an_edited_one_are_named(tmp_path):
    app = app_for(tmp_path)
    cards = [
        {
            "situation": "a customer pushes back on a reorder price",
            "action": "offer a reorder discount within authority",
            "evidence": [],
            "records": [],
            "unknowns": [],
        },
        {
            "situation": "a customer asks for a lower price on a reorder",
            "action": "give a reorder discount within authority",
            "evidence": [],
            "records": [],
            "unknowns": [],
        },
        {
            "situation": "stock is short",
            "action": "offer split shipment",
            "evidence": [],
            "records": [],
            "unknowns": [],
        },
    ]
    app.book.add_proposed(cards, "test")
    for cid in ("R1", "R2", "R3"):
        app.book.set_status(cid, "confirmed")
    assert app.overlapping("R1") == ["R2"]


def test_the_agent_pane_polls_instead_of_reloading_and_the_stamp_moves_with_new_results(tmp_path):
    app = app_for(tmp_path)
    page = app.page_for("/agent")
    assert "http-equiv='refresh'" not in page and f'const shown = "{app.status()["stamp"]}"' in page
    before = app.status()["stamp"]
    app.post("/agent/work", {"case": "4"})
    app.todo.join()
    assert app.status()["stamp"] != before


def test_a_draft_that_still_fails_the_checks_cannot_be_approved_and_nothing_goes_out_before_approval(tmp_path):
    def stray(messages, tools):
        if not tools:
            return {"content": "Rules used: none"}
        ticket = re.search(r"Ticket (T-\d+)", messages[1]["content"]).group(1)
        return {
            "tool_calls": [{"name": "send_reply", "arguments": {"ticket_id": ticket, "message": "That is $123.45."}}]
        }

    app = app_for(tmp_path, stray)
    app.post("/agent/work", {"case": "0"})
    app.todo.join()
    [job] = [j for j in app.jobs.values() if j["kind"] == "work"]
    assert job["first"] and not all(c["ok"] for c in job["checks"]) and "job" not in job["first"]
    assert not (tmp_path / "loop" / "sent.jsonl").exists()
    app.post("/agent/approve", {"job": job["id"]})
    assert job["decision"] is None and not (tmp_path / "loop" / "sent.jsonl").exists()


def test_what_the_person_settled_is_shown_as_theirs_and_no_longer_as_unknown(tmp_path):
    app = app_for(tmp_path)
    app.post("/agent/load", {})
    app.todo.join()
    app.post("/agent/status", {"card": "R1", "status": "confirmed"})
    app.book.settle("R1", "how soon is too soon", "이틀이면 충분합니다", "Two days is enough.")
    page = app.page_for("/agent")
    assert "이틀이면 충분합니다" in page and "담당자가 정함" in page and "how soon is too soon</li>" not in page
    assert "Two days is enough." in app.book.as_procedure()


def test_the_active_adapter_works_with_no_rules_and_its_state_is_shown(tmp_path):
    from workalike.envs import V2

    seen = {}

    def adapter(messages, tools):
        seen.update(system=messages[0]["content"])
        return asking_model(messages, tools)

    versions = tmp_path / "me"
    versions.mkdir()
    (versions / "status.json").write_text(
        json.dumps(
            {
                "state": "v1 켰습니다",
                "steps": [{"t": "t", "what": "v1 켰습니다"}],
                "active": {
                    "version": "v1",
                    "name": "me-v1",
                    "path": "p",
                    "tickets": 12,
                    "rows": 40,
                    "train_seconds": 9,
                    "check": {"good": 3, "tickets": 4, "dangerous": 0},
                },
            }
        )
    )
    app = app_for(tmp_path)
    app.versions, app.adapter_model = versions, lambda name: adapter
    app.post("/agent/load", {})
    app.todo.join()
    app.post("/agent/status", {"card": "R1", "status": "confirmed"})
    app.post("/agent/work", {"case": "0"})
    app.todo.join()
    [job] = [j for j in app.jobs.values() if j["kind"] == "work"]
    assert job["adapter"] == "me-v1" and job["outcome"] == "draft" and seen["system"] == V2.system_prompt
    assert "R1. When" not in seen["system"]
    page = app.page_for("/agent")
    assert "켜진 어댑터" in page and "학습: " in page


def test_found_rules_work_at_once_and_only_the_open_points_are_asked(tmp_path):
    app = app_for(tmp_path)
    app.post("/agent/load", {})
    app.todo.join()
    card = app.book.cards[0]
    assert card["status"] == "applied" and app.book.version == 1 and card["situation"]["by"] == "records"
    text = app.book.as_procedure()
    assert "They have not gone through each one" in text and "R1." in text
    page = app.page_for("/agent")
    assert "/agent/settle" in page and "적용 2개" in page
    question = card["unknowns"][0]["en"]
    app.post("/agent/settle", {"card": "R1", "question": question, "answer": "Two days is enough."})
    assert app.book.version == 2 and "Two days is enough." in app.book.as_procedure()
    assert app.book.cards[0]["settled"][0]["by"] == "person"


def test_a_settled_answer_must_answer_a_question_the_card_has(tmp_path):
    app = app_for(tmp_path)
    app.post("/agent/load", {})
    app.todo.join()
    before = app.book.version
    app.post("/agent/settle", {"card": "R1", "question": "something else", "answer": "Yes."})
    app.post("/agent/settle", {"card": "R1", "question": app.book.cards[0]["unknowns"][0]["en"], "answer": "  "})
    assert app.book.version == before and "settled" not in app.book.cards[0]


def test_an_adapter_trained_on_the_products_prompt_keeps_it(tmp_path):
    seen = []

    def model(messages, tools):
        seen.append(messages[0]["content"])
        return {
            "content": None,
            "tool_calls": [
                {
                    "name": "send_reply",
                    "arguments": {
                        "ticket_id": messages[1]["content"].split("Ticket ")[1].split()[0],
                        "message": "When do you need them?",
                    },
                }
            ],
        }

    learned = tmp_path / "learned"
    learned.mkdir()
    (learned / "status.json").write_text(
        json.dumps(
            {
                "state": "x",
                "steps": [],
                "active": {
                    "version": "p",
                    "name": "p",
                    "path": "a",
                    "rule_version": 0,
                    "prompt": "product",
                    "tickets": 155,
                    "check": {"good": "21, 20", "tickets": 24, "dangerous": "0, 1"},
                },
            }
        )
    )
    app = app_for(tmp_path, adapter_model=lambda name: model, versions=learned)
    assert app.active_adapter()["name"] == "p"
    page = app.page_for("/agent")
    assert "21, 20/24" in page and "위험 0, 1건" in page
    app.post("/agent/work", {"case": "0"})
    app.todo.join()
    assert seen and seen[0].startswith(product.PRODUCT_SYSTEM) and "Rules found" not in seen[0]
