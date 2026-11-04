"""The person's stopwatch run: the same grader as the LoRA, and the wait for a translator is not the person's time."""

import json
from collections import Counter

from testbed.person_time import PersonTime, load_lora, load_sheet, make_plan, summary
from workalike.record import Recorder
from workalike.sft import read_jsonl

TASKS, ANSWERS = load_sheet()
LORA = load_lora()
TEST = read_jsonl("baselines/test24b/tickets.jsonl")
KO = {"tickets": {}, "lora": {}, "rules": "규칙"}


def desk_for(tmp_path, items, translate=lambda ko: ("Hello, here is your quote.", "fake")):
    return PersonTime(items, TASKS, ANSWERS, LORA, KO, Recorder(tmp_path / "r.jsonl"), translate, translate, tmp_path)


def by_key(key):
    return next(t for t, a in ANSWERS.items() if a["key"] == key)


def quote(desk, task_id):
    w = LORA[task_id]["quotes"]
    qid, q = next(iter(w.items()))
    line = q["lines"][0]
    desk.post(
        "/act",
        {
            "tool": "create_quote",
            "sku1": line["sku"],
            "qty1": str(line["qty"]),
            "price1": str(line["unit_price"]),
            "ship1": line["ship_by"],
        },
    )
    return desk.log[0]["result"]["quote_id"]


def test_the_plan_is_even_fixed_and_keeps_the_sheets_mix():
    plan = make_plan(TEST, ANSWERS)
    assert plan == make_plan(TEST, ANSWERS)
    assert sorted(p["task_id"] for p in plan) == sorted(t["task_id"] for t in TEST)
    assert Counter(p["mode"] for p in plan) == {"hand": 12, "review": 12}
    first = plan[:12]
    assert Counter(p["mode"] for p in first) == {"hand": 6, "review": 6}
    assert Counter(p["group"] for p in first)["prepare"] >= 4


def test_a_hand_ticket_is_graded_by_the_same_grader_and_translation_is_not_counted(tmp_path):
    task_id = by_key("date-2-given")
    clock = {"waited": 0}

    def slow(ko):
        import time

        time.sleep(0.2)
        clock["waited"] += 0.2
        return f"Hello, here is quote {clock['qid']}.", "fake"

    desk = desk_for(tmp_path, [{"task_id": task_id, "mode": "hand"}], slow)
    assert "시작" in desk.page()
    desk.post("/start", {})
    clock["qid"] = quote(desk, task_id)
    desk.post("/draft", {"kind": "reply", "text": f"안녕하세요, 견적 {clock['qid']}입니다.", "quote_id": clock["qid"]})
    desk.post("/send", {})
    row = desk.last
    assert row["verdict"] == "ready", row["reasons"]
    assert row["machine_seconds"] >= 0.2 and row["seconds"] < 0.2
    assert row["lora"]["verdict"] == LORA[task_id]["verdict"]
    assert json.loads((tmp_path / "results.jsonl").read_text())["task_id"] == task_id
    assert "이 문의의 답" in desk.page()
    desk.post("/next", {})
    assert "결과" in desk.page()


def test_a_held_draft_can_be_written_once_more_like_the_models(tmp_path):
    task_id = by_key("date-2-given")
    texts = iter(["We can do this at $1.23 each.", "We can do this at $1.23 each."])
    desk = desk_for(tmp_path, [{"task_id": task_id, "mode": "hand"}], lambda ko: (next(texts), "fake"))
    desk.post("/start", {})
    for _ in range(2):
        assert desk.phase == "work"
        desk.post("/draft", {"kind": "reply", "text": "단가는 $1.23입니다", "quote_id": ""})
        desk.post("/send", {})
        if desk.phase == "work":
            assert "데스크 검사에 막혔습니다" in desk.page()
    assert desk.last["verdict"] == "failed" and desk.last["rewrote"]


def test_a_review_records_the_decision_and_a_held_draft_cannot_be_approved(tmp_path):
    held = next(t for t, r in LORA.items() if r["verdict"] == "failed")
    good = next(t for t, r in LORA.items() if r["verdict"] == "ready")
    desk = desk_for(tmp_path, [{"task_id": held, "mode": "review"}, {"task_id": good, "mode": "review"}])
    desk.post("/start", {})
    assert "데스크 검사에 막혀" in desk.page() and "/approve" not in desk.page()
    desk.post("/approve", {})
    assert desk.phase == "work"
    desk.post("/fix", {"reason": "다시"})
    assert desk.last["decision"] == "fix" and desk.last["reason"] == "다시"
    desk.post("/next", {})
    desk.post("/start", {})
    assert "/approve" in desk.page()
    desk.post("/approve", {})
    s = summary(desk.rows, LORA)["review"]
    assert s["tickets"] == 2 and s["caught"] == 1 and s["missed"] == 0 and s["approved"] == 1


def test_a_pause_is_not_counted_and_a_restart_keeps_finished_tickets(tmp_path):
    task_id = by_key("date-2-given")
    items = [{"task_id": task_id, "mode": "review"}, {"task_id": by_key("reorder-valid"), "mode": "review"}]
    desk = desk_for(tmp_path, items)
    desk.post("/start", {})
    desk.post("/pause", {})
    assert "멈춰 있습니다" in desk.page()
    desk.post("/approve", {})  # nothing moves while paused
    assert desk.phase == "work"
    desk.post("/resume", {})
    desk.post("/approve", {})
    assert desk.last["decision"] == "approve"
    again = desk_for(tmp_path, items)
    assert again.current["task_id"] == by_key("reorder-valid") and len(again.rows) == 1


def test_practice_tickets_do_not_count(tmp_path):
    desk = desk_for(tmp_path, [{"task_id": "test24dev-00", "mode": "hand", "warmup": True}])
    assert "연습" in desk.page()
    desk.post("/start", {})
    desk.post("/finish", {})
    assert desk.last["verdict"] == "failed" and desk.last["lora"] is None
    assert summary(desk.rows, LORA)["hand"]["tickets"] == 0
    assert "이 문의의 답" in desk.page()


def test_the_newest_quote_fills_the_replys_attachment_box_and_not_the_lookup(tmp_path):
    task_id = by_key("date-2-given")
    desk = desk_for(tmp_path, [{"task_id": task_id, "mode": "hand"}])
    desk.post("/start", {})
    qid = quote(desk, task_id)
    lookup, reply = desk.page().split("<h2>손님에게 답장")
    assert f"value='{qid}'" in reply and f"value='{qid}'" not in lookup
