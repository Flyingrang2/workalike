"""The Korean web entrance: what goes out and into the record is English; Korean stays private."""
import json

from island.score2 import score
from island.tasks2 import make_tasks
from workalike.envs import V2
from workalike.filter import clean_episode
from workalike.record import Recorder, read_events
from workalike.sft import sft_row
from workalike.web2 import Korean, KoreanDesk, translation_problems

KO_REPLY = "안녕하세요. 견적 Q-{n}-1을 보내 드립니다. 10월 2일까지 출고합니다."
EN_REPLY = "Hello. Here is quote Q-{n}-1. It ships by October 2."


def korean_for(task):
    says = task["hidden"]["state"]["customers"][task["view"]["ticket_id"]]["says"]
    return Korean({"tickets": {task["task_id"]: {"who": "손님", "body": "한국어 본문", "thread": [],
                                                 "sent_so_far": None,
                                                 "customer": {k: f"[한국어] {k}" for k in says}}},
                   "fixed": {"en": {"owner": [], "customer_defaults": ["Thanks, please go ahead with {qid}.",
                                                                       "We already confirmed {qid}, thanks.", "As I said: "]},
                             "ko": {"owner": [], "customer_defaults": ["감사합니다, {qid} 진행해 주세요.",
                                                                       "이미 {qid} 확정했습니다, 감사합니다.", "말씀드렸듯이: "]}}})


def desk_for(task, tmp_path, translate):
    rec = Recorder(tmp_path / "web.jsonl")
    session = {"episode": "e-1", "view": task["view"], "state": task["hidden"]["state"], "task_id": task["task_id"],
               "actor": "me"}
    return KoreanDesk([session], rec, korean_for(task), translate)


def standard():
    return next(t for t in make_tasks(11, "train") if t["family"] == "standard" and len(t["grader"]["wants"]) == 1)


def quote_form(task, desk):
    w = task["grader"]["wants"][0]
    cost = task["hidden"]["state"]["catalog"][w["sku"]]["unit_cost"]
    desk.post("/act", {"tool": "calc_price", "sku": w["sku"], "qty": str(w["qty"]), "unit_cost": str(cost)})
    price = desk.log[0]["result"]["unit_price"]
    desk.post("/act", {"tool": "create_quote", "sku1": w["sku"], "qty1": str(w["qty"]), "price1": str(price),
                       "ship1": task["grader"]["need_by"]})
    return desk.log[0]["result"]["quote_id"]


def test_a_korean_reply_goes_out_in_english_and_the_korean_stays_private(tmp_path):
    t = standard()
    n = t["view"]["ticket_id"].split("-")[1]
    seen = []

    def translate(ko):
        seen.append(ko)
        return EN_REPLY.format(n=n).replace("October 2", "October 2"), "fake-1"
    desk = desk_for(t, tmp_path, translate)
    qid = quote_form(t, desk)
    desk.post("/draft", {"kind": "reply", "text": KO_REPLY.format(n=n), "quote_id": qid})
    assert desk.draft["problems"] == [] and "이대로 보내기" in desk.page()
    desk.post("/send", {})
    sent = desk.log[0]
    assert sent["name"] == "send_reply" and sent["args"]["message"] == EN_REPLY.format(n=n)
    assert "감사합니다" in desk.page()  # the customer's acceptance, shown in Korean
    desk.post("/finish", {})
    events = read_events(tmp_path / "web.jsonl")
    tr = [e for e in events if e["type"] == "translation"]
    assert tr and tr[0]["vis"] == "private" and tr[0]["ko"] == KO_REPLY.format(n=n) and tr[0]["translator"] == "fake-1"
    ep = clean_episode(events)
    text = json.dumps(sft_row(ep, V2), ensure_ascii=False)
    assert "안녕하세요" not in text and EN_REPLY.format(n=n) in text
    assert score(t, ep)["business_ok"] and seen


def test_a_translation_that_changes_a_number_cannot_be_sent(tmp_path):
    t = standard()
    n = t["view"]["ticket_id"].split("-")[1]
    desk = desk_for(t, tmp_path, lambda ko: (EN_REPLY.format(n=n).replace("October 2", "October 3"), "fake"))
    qid = quote_form(t, desk)
    desk.post("/draft", {"kind": "reply", "text": KO_REPLY.format(n=n), "quote_id": qid})
    assert desk.draft["problems"] and "이대로 보내기" not in desk.page()
    desk.post("/send", {})
    assert desk.log[0]["name"] == "create_quote"  # nothing went out


def test_english_is_sent_as_written_and_questions_show_the_customers_answer_in_korean(tmp_path):
    t = next(t for t in make_tasks(11, "train") if t["family"] == "missing_date")
    calls = []
    desk = desk_for(t, tmp_path, lambda ko: calls.append(ko) or ("x", "fake"))
    desk.post("/draft", {"kind": "reply", "text": "When do you need them?"})
    desk.post("/send", {})
    assert not calls and desk.log[0]["args"]["message"] == "When do you need them?"
    assert "[한국어] answer" in desk.page()


def test_the_owner_question_goes_through_the_same_draft(tmp_path):
    t = standard()
    desk = desk_for(t, tmp_path, lambda ko: ("May I take 3% off?", "fake"))
    desk.post("/draft", {"kind": "owner", "text": "3% 깎아 드려도 될까요?"})
    desk.post("/send", {})
    assert desk.log[0]["name"] == "ask_owner" and desk.log[0]["args"]["question"] == "May I take 3% off?"


def test_translation_check_reads_months_and_codes():
    assert translation_problems("MUG-15W 300개를 11월 20일까지, 견적 Q-12-1", "300 MUG-15W by Nov 20, quote Q-12-1") == []
    assert translation_problems("MUG-15W 300개", "300 MUG-11W")
    assert translation_problems("$4.08에", "at $4.80")
