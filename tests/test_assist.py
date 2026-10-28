"""The assist window: a request becomes drafts, the checks read them, and nothing is sent before approval."""
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, timedelta

import pytest

from island.shop import shop_from
from island.tasks import make_tasks
from workalike.assist import Assist, check
from workalike.record import group_episodes, read_events
from workalike.web import serve_in_background

QTY = 100


@pytest.fixture
def shop():
    shop = shop_from(make_tasks(7, "dev", 12)[0])
    sku = next(s for s, item in shop["catalog"].items() if item["on_hand"] >= QTY)
    ship = (date.fromisoformat(shop["today"]) + timedelta(days=14)).isoformat()
    return shop, sku, ship


def call(name, **args):
    return {"name": name, "arguments": args}


def quoting_model(sku, ship, reply):
    """A stand-in for the served model: looks up, prices and quotes one line, then replies with reply(quote)."""
    def model(messages, tools):
        results = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
        ticket = re.search(r"Ticket (T-\d+)", messages[1]["content"]).group(1)
        step = sum(1 for m in messages if m["role"] == "assistant")
        if step == 0:
            return {"tool_calls": [call("get_cost", sku=sku), call("check_stock", sku=sku, qty=QTY)]}
        if step == 1:
            return {"tool_calls": [call("calc_price", sku=sku, qty=QTY, unit_cost=results[0]["unit_cost"])]}
        if step == 2:
            line = {"sku": sku, "qty": QTY, "unit_price": results[-1]["unit_price"]}
            return {"tool_calls": [call("create_quote", ticket_id=ticket, lines=[line], ship_by=ship)]}
        if step == 3:
            q = results[-1]
            return {"tool_calls": [call("send_reply", ticket_id=ticket, message=reply(q), quote_id=q["quote_id"])]}
        return {"content": "Done."}
    return model


def honest(q):
    return f"Hi Ana, here is quote {q['quote_id']}: total ${q['total']:,.2f}, ready by {q['ship_by']}."


def test_request_becomes_a_draft_and_is_sent_only_after_approval(tmp_path, shop):
    state, sku, ship = shop
    app = Assist(state, {"fake": quoting_model(sku, ship, honest)}, tmp_path)
    job = app.ask("Ana Ruiz, Ruiz Cafe", f"Please quote {QTY} of {sku}.")
    app.todo.join()

    assert job["status"] == "waiting"
    assert all(c["ok"] for c in job["checks"]), job["checks"]
    assert [c["name"] for c in job["calls"]] == ["get_cost", "check_stock", "calc_price", "create_quote", "send_reply"]
    assert not (tmp_path / "sent.jsonl").exists()
    assert "Approve and send" in app.page()

    app.decide(job["id"], approve=True)
    sent = [json.loads(line) for line in (tmp_path / "sent.jsonl").read_text().splitlines()]
    assert len(sent) == 1 and sent[0]["quote"]["lines"][0]["sku"] == sku
    app.decide(job["id"], approve=True)  # a decided job stays decided
    assert len((tmp_path / "sent.jsonl").read_text().splitlines()) == 1
    assert state["outbox"] == [] and state["drafts"] == {}  # the shop itself is untouched

    (events,) = group_episodes(read_events(tmp_path / "records.jsonl")).values()
    assert events[0]["actor"] == "model:fake" and events[0]["entrance"] == "assist"
    assert events[-1]["status"] == "done"


def test_checks_flag_amounts_and_quote_ids_the_desk_never_gave(tmp_path, shop):
    state, sku, ship = shop
    def invented(q):
        return f"Quote {q['quote_id']} comes to $1.00, and Q-9999-1 still stands."
    app = Assist(state, {"fake": quoting_model(sku, ship, invented)}, tmp_path)
    job = app.ask("Ana", "Quote please.")
    app.todo.join()
    failed = {c["name"] for c in job["checks"] if not c["ok"]}
    assert failed == {"amounts come from the desk", "quote ids exist"}
    assert "Approve anyway" in app.page()


def test_a_price_without_a_quote_or_a_cost_in_the_reply_fails(shop):
    state, sku, ship = shop
    cost = state["catalog"][sku]["unit_cost"]
    history = [{"id": "c1", "name": "get_cost", "args": {"sku": sku}, "result": {"sku": sku, "unit_cost": cost}}]
    state = {**state, "tickets": ["T-1001"], "drafts": {}, "outbox": [
        {"ticket_id": "T-1001", "message": "Thanks, we'll check and come back to you.", "quote_id": None},
        {"ticket_id": "T-1001", "message": f"Option 1 is 40 dollars; our cost is ${cost:.2f}.", "quote_id": None}]}
    failed = {c["name"]: c["detail"] for c in check(state, "T-1001", history, "done") if not c["ok"]}
    assert failed.keys() == {"prices go out on a quote", "costs stay inside", "amounts come from the desk"}
    assert failed["prices go out on a quote"].startswith("1 reply")
    assert f"${cost:.2f}" in failed["costs stay inside"]


def test_quote_check_reads_the_current_cost_and_stock(shop):
    state, sku, ship = shop
    item = state["catalog"][sku]
    state = {**state, "tickets": ["T-1001"], "drafts": {
        "Q-1001-1": {"ticket_id": "T-1001", "ship_by": ship,
                     "lines": [{"sku": sku, "qty": item["on_hand"] + 1, "unit_price": item["unit_cost"]}]}},
             "outbox": [{"ticket_id": "T-1001", "message": "See the quote.", "quote_id": "Q-1001-1"}]}
    (quote,) = [c for c in check(state, "T-1001", [], "done") if c["name"].startswith("quote Q-")]
    assert not quote["ok"] and "margin floor" in quote["detail"]
    if item["restock_date"] is None or ship < item["restock_date"]:
        assert "short" in quote["detail"]


def test_no_reply_or_a_model_error_cannot_be_approved(tmp_path, shop):
    state, _, _ = shop
    def broken(messages, tools):
        raise ConnectionError("server down")
    for model in (lambda m, t: {"content": "Done."}, broken):
        app = Assist(state, {"fake": model}, tmp_path)
        job = app.ask("Ana", "Quote please.")
        app.todo.join()
        app.decide(job["id"], approve=True)
        assert job["status"] == "failed"
    assert not (tmp_path / "sent.jsonl").exists()
    assert job["run_status"].startswith("error: ConnectionError")


def post(url, form):
    data = urllib.parse.urlencode(form).encode()
    return urllib.request.urlopen(urllib.request.Request(url, data=data, method="POST")).read().decode()


def test_web_window_needs_the_page_token(tmp_path, shop):
    state, sku, ship = shop
    app = Assist(state, {"fake": quoting_model(sku, ship, honest)}, tmp_path)
    server = serve_in_background(app, port=0)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with pytest.raises(urllib.error.HTTPError) as err:
            post(url + "/ask", {"from": "Ana", "body": "Quote please."})
        assert err.value.code == 403 and app.jobs == []
        post(url + "/ask", {"from": "Ana", "body": f"Please quote {QTY} of {sku}.", "csrf": app.token})
        app.todo.join()
        assert "Draft reply" in post(url + "/approve", {"job": "1", "csrf": app.token})
        status = json.loads(urllib.request.urlopen(url + "/status").read())
        assert status["jobs"] == [{"id": 1, "model": "fake", "ticket_id": "T-1001", "status": "approved", "checks_ok": True}]
    finally:
        server.shutdown()


def test_every_model_drafts_the_same_request_for_comparison(tmp_path, shop):
    state, sku, ship = shop
    def other(q):
        return f"Thanks Ana, quote {q['quote_id']} is attached."
    app = Assist(state, {"first": quoting_model(sku, ship, honest), "second": quoting_model(sku, ship, other)}, tmp_path)
    assert "every model (compare)" in app.page()
    app.post("/ask", {"from": "Ana", "body": f"Please quote {QTY} of {sku}.", "model": "*"})
    app.post("/ask", {"from": "Ana", "body": "Again, please.", "model": "second"})
    app.todo.join()
    assert [(j["model"], j["view"]["body"][:6]) for j in app.jobs] == [("first", "Please"), ("second", "Please"),
                                                                      ("second", "Again,")]
    assert app.jobs[0]["replies"][0]["message"] != app.jobs[1]["replies"][0]["message"]
    assert "drafted by second" in app.page()
