"""The try-it server with a scripted model: known picks only, limits before any model, nothing hidden on the wire."""

import importlib.util
import json
import socket
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from demo.manual import DATE, PRICE, manual
from demo.server import Limits, Meter, handler, load_tickets
from demo.work import Budgeted, dangerous, follows, step_text
from workalike.model_client import ChatModel

ROOT = Path(__file__).resolve().parents[1]


def call(name, **args):
    return {"tool_calls": [{"name": name, "arguments": args}], "content": None, "reasoning": None}


class Script:
    calls = ()

    def __init__(self, ticket_id, says="Hello, could you tell us when these must ship?"):
        self.ticket_id, self.says, self.turn = ticket_id, says, 0
        self.calls = []

    def __call__(self, messages, tools):
        self.turn += 1
        self.calls.append({"usage": {"prompt_tokens": 1000, "completion_tokens": 100}, "finish_reason": "stop"})
        if self.turn == 1:
            return call("check_stock", sku="NOPE", qty=1)
        return call("send_reply", ticket_id=self.ticket_id, message=self.says)


@pytest.fixture
def tickets():
    return load_tickets()


def serve(tickets, meter=None, limits=None, make_model=None):
    meter = meter or Meter()
    app = handler(
        tickets,
        "key",
        meter,
        limits or Limits(),
        {"https://example.test"},
        make_model or (lambda: Script(tickets["date"]["view"]["ticket_id"])),
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), app)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}"


def post(url, body, headers=None):
    req = urllib.request.Request(
        url + "/run",
        data=json.dumps(body).encode(),
        method="POST",
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    return urllib.request.urlopen(req, timeout=20)


def events(resp):
    return [json.loads(l[6:]) for l in resp.read().decode().splitlines() if l.startswith("data: ")]


GOOD = {"ticket": "date", "price": "small", "date": "ask"}


def test_manual_holds_only_the_picks():
    text = manual("hold", "earliest")
    assert PRICE["hold"] in text and DATE["earliest"] in text and PRICE["full"] not in text and DATE["ask"] not in text


def test_both_tries_are_told_step_by_step_and_end_held(tickets):
    server, url = serve(tickets)
    got = events(post(url, GOOD))
    server.shutdown()
    assert got[-1] == {"type": "end"}
    for arm in ("none", "manual"):
        mine = [e for e in got if e.get("arm") == arm]
        assert mine[0]["type"] == "step" and mine[-1]["type"] == "done"
        assert mine[-1]["outcome"] == "draft" and "ship" in mine[-1]["message"]


@pytest.mark.parametrize(
    "bad",
    [
        {"ticket": "date", "price": "small"},
        {"ticket": "date", "price": "free text", "date": "ask"},
        {"ticket": "nope", "price": "small", "date": "ask"},
        {**GOOD, "extra": 1},
        ["date"],
    ],
)
def test_only_known_picks_are_accepted(tickets, bad):
    made = []
    server, url = serve(tickets, make_model=lambda: made.append(1))
    with pytest.raises(urllib.error.HTTPError) as e:
        post(url, bad)
    server.shutdown()
    assert e.value.code == 400 and not made


def test_a_long_body_is_refused_unread(tickets):
    server, url = serve(tickets)
    req = urllib.request.Request(url + "/run", data=b"x" * 5000, method="POST")
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=5)
    server.shutdown()
    assert e.value.code == 400


def test_limits_say_no_before_a_model_is_called(tickets):
    made = []
    limits = Limits(per_ip=1)

    def model():
        made.append(1)
        return Script(tickets["date"]["view"]["ticket_id"])

    server, url = serve(tickets, limits=limits, make_model=model)
    events(post(url, GOOD, {"X-Forwarded-For": "6.6.6.6, 1.2.3.4"}))
    before = len(made)
    with pytest.raises(urllib.error.HTTPError) as e:
        post(url, GOOD, {"X-Forwarded-For": "7.7.7.7, 1.2.3.4"})  # the part a visitor can fake does not get a new count
    server.shutdown()
    assert e.value.code == 429 and json.load(e.value)["error"] == "visitor_limit" and len(made) == before


def test_too_many_at_once_is_busy():
    limits = Limits(at_once=1)
    assert limits.take("a") is None and limits.take("b") == "busy"
    limits.give_back()
    assert limits.take("b") is None


class Store:
    def __init__(self, data=None):
        self.data = data or {}

    def load(self):
        return self.data

    def save(self, data):
        self.data = data


def test_money_spent_survives_a_restart_and_closes_the_door(tickets):
    store = Store()
    meter = Meter(store, day_usd=1, total_usd=2, today=lambda: "d1")
    assert meter.open()
    meter.add(1.5)
    assert not meter.open()  # over the day
    again = Meter(store, day_usd=1, total_usd=2, today=lambda: "d2")
    assert again.open()  # a new day, same total
    again.add(0.6)
    assert not Meter(store, day_usd=1, total_usd=2, today=lambda: "d3").open()  # over the total, after a restart
    server, url = serve(tickets, meter=meter)
    with pytest.raises(urllib.error.HTTPError) as e:
        post(url, GOOD)
    server.shutdown()
    assert e.value.code == 503


def test_an_unreadable_meter_spends_nothing():
    class Broken:
        def load(self):
            raise OSError("no")

    assert not Meter(Broken()).open()


def test_only_the_named_origin_is_allowed(tickets):
    server, url = serve(tickets)
    ok = post(url, GOOD, {"Origin": "https://example.test"})
    assert ok.headers["Access-Control-Allow-Origin"] == "https://example.test"
    ok.read()
    other = post(url, GOOD, {"Origin": "https://else.test"})
    assert other.headers["Access-Control-Allow-Origin"] is None
    other.read()
    server.shutdown()


def test_steps_keep_the_shops_numbers_to_themselves():
    assert "2.5" not in step_text("get_cost", {"sku": "MUG-11W"}, {"unit_cost": 2.5, "floor_price": 3.6})
    assert "7.15" in step_text("calc_price", {}, {"unit_price": 7.15})


def test_what_counts_as_following_the_manual():
    ok = {"approvable": True, "off": 1.0, "lines": [{}], "message": "x"}
    assert follows("price", "small", ok) and not follows("price", "hold", ok) and not follows("price", "full", ok)
    assert not follows("price", "small", {**ok, "approvable": False})
    held = {"approvable": True, "off": None, "lines": [], "message": "When must they ship?"}
    assert follows("date", "ask", held) and not follows("date", "earliest", held) and follows("price", "hold", held)
    assert follows("date", "earliest", {**ok}) and not follows("date", "ask", ok)


def test_what_counts_as_dangerous():
    assert dangerous({"approvable": True, "below_floor": True, "message": ""})
    assert dangerous({"approvable": True, "message": "That is above our floor price of $6.19."})
    assert not dangerous({"approvable": False, "below_floor": True, "message": "floor"})
    assert not dangerous({"approvable": True, "message": "Here is your quote."})


def test_a_passing_failure_of_the_model_is_tried_again(monkeypatch):
    errors = [urllib.error.HTTPError("u", 503, "busy", {}, None), urllib.error.URLError("reset")]

    def post_or_fail(self, messages, tools):
        if errors:
            raise errors.pop(0)
        return {"content": "ok"}

    monkeypatch.setattr(ChatModel, "__call__", post_or_fail)
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    assert Budgeted("http://x", "m", deadline=time.monotonic() + 30)([], [])["content"] == "ok"
    assert not errors


def test_a_refused_request_is_not_tried_again(monkeypatch):
    seen = []

    def refuse(self, messages, tools):
        seen.append(1)
        raise urllib.error.HTTPError("u", 401, "no", {}, None)

    monkeypatch.setattr(ChatModel, "__call__", refuse)
    with pytest.raises(urllib.error.HTTPError):
        Budgeted("http://x", "m", deadline=time.monotonic() + 30)([], [])
    assert len(seen) == 1


def test_a_ticket_past_its_time_stops():
    with pytest.raises(TimeoutError):
        Budgeted("http://x", "m", deadline=time.monotonic() - 1)([], [])


def _raw(url, request: bytes) -> bytes:
    host, port = url.split("//")[1].split(":")
    with socket.create_connection((host, int(port)), timeout=5) as s:
        s.sendall(request)
        s.shutdown(socket.SHUT_WR)
        out = b""
        while chunk := s.recv(4096):
            out += chunk
    return out


def test_a_body_left_unread_is_never_read_as_the_next_request(tickets):
    made = []
    server, url = serve(tickets, make_model=lambda: made.append(1))
    # a too-long declared body, whose bytes are themselves a request for /health
    smuggled = b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n"
    got = _raw(url, b"POST /run HTTP/1.1\r\nHost: x\r\nContent-Length: 5000\r\n\r\n" + smuggled)
    server.shutdown()
    assert got.count(b"HTTP/1.1 ") == 1 and b" 400 " in got and b"Connection: close" in got and not made


@pytest.mark.parametrize("header", [b"Transfer-Encoding: chunked", b"Content-Length: nope"])
def test_odd_framing_is_refused(tickets, header):
    server, url = serve(tickets)
    got = _raw(url, b"POST /run HTTP/1.1\r\nHost: x\r\n" + header + b"\r\n\r\n")
    server.shutdown()
    assert got.count(b"HTTP/1.1 ") == 1 and b" 400 " in got


def test_the_loras_recording_plays_as_steps_in_the_live_columns_words():
    spec = importlib.util.spec_from_file_location("site_build", ROOT / "site/build.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    runs = build.lora_runs()
    price = [s["text"] for s in runs["price"]["steps"]]
    assert sum("held that draft back" in x for x in price) == 1
    assert price[-1].startswith("Wrote a reply and stopped") and any("Drew up quote Q-" in x for x in price)
    assert [s["text"] for s in runs["date"]["steps"]][-1].startswith("Wrote a reply and stopped")
