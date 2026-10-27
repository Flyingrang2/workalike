"""The web entrance leaves the same records as the tool entrance, and the HTTP client speaks OpenAI format."""
import json
import random
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

from island import scripted
from island.score import score_records
from island.serve_web import sessions_for
from island.tasks import make_tasks
from workalike import agent
from workalike.desk import Desk
from workalike.filter import clean_episode
from workalike.model_client import ChatModel
from workalike.record import Recorder, group_episodes, read_events
from workalike.web import WebDesk, describe, serve_in_background


def call_to_form(name, args):
    form = {"tool": name}
    if name == "create_quote":
        for i, line in enumerate(args["lines"], 1):
            form.update({f"sku{i}": line["sku"], f"qty{i}": str(line["qty"]), f"price{i}": str(line["unit_price"])})
        form["ship_by"] = args["ship_by"]
    else:
        form.update({k: str(v) for k, v in args.items() if k != "ticket_id"})
    return form


def post(url, form, headers=None):
    data = urllib.parse.urlencode(form).encode()
    with urllib.request.urlopen(urllib.request.Request(url, data=data, method="POST", headers=headers or {})) as r:
        return r.read().decode()


def test_web_entrance_matches_tool_entrance(tmp_path):
    tasks = [t for t in make_tasks(7, "eval_h1", 30) if t["family"] in ("stock_short", "unknown_sku")][:3]
    tool_eps = []
    for t in tasks:
        rec = Recorder(tmp_path / "tool.jsonl")
        rec.start("e-" + t["task_id"], t["view"], task_id=t["task_id"])
        rec.final(scripted.work(Desk(t["hidden"]["state"], rec), t, "a", random.Random(t["task_id"])))
        rec.end()
    for events in group_episodes(read_events(tmp_path / "tool.jsonl")).values():
        tool_eps.append(clean_episode(events))

    web = WebDesk(sessions_for(tasks, "person"), Recorder(tmp_path / "web.jsonl"))
    server = serve_in_background(web, port=0)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        for ep in tool_eps:
            for turn in ep["turns"]:
                for c in turn["calls"]:
                    page = post(url + "/act", {**call_to_form(c["name"], c["args"]), "csrf": web.token})
                    assert describe(c["name"], c["args"], c["result"]).split(":")[0] in page.replace("&#x27;", "'")
            post(url + "/finish", {"csrf": web.token})
        assert "No more tickets" in urllib.request.urlopen(url).read().decode()
    finally:
        server.shutdown()

    web_eps = [clean_episode(evs) for evs in group_episodes(read_events(tmp_path / "web.jsonl")).values()]
    strip = lambda ep: [(c["name"], c["args"], c["result"]) for t in ep["turns"] for c in t["calls"]]
    assert [strip(e) for e in web_eps] == [strip(e) for e in tool_eps]
    rows, summary = score_records(tmp_path / "web.jsonl", {t["task_id"]: t for t in tasks})
    assert summary["correct"] == 3 and summary["cost_before_first_reply"] == 3
    starts = [e for e in read_events(tmp_path / "web.jsonl") if e["type"] == "episode_start"]
    assert {e["entrance"] for e in starts} == {"web"}


def test_other_sites_cannot_post_into_the_record(tmp_path):
    task = make_tasks(7, "eval_h1", 30)[0]
    web = WebDesk(sessions_for([task], "person"), Recorder(tmp_path / "web.jsonl"))
    server = serve_in_background(web, port=0)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    form = {"tool": "send_reply", "message": "injected"}
    try:
        for extra, headers in [({}, {}), ({"csrf": web.token}, {"Origin": "https://evil.example"}),
                               ({"csrf": web.token}, {"Host": "evil.example"})]:
            try:
                post(url + "/act", {**form, **extra}, headers)
                raise AssertionError("request should be rejected")
            except urllib.error.HTTPError as e:
                assert e.code == 403
    finally:
        server.shutdown()
    assert web.log == []
    assert not any(e["type"] == "tool_call" for e in read_events(tmp_path / "web.jsonl"))


def test_bad_form_input_is_recorded_as_an_error(tmp_path):
    task = make_tasks(7, "eval_h1", 30)[0]
    web = WebDesk(sessions_for([task], "person"), Recorder(tmp_path / "web.jsonl"))
    web.act({"tool": "check_stock", "sku": "MUG-11W", "qty": "twelve"})
    assert web.log[0]["result"]["error"] == "invalid_arguments"
    assert "failed" in web.page()


class _FakeServer(BaseHTTPRequestHandler):
    bodies = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _FakeServer.bodies.append(body)
        turn = sum(1 for m in body["messages"] if m["role"] == "assistant")
        if turn == 0:
            msg = {"role": "assistant", "content": None, "reasoning": "check cost first",
                   "tool_calls": [{"id": "x1", "type": "function",
                                   "function": {"name": "get_cost", "arguments": '{"sku": "MUG-11W"}'}}]}
        else:
            msg = {"role": "assistant", "content": "Done."}
        data = json.dumps({"choices": [{"message": msg, "finish_reason": "stop"}],
                           "usage": {"prompt_tokens": 10, "completion_tokens": 2}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def test_chat_client_round_trip(tmp_path):
    server = HTTPServer(("127.0.0.1", 0), _FakeServer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        task = next(t for t in make_tasks(7, "train", 40) if "MUG-11W" in t["hidden"]["state"]["catalog"])
        rec = Recorder(tmp_path / "r.jsonl")
        rec.start("e-1", task["view"], task_id=task["task_id"])
        model = ChatModel(f"http://127.0.0.1:{server.server_address[1]}/v1", "a")
        messages, status = agent.run(Desk(task["hidden"]["state"], rec), task["view"], model)
        rec.end(status)
    finally:
        server.shutdown()
    assert status == "done" and messages[-1] == {"role": "assistant", "content": "Done."}
    sent = _FakeServer.bodies[-1]["messages"]
    assert isinstance(sent[2]["tool_calls"][0]["function"]["arguments"], str)
    assert sent[3]["role"] == "tool" and '"unit_cost"' in sent[3]["content"]
    events = read_events(tmp_path / "r.jsonl")
    note = next(e for e in events if e["type"] == "note")
    assert note["vis"] == "private" and note["source"] == "reasoning"
    assert "check cost first" not in json.dumps(clean_episode(events))
    assert model.calls[0]["usage"]["prompt_tokens"] == 10
