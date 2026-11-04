"""Learning from your own records: what review shows is what build trains on, minus what you leave out."""

import json
import urllib.parse
import urllib.request

import pytest

from testbed.gen_h1 import build as build_h1
from workalike.learn import Review, build, episodes, list_versions
from workalike.record import Recorder
from workalike.sft import read_jsonl, sft_row
from workalike.web import serve_in_background


@pytest.fixture(scope="module")
def records(tmp_path_factory):
    out = tmp_path_factory.mktemp("h1")
    build_h1(out, 7)
    path = out / "records" / "a-dev.jsonl"
    rec = Recorder(path)  # a ticket the person opened and never finished
    rec.start(
        "e-unfinished", {"today": "2026-11-02", "ticket_id": "T-1", "from": "X", "body": "Quote?"}, actor="w-82d9d6"
    )
    return path


def test_build_learns_finished_episodes_minus_the_ones_left_out(records, tmp_path):
    versions = tmp_path / "me"
    eps = episodes(records)
    done = [e for e in eps if e["status"] == "done"]
    assert len(done) == len(eps) - 1
    (versions).mkdir()
    (versions / "exclude.json").write_text(json.dumps({"excluded": [done[0]["ep"], done[1]["ep"]]}))
    m = build(records, versions)
    assert (m["version"], m["rows"]) == ("v1", len(done) - 2)
    assert m["unfinished"] == ["e-unfinished"] and len(m["left_out"]) == 2
    rows = read_jsonl(versions / "v1" / "sft" / "me-train.jsonl")
    assert rows == [sft_row(e["clean"]) for e in done[2:]]
    assert build(records, versions)["version"] == "v2"

    (versions / "v1" / "adapter").mkdir()
    (versions / "v1" / "adapter" / "adapter_config.json").write_text("{}")
    assert [(v["name"], bool(v["adapter"])) for v in list_versions(versions)] == [("me-v1", True), ("me-v2", False)]


def test_review_page_saves_what_to_leave_out(records, tmp_path):
    app = Review(records, tmp_path / "me")
    server = serve_in_background(app, port=0)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        page = urllib.request.urlopen(url).read().decode()
        assert "Unfinished, always left out" in page and "Leave out" in page
        ep = next(e["ep"] for e in episodes(records) if e["status"] == "done")
        data = urllib.parse.urlencode({"csrf": app.token, f"out:{ep}": "on"}).encode()
        urllib.request.urlopen(urllib.request.Request(url + "/save", data=data, method="POST")).read()
        assert app.status()["excluded"] == [ep]
        assert "checked" in urllib.request.urlopen(url).read().decode()
    finally:
        server.shutdown()
