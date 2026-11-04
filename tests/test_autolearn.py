"""The learning round with fakes: records in, a version out, the adapter on only if it is no worse."""

import json

from testbed.tasks2 import make_tasks
from workalike import autolearn
from workalike.desk import Desk
from workalike.envs import V2
from workalike.record import Recorder


def record_tickets(path, n):
    """A person who answers every ticket with one reply; the reply is what gets learned."""
    tasks = make_tasks(11, "train")[:n]
    for i, t in enumerate(tasks):
        rec = Recorder(path)
        rec.start(f"e-{i}", t["view"], task_id=t["task_id"], actor="me", env=V2.version)
        desk = Desk(t["hidden"]["state"], rec, env=V2)
        desk.call_many([("send_reply", {"ticket_id": t["view"]["ticket_id"], "message": "When do you need them?"})])
        rec.final("Done.")
        rec.end("done")


TALLY = {
    "ready": 2,
    "right_hold": 1,
    "needless_hold": 1,
    "dangerous": 0,
    "failed": 0,
    "good": 3,
    "look": 0,
    "tickets": 4,
}


def test_records_become_an_active_adapter_and_a_worse_one_does_not_replace_it(tmp_path):
    records, versions = tmp_path / "r.jsonl", tmp_path / "me"
    record_tickets(records, 6)
    trained = []
    shell = lambda step, paths: trained.append(paths)  # noqa: E731
    autolearn.learn_once(records, versions, shell, lambda name, path: TALLY, target=12)
    active = autolearn.active_adapter(versions)
    assert active["version"] == "v1" and active["tickets"] == 6 and active["rows"] >= 6
    assert trained and "v1" in trained[0]["adapter"]
    row = json.loads(next((versions / "v1" / "sft").glob("*-train.jsonl")).read_text().splitlines()[0])
    assert row["messages"][0]["content"] == V2.system_prompt  # trained on the prompt it will work under
    worse = {**TALLY, "good": 2, "right_hold": 0}
    m = autolearn.learn_once(records, versions, shell, lambda name, path: worse, target=12)
    assert m["activated"] is False and autolearn.active_adapter(versions)["version"] == "v1"
    status = json.loads((versions / "status.json").read_text())
    assert "켜지 않았습니다" in status["state"] and status["learned_from"] == 6


def test_a_first_adapter_with_a_dangerous_draft_does_not_go_on(tmp_path):
    records, versions = tmp_path / "r.jsonl", tmp_path / "me"
    record_tickets(records, 3)
    bad = {**TALLY, "dangerous": 1}
    autolearn.learn_once(records, versions, lambda s, p: None, lambda n, p: bad, target=6)
    assert autolearn.active_adapter(versions) is None
