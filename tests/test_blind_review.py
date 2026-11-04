"""Blind review: drafts are shown without run names, answers are kept, a restart skips answered tickets."""

import json
import random

from testbed import worker_c
from testbed.blind_review import Review, draft_steps, episodes_by_task
from testbed.tasks2 import make_tasks
from workalike.desk import Desk
from workalike.envs import V2
from workalike.person_desk import Korean
from workalike.record import Recorder


def c_records(tasks, path, seed):
    rec = Recorder(path)
    for t in tasks:
        rec.start("e-" + t["task_id"], t["view"], task_id=t["task_id"], env=V2.version)
        rec.final(worker_c.work(Desk(t["hidden"]["state"], rec, env=V2), t, random.Random(f"{seed}:{t['task_id']}")))
        rec.end()
    return episodes_by_task(path)


def korean():
    return Korean(
        {
            "tickets": {},
            "fixed": {
                "en": {"owner": [], "customer_defaults": ["a {qid}", "b {qid}", "As I said: "]},
                "ko": {"owner": [], "customer_defaults": ["가 {qid}", "나 {qid}", "말씀드렸듯이: "]},
            },
        }
    )


def test_drafts_hide_the_run_and_keep_answers(tmp_path):
    tasks = make_tasks(11, "dev")[:3]
    runs = {
        "secret_base": c_records(tasks, tmp_path / "a.jsonl", 1),
        "secret_lora": c_records(tasks, tmp_path / "b.jsonl", 2),
    }
    out = tmp_path / "review"
    out.mkdir()
    review = Review(tasks, runs, korean(), lambda en: "[ko] " + en, out, seed=1)
    page = review.page()
    assert "secret_base" not in page and "secret_lora" not in page and "초안 A" in page and "[ko] " in page
    steps = draft_steps(runs["secret_base"][tasks[0]["task_id"]])
    assert steps and all(s["to"] in ("customer", "owner") for s in steps)
    first = tasks[0]["task_id"]
    review.post("/answer", {"task_id": first, "closest": "B", "vA": "approve", "vB": "edit", "nB": "날짜만 고침"})
    row = json.loads((out / "answers.jsonl").read_text())
    assert row["closest"] == "B" and row["drafts"]["B"] == {"verdict": "edit", "note": "날짜만 고침"}
    key = json.loads((out / "key.json").read_text())
    assert set(key[first].values()) == {"secret_base", "secret_lora"}
    again = Review(tasks, runs, korean(), lambda en: en, out, seed=1)
    assert again.status()["task_id"] == tasks[1]["task_id"] and again.key[first] == key[first]
