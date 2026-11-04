"""End-to-end checks on the Hypothesis 1 dataset (plan tests 1 to 8)."""

import json
import random
import re
from pathlib import Path

import pytest

from testbed import scripted
from testbed.gen_h1 import build, load_episodes
from testbed.report_dev import report
from testbed.score import cost_before_first_reply, score, score_records
from testbed.tasks import CATALOG, EVAL_ONLY_SKUS, POOLS, make_tasks
from workalike import agent
from workalike.desk import Desk
from workalike.filter import clean_episode
from workalike.record import Recorder, read_events
from workalike.serialize import SYSTEM_PROMPT, to_messages
from workalike.sft import read_jsonl

SEED = 7


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    out = tmp_path_factory.mktemp("h1")
    manifest = build(out, SEED)
    tasks = {}
    for split in ("train", "eval_h1", "eval_trap"):
        for t in read_jsonl(out / "tasks" / f"{split}.jsonl"):
            tasks[t["task_id"]] = t
    return out, manifest, tasks


def episodes(out, style, split):
    return load_episodes(Path(out) / "records" / f"{style}-{split}.jsonl")


def test_1_same_seed_same_bytes(data, tmp_path):
    out, manifest, _ = data
    again = build(tmp_path / "again", SEED)
    assert again["files"] == manifest["files"]


def _secrets(task):
    state = task["hidden"]["state"]
    out = {}
    for sku, item in state["catalog"].items():
        out[f"cost {sku}"] = item["unit_cost"]
        if item["restock_date"]:
            out[f"restock {sku}"] = item["restock_date"]
    for qid, q in state["quotes"].items():
        for line in q["lines"]:
            out[f"approved {qid} {line['sku']}"] = line["unit_price"]
    return out


def _pattern(value):
    text = json.dumps(value) if not isinstance(value, str) else value
    return re.compile(r"(?<![\d.])" + re.escape(text.strip('"')) + r"(?![\d])")


def test_2_no_hidden_value_before_a_tool_shows_it(data):
    out, manifest, tasks = data
    for style in "ab":
        for meta, ep in episodes(out, style, "train"):
            task = tasks[meta["task_id"]]
            messages = to_messages(ep)
            for label, value in _secrets(task).items():
                pat = _pattern(value)
                for i, m in enumerate(messages):
                    text = json.dumps(m, ensure_ascii=False) if m["role"] != "tool" else m["content"]
                    if m["role"] == "tool" and pat.search(text):
                        break
                    assert not pat.search(text), f"{label} leaked in message {i} of {meta['task_id']}"


def test_2_no_labels_in_training_rows(data):
    out, manifest, _ = data
    # "unknown_sku" is also the desk's own error code, so the family names are checked through the key instead.
    forbidden = [
        *manifest["actors"].values(),
        "stock_short",
        "approved_valid",
        "ack_already_sent",
        "approved_stale",
        '"family"',
        "grader",
        "hidden",
        "task_id",
        "eval_h1",
        "worker",
    ]
    for name in ("a", "b", "mixed"):
        text = (out / "sft" / f"{name}-train.jsonl").read_text()
        for word in forbidden:
            assert word not in text, f"{word!r} found in sft/{name}-train.jsonl"


def test_3_filter_drops_private_and_keeps_recovery(tmp_path):
    task = next(t for t in make_tasks(SEED, "train", 40) if t["family"] == "unknown_sku")
    rec = Recorder(tmp_path / "r.jsonl")
    rec.start("e-1", {**task["view"], "incident": "cost spiked last month"}, task_id=task["task_id"], actor="w-x")
    rec.note("remember: always check cost first")
    desk = Desk(task["hidden"]["state"], rec)
    rec.final(scripted.work(desk, task, "a", random.Random(1)))
    rec.end()
    ep = clean_episode(read_events(tmp_path / "r.jsonl"))
    text = json.dumps(to_messages(ep))
    assert "remember" not in text and "incident" not in text and "spiked" not in text and "w-x" not in text
    results = [c["result"] for t in ep["turns"] for c in t["calls"]]
    assert any(r.get("error") == "unknown_sku" for r in results)
    assert ep["final"] == "Done."


def test_4_both_ways_are_correct_and_traps_hold(data):
    out, _, tasks = data
    for style in "ab":
        for split in ("train", "eval_h1", "eval_trap"):
            for meta, ep in episodes(out, style, split):
                task = tasks[meta["task_id"]]
                s = score(task, ep)
                assert s["correct"], (style, meta["task_id"], s["reasons"])
                fam = task["family"]
                if fam == "approved_valid" and style == "a":
                    assert not s["called_get_cost"]
                if fam == "ack_already_sent" and style == "b":
                    assert not s["acked"]
                if fam == "approved_stale":
                    assert s["called_get_cost"]
                if fam == "stock_short":
                    assert s["short_choice"] == ("later_date" if style == "a" else "substitute")


def test_5_metric_ceiling_on_eval_h1(data):
    out, _, _ = data
    counts = {
        style: sum(bool(cost_before_first_reply(ep)) for _, ep in episodes(out, style, "eval_h1")) for style in "ab"
    }
    assert counts == {"a": 30, "b": 0}


def test_6_agent_loop_builds_the_training_messages(data, tmp_path):
    out, _, tasks = data
    for style in "ab":
        for meta, ep in episodes(out, style, "train")[:25]:
            task = tasks[meta["task_id"]]
            turns = iter(ep["turns"])

            def replay(messages, tools):
                turn = next(turns, None)
                if turn is None:
                    return {"content": ep["final"]}
                return {"tool_calls": [{"name": c["name"], "arguments": c["args"]} for c in turn["calls"]]}

            path = tmp_path / f"{style}-{meta['task_id']}.jsonl"
            rec = Recorder(path)
            rec.start("e-replay", task["view"])
            messages, status = agent.run(Desk(task["hidden"]["state"], rec), task["view"], replay)
            rec.end(status)
            assert status == "done"
            assert messages == to_messages(ep)
            assert to_messages(clean_episode(read_events(path))) == messages


def test_7_eval_surface_is_held_out(data):
    out, _, tasks = data
    train_text = (out / "tasks" / "train.jsonl").read_text() + (out / "sft" / "mixed-train.jsonl").read_text()
    for sku in EVAL_ONLY_SKUS:
        assert sku not in train_text
    for name in POOLS["eval"]["first"] + POOLS["eval"]["last"]:
        assert re.search(rf"\b{name}\b", train_text) is None, name
    train_ids = {t for t in tasks if t.startswith("train-")}
    assert train_ids and not train_ids & {t for t in tasks if t.startswith("eval")}
    train_tickets = {t["view"]["ticket_id"] for t in tasks.values() if t["split"] == "train"}
    eval_tickets = {t["view"]["ticket_id"] for t in tasks.values() if t["split"] != "train"}
    assert not train_tickets & eval_tickets
    eval_skus = {l["sku"] for t in tasks.values() if t["split"] != "train" for l in t["grader"]["lines"]}
    assert eval_skus & EVAL_ONLY_SKUS, "evaluation should exercise some held-out SKUs"


def test_8_format(data):
    out, _, _ = data
    for row in read_jsonl(out / "sft" / "mixed-train.jsonl"):
        msgs = row["messages"]
        assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT}
        assert msgs[1]["role"] == "user" and msgs[-1] == {"role": "assistant", "content": "Done."}
        pending = []
        for m in msgs[2:]:
            if m.get("tool_calls"):
                assert not pending
                for tc in m["tool_calls"]:
                    assert isinstance(tc["function"]["arguments"], dict)
                pending = [tc["id"] for tc in m["tool_calls"]]
            elif m["role"] == "tool":
                assert m["tool_call_id"] == pending.pop(0)
        assert not pending
        assert [t["function"]["name"] for t in row["tools"]] == [
            "lookup_quote",
            "get_cost",
            "check_stock",
            "calc_price",
            "create_quote",
            "send_reply",
        ]


def test_families_and_paired_tasks(data):
    out, manifest, _ = data
    assert manifest["families"]["eval_h1"] == {"standard": 19, "stock_short": 7, "unknown_sku": 4}
    assert manifest["families"]["eval_trap"] == {"approved_valid": 4, "ack_already_sent": 4, "approved_stale": 4}
    a = [m["task_id"] for m, _ in episodes(out, "a", "train")]
    b = [m["task_id"] for m, _ in episodes(out, "b", "train")]
    assert a == b and len(a) == 150
    assert len(CATALOG) == 42


def test_dev_report_compares_every_run_with_both_workers(data, tmp_path):
    data = data[0]
    tasks = read_jsonl(data / "tasks" / "dev.jsonl")
    by_id = {t["task_id"]: t for t in tasks}
    for style in ("a", "b"):
        rows, _ = score_records(data / "records" / f"{style}-dev.jsonl", by_id)
        (tmp_path / style).mkdir()
        (tmp_path / style / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = report(tasks, data / "records", [("a", "A", tmp_path / "a"), ("b", None, tmp_path / "b")])
    differ = result["inputs"]["differ"]
    a, b = result["runs"]["a"], result["runs"]["b"]
    # After an acknowledgement already went out, both workers check cost first and quote once; everywhere else they differ.
    assert differ == len(tasks) - sum(t["family"] == "ack_already_sent" for t in tasks)
    assert a["outcomes"] == {"match": len(tasks)} and (a["like_A"], a["like_B"]) == (differ, 0)
    assert b["outcomes"] is None and (b["like_A"], b["like_B"]) == (0, differ)

    rows = read_jsonl(tmp_path / "a" / "scores.jsonl")
    for name, broken in (("one-missing", rows[:-1]), ("twice", rows + rows[:1]), ("old-dev", rows[:12])):
        (tmp_path / name).mkdir()
        (tmp_path / name / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in broken))
        with pytest.raises(SystemExit):
            report(tasks, data / "records", [(name, "A", tmp_path / name)])
