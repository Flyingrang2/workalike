"""quote-desk/2 tickets, worker C and the scorer: what counts as valid work, and as the same way of working.

The meaning is pinned down on worker C's episodes first, then on mutated ones. The record contract
is checked here too: a training example holds only what was on screen at that moment.
"""
import copy
import hashlib
import json
import random

import pytest

from island import worker_c
from island.score2 import compare, score, score_records
from island.tasks2 import FIXED_SHA256, SPLITS, make_tasks
from workalike.desk import Desk
from workalike.desk2 import floor_price
from workalike.envs import V2
from workalike.filter import clean_episode
from workalike.record import Recorder, read_events
from workalike.sft import sft_row, write_jsonl


def tasks(split="dev"):
    return make_tasks(11, split)


def by_family(name, split="train", n=0):
    return [t for t in tasks(split) if t["family"] == name][n]


def run_c(task, tmp_path):
    path = tmp_path / f"c-{task['task_id']}.jsonl"
    rec = Recorder(path)
    rec.start("e-" + task["task_id"], task["view"], task_id=task["task_id"], actor="c", env=V2.version)
    rec.final(worker_c.work(Desk(task["hidden"]["state"], rec, env=V2), task, random.Random(task["task_id"])))
    rec.end()
    return path, clean_episode(read_events(path))


def calls_of(ep):
    return [c for t in ep["turns"] for c in t["calls"]]


def replay(task, calls):
    """An episode from a list of (name, args), run against the real desk."""
    done = Desk(task["hidden"]["state"], env=V2).call_many(calls)
    return {"view": task["view"], "turns": [{"calls": done}], "final": "Done."}


def test_splits_have_the_planned_sizes_and_fixed_contents(tmp_path):
    assert {s: len(make_tasks(11, s)) for s in SPLITS} == {"train": 60, "dev": 12, "eval": 24, "dev2": 44, "train150": 150}
    for s in SPLITS:  # written the way gen_v2 writes them
        path = tmp_path / f"{s}.jsonl"
        write_jsonl(make_tasks(11, s), path)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == FIXED_SHA256[s], s
    assert json.dumps(make_tasks(11, "eval")) == json.dumps(make_tasks(11, "eval"))
    eval_combos = {t["family"] for t in tasks("eval") if "+" in t["family"]}
    assert not eval_combos & {t["family"] for t in tasks("train")}


def test_twins_differ_only_in_their_condition():
    pairs = {}
    for t in tasks("train"):
        if t["grader"]["pair"]:
            pairs.setdefault(t["grader"]["pair"], []).append(t)
    assert len(pairs) == 4
    for a, b in pairs.values():
        assert a["view"]["from"] == b["view"]["from"] and a["view"]["today"] == b["view"]["today"]
        assert a["family"] != b["family"]


@pytest.mark.parametrize("split", ["train", "dev"])
def test_worker_c_handles_every_ticket_validly(split, tmp_path):
    for t in tasks(split):
        _, ep = run_c(t, tmp_path)
        s = score(t, ep)
        assert s["business_ok"] and s["tool_errors"] == 0, (t["task_id"], t["family"], s)
        assert s["end"] == ("accepted" if t["grader"]["feasible"] else "closed")


def test_worker_c_switches_with_the_condition():
    def way(t):
        return score(t, run_c_mem(t))["way"]
    open_, done = by_family("typo_open"), by_family("typo_confirmed")
    assert way(open_)["asked_first"] and not way(done)["asked_first"]
    so, sc = by_family("short_open"), by_family("short_constrained")
    assert way(so)["asked_first"] and not way(sc)["asked_first"] and way(sc)["offered_choice"]
    assert not way(by_family("standard"))["asked_first"] and not way(by_family("ack_followup"))["asked_first"]


def run_c_mem(task):
    desk = Desk(task["hidden"]["state"], env=V2)
    worker_c.work(desk, task, random.Random(task["task_id"]))
    return {"view": task["view"], "turns": [{"calls": desk.history}], "final": "Done."}


def test_c_against_itself_matches_everywhere(tmp_path):
    ts = tasks("dev")
    path = tmp_path / "c.jsonl"
    rec = Recorder(path)
    for t in ts:
        rec.start("e-" + t["task_id"], t["view"], task_id=t["task_id"], env=V2.version)
        rec.final(worker_c.work(Desk(t["hidden"]["state"], rec, env=V2), t, random.Random(t["task_id"])))
        rec.end()
    rows, summary = score_records(path, {t["task_id"]: t for t in ts})
    c = compare(rows, rows)
    assert summary["business_ok"] == 12 and c["same_way"] == 12 and c["both"] == 12 and c["pairs_both_sides"] == 3


def test_a_quote_turned_down_for_an_unrevealed_condition_is_not_an_error():
    t = next(t for t in tasks("train") if t["family"] == "short_open" and t["grader"]["short"]["preference"] == "split")
    g, ticket = t["grader"], t["view"]["ticket_id"]
    sku, qty = g["wants"][0]["sku"], g["wants"][0]["qty"]
    cost = t["hidden"]["state"]["catalog"][sku]["unit_cost"]
    price = Desk(t["hidden"]["state"], env=V2).call("calc_price", {"sku": sku, "qty": qty, "unit_cost": cost})["unit_price"]
    other = [{"sku": w["sku"], "qty": w["qty"], "ship_by": g["need_by"],
              "unit_price": Desk(t["hidden"]["state"], env=V2).call(
                  "calc_price", {"sku": w["sku"], "qty": w["qty"],
                                 "unit_cost": t["hidden"]["state"]["catalog"][w["sku"]]["unit_cost"]})["unit_price"]}
             for w in g["wants"][1:]]
    later = [{"sku": sku, "qty": qty, "unit_price": price, "ship_by": g["short"]["restock_date"]}] + other
    split = [{"sku": sku, "qty": g["short"]["on_hand"], "unit_price": price, "ship_by": g["need_by"]},
             {"sku": sku, "qty": qty - g["short"]["on_hand"], "unit_price": price,
              "ship_by": g["short"]["restock_date"]}] + other
    n = ticket.split("-")[1]
    ep = replay(t, [("create_quote", {"ticket_id": ticket, "lines": later}),
                    ("send_reply", {"ticket_id": ticket, "message": "Quote attached.", "quote_id": f"Q-{n}-1"}),
                    ("create_quote", {"ticket_id": ticket, "lines": split}),
                    ("send_reply", {"ticket_id": ticket, "message": "Split it is.", "quote_id": f"Q-{n}-2"})])
    s = score(t, ep)
    assert s["business_ok"] and s["turned_down"] == 1 and not s["way"]["asked_first"]
    stop = copy.deepcopy(ep)
    stop["turns"][0]["calls"] = stop["turns"][0]["calls"][:2]
    assert score(t, stop)["end"] == "waiting" and not score(t, stop)["business_ok"]


def test_prices_shown_to_the_customer_must_be_valid():
    t = next(t for t in tasks("train") if t["family"] == "standard"
             and t["hidden"]["state"]["catalog"][t["grader"]["wants"][0]["sku"]]["unit_cost"] >= 1)
    ep = run_c_mem(t)
    quote = next(c for c in calls_of(ep) if c["name"] == "create_quote")
    over = copy.deepcopy(ep)
    next(c for c in calls_of(over) if c["name"] == "create_quote")["result"]["lines"][0]["unit_price"] += 1.0
    assert any(r.startswith("over list") for r in score(t, over)["reasons"])
    cut = copy.deepcopy(ep)
    line = next(c for c in calls_of(cut) if c["name"] == "create_quote")["result"]["lines"][0]
    cost = t["hidden"]["state"]["catalog"][line["sku"]]["unit_cost"]
    line["unit_price"] = floor_price(cost)  # well under list, still above the floor
    t2 = copy.deepcopy(t)
    t2["grader"]["authority"], t2["grader"]["owner"] = 0.0, "hold"
    assert any("discount without authority" in r for r in score(t2, cut)["reasons"])
    loud = copy.deepcopy(ep)
    ask = {"id": "x", "name": "send_reply", "args": {"ticket_id": t["view"]["ticket_id"], "message": "About $40 all in."},
           "result": {"status": "sent", "customer_reply": "Thanks."}}
    loud["turns"].insert(0, {"calls": [ask]})
    s = score(t, loud)
    assert "price in a reply without a quote" in s["reasons"] and s["way"]["asked_first"]
    assert quote["result"]["quote_id"]


def test_owner_consent_counts_only_when_asked_before_the_quote():
    t = next(t for t in tasks("train") if t["family"] == "reorder_room" and t["grader"]["owner"] == "floor_ok"
             and t["grader"]["authority"] < 0.05)
    ep = run_c_mem(t)
    names = [c["name"] for c in calls_of(ep)]
    assert "ask_owner" in names and score(t, ep)["business_ok"] and score(t, ep)["way"]["discounted"]
    late = copy.deepcopy(ep)
    calls = calls_of(late)
    ask = calls.pop(names.index("ask_owner"))
    calls.append(ask)
    late["turns"] = [{"calls": calls}]
    assert any("discount without authority" in r for r in score(t, late)["reasons"])


def test_training_examples_hold_only_what_was_on_screen(tmp_path):
    hidden_words = ["push_back_over", "split_ok", "later_ok", "substitute_ok", "floor_ok", '"policy"', "preference",
                    "feasible", "grader", "wants"]
    for t in tasks("train") + tasks("dev"):
        _, ep = run_c(t, tmp_path)
        assert set(ep["view"]) <= {"today", "ticket_id", "from", "body", "thread", "authority", "sent_so_far"}
        row = sft_row(ep, V2)
        text = json.dumps(row["messages"], ensure_ascii=False)
        assert not any(w in text for w in hidden_words), t["task_id"]
        assert t["family"] not in text.replace("short_open", "") or t["family"] == "standard"
        state = t["hidden"]["state"]
        says = state["customers"][t["view"]["ticket_id"]]["says"]
        owner = state["owner"][t["view"]["ticket_id"]]["says"]
        msgs = [json.dumps(m, ensure_ascii=False) for m in row["messages"]]

        def first(pred):
            return next((i for i, m in enumerate(msgs) if pred(m)), None)
        for text_, tool in ((says["answer"], "send_reply"), (owner, "ask_owner")):
            seen = first(lambda m: json.dumps(text_)[1:-1] in m)
            called = first(lambda m: f'"name": "{tool}"' in m)
            assert seen is None or (called is not None and called < seen), (t["task_id"], tool)


def test_the_model_is_trained_on_exactly_the_prompts_it_sees_when_working(tmp_path):
    """Replay C's records through the inference loop: every prompt must be a prefix of the training row."""
    from workalike import agent
    from workalike.serialize import to_messages
    for t in tasks("train") + tasks("dev"):
        _, ep = run_c(t, tmp_path)
        assert ep["view"] == t["view"], t["task_id"]  # cleaning keeps every screen field
        train = json.loads(json.dumps(to_messages(ep, V2)))
        turns = [[{"name": c["name"], "arguments": c["args"]} for c in turn["calls"]] for turn in ep["turns"]]
        prompts = []

        def model(messages, tools, turns=turns, prompts=prompts, ep=ep):
            prompts.append(json.loads(json.dumps(messages)))
            k = len(prompts) - 1
            return {"tool_calls": turns[k]} if k < len(turns) else {"content": ep["final"]}
        messages, _ = agent.run(Desk(t["hidden"]["state"], env=V2), t["view"], model, env=V2)
        for p in prompts:
            assert p == train[:len(p)] and train[len(p)]["role"] == "assistant", t["task_id"]
        assert json.loads(json.dumps(messages)) == train, t["task_id"]


def test_asked_about_tells_a_real_question_from_a_bare_note():
    t = by_family("missing_date")
    ep = run_c_mem(t)
    assert score(t, ep)["asked_about"] == {"missing_date": True}
    bare = copy.deepcopy(ep)
    q = next(c for c in calls_of(bare) if c["name"] == "send_reply" and not c["args"].get("quote_id"))
    q["args"]["message"] = "Thanks for your message!"
    s = score(t, bare)
    assert s["way"]["asked_first"] and s["asked_about"] == {"missing_date": False}
