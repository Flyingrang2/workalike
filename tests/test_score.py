"""Broken and partial episodes must not pass: the scorer is checked with mutated records."""
import copy
import random

from island import scripted
from island.score import score, score_records
from island.tasks import make_tasks
from workalike.desk import Desk
from workalike.filter import clean_episode
from workalike.record import Recorder, read_events


def run(task, style, tmp_path):
    rec = Recorder(tmp_path / f"{style}-{task['task_id']}.jsonl")
    rec.start("e-" + task["task_id"], task["view"], task_id=task["task_id"])
    rec.final(scripted.work(Desk(task["hidden"]["state"], rec), task, style, random.Random(0)))
    rec.end()
    return clean_episode(read_events(tmp_path / f"{style}-{task['task_id']}.jsonl"))


def standard(n=0):
    return [t for t in make_tasks(7, "dev", 12) if t["family"] == "standard"][n]


def calls_of(ep):
    return [c for t in ep["turns"] for c in t["calls"]]


def keep(ep, pred):
    out = copy.deepcopy(ep)
    out["turns"] = [{"calls": [c for c in t["calls"] if pred(c)]} for t in out["turns"]]
    out["turns"] = [t for t in out["turns"] if t["calls"]]
    return out


def test_reference_episodes_pass(tmp_path):
    t = standard()
    a, b = score(t, run(t, "a", tmp_path)), score(t, run(t, "b", tmp_path))
    assert a["correct"] and a["a_way"] and not a["b_way"]
    assert b["correct"] and b["b_way"] and not b["a_way"]


def test_empty_episode_fails_everything(tmp_path):
    t = standard()
    s = score(t, {"view": t["view"], "turns": [], "final": None})
    assert not s["correct"] and not s["a_way"] and not s["b_way"] and s["cost_before_first_reply"] is None


def test_cost_lookup_then_stop_is_not_the_a_way(tmp_path):
    t = standard()
    ep = keep(run(t, "a", tmp_path), lambda c: c["name"] == "get_cost")
    s = score(t, ep)
    assert s["cost_before_first_reply"] is True  # the raw order measure alone would count it
    assert not s["correct"] and not s["a_way"]


def test_ack_then_stop_is_not_the_b_way(tmp_path):
    t = standard()
    ep = keep(run(t, "b", tmp_path), lambda c: c["name"] == "send_reply" and not c["args"].get("quote_id"))
    s = score(t, ep)
    assert not s["correct"] and not s["b_way"]


def test_quote_never_sent_fails(tmp_path):
    t = standard()
    ep = keep(run(t, "a", tmp_path), lambda c: c["name"] != "send_reply")
    assert score(t, ep)["reasons"] == ["quote never sent"]


def test_wrong_price_fails(tmp_path):
    t = standard()
    ep = run(t, "a", tmp_path)
    q = next(c for c in calls_of(ep) if c["name"] == "create_quote")
    q["result"]["lines"][0]["unit_price"] += 0.5
    assert any("wrong price" in r for r in score(t, ep)["reasons"])


def test_price_in_acknowledgement_fails(tmp_path):
    t = standard()
    ep = run(t, "b", tmp_path)
    ack = next(c for c in calls_of(ep) if c["name"] == "send_reply" and not c["args"].get("quote_id"))
    ack["args"]["message"] += " Should be around $4 each."
    s = score(t, ep)
    assert "price promised before a quote" in s["reasons"] and not s["b_way"]


def test_failed_cost_lookup_before_reply_is_not_the_a_way(tmp_path):
    t = standard()
    ep = run(t, "a", tmp_path)
    calls = calls_of(ep)
    for c in calls:
        if c["name"] == "get_cost":
            c["result"] = {"error": "unknown_sku", "sku": c["args"]["sku"], "did_you_mean": None}
    s = score(t, ep)
    assert s["cost_before_first_reply"] is True and not s["a_way"]


def test_double_acknowledgement_fails_on_already_acked_ticket(tmp_path):
    t = next(t for t in make_tasks(7, "dev", 12) if t["family"] == "ack_already_sent")
    ep = run(t, "b", tmp_path)
    reply = next(c for c in calls_of(ep) if c["name"] == "send_reply")
    ep["turns"].insert(0, {"calls": [{**copy.deepcopy(reply), "args": {"ticket_id": reply["args"]["ticket_id"],
                                                                         "message": "Thanks, on it."}}]})
    assert "acknowledged twice" in score(t, ep)["reasons"]


def test_unneeded_cost_lookup_is_flagged(tmp_path):
    t = next(t for t in make_tasks(7, "dev", 12) if t["family"] == "approved_valid")
    ep = run(t, "a", tmp_path)
    assert not score(t, ep)["unneeded_cost_lookup"]
    sku = calls_of(ep)[1]["args"]["sku"]
    ep["turns"].insert(1, {"calls": [{"id": "x", "name": "get_cost", "args": {"sku": sku}, "result": {"sku": sku}}]})
    assert score(t, ep)["unneeded_cost_lookup"]


def test_missing_and_duplicate_records_count_against(tmp_path):
    tasks = [t for t in make_tasks(7, "dev", 12)][:3]
    path = tmp_path / "r.jsonl"
    for t in [tasks[0], tasks[0], tasks[1]]:
        rec = Recorder(path)
        rec.start("e-" + t["task_id"] + str(random.random()), t["view"], task_id=t["task_id"])
        rec.final(scripted.work(Desk(t["hidden"]["state"], rec), t, "a", random.Random(0)))
        rec.end()
    rows, summary = score_records(path, {t["task_id"]: t for t in tasks})
    assert summary["tasks"] == 3 and summary["correct"] == 2 and summary["duplicates_or_unknown"] == 1
    assert summary["status"]["missing"] == 1


def test_judge_passes_reference_and_catches_order_only(tmp_path):
    from island.gen_h1 import build
    from island.judge_h1 import judge
    from workalike.sft import read_jsonl

    out = tmp_path / "h1"
    build(out, 7)
    rows = {}
    for split in ("eval_h1", "eval_trap"):
        tasks = {t["task_id"]: t for t in read_jsonl(out / "tasks" / f"{split}.jsonl")}
        for style in "ab":
            rows[style, split] = score_records(out / "records" / f"{style}-{split}.jsonl", tasks)[0]
    ref = judge(rows["a", "eval_h1"], rows["a", "eval_h1"], rows["b", "eval_h1"],
                rows["a", "eval_trap"], rows["b", "eval_trap"])
    assert ref["verdict"] == "pass" and ref["first_criteria"]["pass"]

    # B that acknowledges twice on already-acknowledged tickets: order holds, restraint fails
    b_trap = [dict(r, correct=False) if r["family"] == "ack_already_sent" else r for r in rows["b", "eval_trap"]]
    res = judge(rows["a", "eval_h1"], rows["a", "eval_h1"], rows["b", "eval_h1"], rows["a", "eval_trap"], b_trap)
    assert res["verdict"] == "order imitated, conditional behaviour not reproduced"

    # an adapter that only looks up cost and stops: the first criteria pass it, the new ones do not
    stopped = [dict(r, correct=False, a_way=False, cost_before_first_reply=True) for r in rows["a", "eval_h1"]]
    res = judge(stopped, stopped, rows["b", "eval_h1"], rows["a", "eval_trap"], rows["b", "eval_trap"])
    assert res["first_criteria"]["pass"] and res["verdict"] == "fail"


def family(name, n=0):
    return [t for t in make_tasks(7, "dev", 24) if t["family"] == name][n]


def test_cost_lookup_on_a_valid_approved_quote_is_not_correct(tmp_path):
    t = family("approved_valid")
    ep = run(t, "a", tmp_path)
    sku = calls_of(ep)[0]["result"]["lines"][0]["sku"]
    ep["turns"].insert(1, {"calls": [{"id": "x1", "name": "get_cost", "args": {"sku": sku},
                                      "result": Desk(t["hidden"]["state"]).call("get_cost", {"sku": sku})}]})
    s = score(t, ep)
    assert s["business_correct"] and not s["restraint_ok"] and not s["correct"]
    assert s["reasons"] == ["unneeded cost lookup"]


def test_kept_item_shipping_before_restock_is_wrong(tmp_path):
    t = family("stock_short")
    ep = run(t, "a", tmp_path)
    quote = next(c for c in calls_of(ep) if c["name"] == "create_quote")
    quote["result"]["ship_by"] = t["grader"]["need_by"]  # the desk refuses this date; the scorer must too
    s = score(t, ep)
    assert not s["correct"] and "ships before restock" in s["reasons"]


def test_a_second_quote_or_an_amount_the_desk_never_gave_is_wrong(tmp_path):
    t = standard()
    ep = run(t, "a", tmp_path)
    reply = [c for c in calls_of(ep) if c["name"] == "send_reply"][-1]
    wrong = copy.deepcopy(ep)
    [c for c in calls_of(wrong) if c["name"] == "send_reply"][-1]["args"]["message"] = reply["args"]["message"] + " Or $1.00 flat."
    assert "reply amount the desk never gave" in score(t, wrong)["reasons"]
    early = copy.deepcopy(reply)
    early["args"]["quote_id"] = "Q-0-1"
    twice = copy.deepcopy(ep)
    twice["turns"].insert(0, {"calls": [early]})
    assert "sent a quote the desk never made" in score(t, twice)["reasons"]


def two_quotes(t, tmp_path, order):
    """A short ticket answered with both valid quotes, made in the given order, the first one attached."""
    desk = Desk(t["hidden"]["state"])
    g, sku = t["grader"], t["grader"]["short"]["sku"]
    sub = next(iter(g["short"]["substitutes"]))
    later = [dict(l) for l in g["lines"]], g["short"]["restock_date"]
    swap = [dict(l, sku=sub, unit_price=g["short"]["substitutes"][sub]) if l["sku"] == sku else dict(l)
            for l in g["lines"]], g["need_by"]
    calls = []
    for lines, ship in (later, swap) if order == "later_first" else (swap, later):
        calls.append(("create_quote", {"ticket_id": t["view"]["ticket_id"], "lines": lines, "ship_by": ship}))
    done = desk.call_many(calls)
    ids = [d["result"]["quote_id"] for d in done]
    msg = f"Two options: {ids[0]} is attached, and {ids[1]} is the other one. Pick whichever suits you."
    done += desk.call_many([("send_reply", {"ticket_id": t["view"]["ticket_id"], "message": msg, "quote_id": ids[0]})])
    return {"view": t["view"], "turns": [{"calls": done}], "final": "Done."}


def test_two_valid_quotes_score_the_same_in_either_order(tmp_path):
    t = family("stock_short")
    scores = [score(t, two_quotes(t, tmp_path, o)) for o in ("later_first", "swap_first")]
    for s in scores:
        assert s["correct"] and s["reasons"] == [] and s["short_choice"] == "both"


def test_an_invalid_second_quote_offered_is_wrong(tmp_path):
    t = family("stock_short")
    ep = two_quotes(t, tmp_path, "later_first")
    made = [c for c in calls_of(ep) if c["name"] == "create_quote"]
    made[1]["result"]["lines"][0]["unit_price"] += 0.5
    s = score(t, ep)
    assert not s["correct"] and any(r.startswith(made[1]["result"]["quote_id"] + ": wrong price") for r in s["reasons"])


def test_a_price_in_words_before_the_quote_is_a_promise(tmp_path):
    t = standard()
    ep = run(t, "b", tmp_path)
    ack = next(c for c in calls_of(ep) if c["name"] == "send_reply")
    ack["args"]["message"] += " Roughly 100 USD all in."
    assert "price promised before a quote" in score(t, ep)["reasons"]
