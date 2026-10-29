"""The product's way of working a request: drafts wait, questions go to the person, guards hold whatever it says."""
import itertools

from island import score2
from island.tasks2 import make_tasks
from workalike.desk import unit_price_for
from workalike.envs import V2
from workalike.product import PRODUCT_ENV, guard, price_problem, work
from workalike.record import Recorder, read_events


def task(family):
    return next(t for t in make_tasks(11, "train") if t["family"] == family and len(t["grader"]["wants"]) == 1)


def scripted(steps, rules_line="Rules used: R2, R7"):
    """A fake model that makes the given calls one turn at a time (each step a list of (name, args)), then
    finishes; a call without tools is the rules-used question."""
    turns = iter(steps)
    seen = []

    def model(messages, tools):
        seen.append(messages[-1])
        if not tools:
            return {"content": rules_line}
        step = next(turns, None)
        if step is None:
            return {"content": "Done."}
        return {"tool_calls": [{"name": n, "arguments": a} for n, a in step]}
    model.seen = seen
    return model


def quote_steps(t, off=0.0, then=()):
    tid, want = t["view"]["ticket_id"], t["grader"]["wants"][0]
    cost = t["hidden"]["state"]["catalog"][want["sku"]]["unit_cost"]
    price = round(unit_price_for(cost, want["qty"]) * (1 - off), 2)
    qid = f"Q-{tid.split('-')[-1]}-1"
    return [[("check_stock", {"sku": want["sku"], "qty": want["qty"]})],
            [("get_cost", {"sku": want["sku"]})],
            [("calc_price", {"sku": want["sku"], "qty": want["qty"], "unit_cost": cost})],
            [("create_quote", {"ticket_id": tid, "lines": [{"sku": want["sku"], "qty": want["qty"],
                                                              "unit_price": price, "ship_by": t["grader"]["need_by"]}]})],
            [("send_reply", {"ticket_id": tid, "message": f"Here is {qid}.", "quote_id": qid}), *then]]


def test_the_model_sees_the_test_bed_tools():
    assert PRODUCT_ENV.tool_specs is V2.tool_specs and PRODUCT_ENV.render_view is V2.render_view


def test_a_reply_is_held_and_nothing_after_it_runs(tmp_path):
    t = task("standard")
    tid = t["view"]["ticket_id"]
    extra = ("create_quote", {"ticket_id": tid, "lines": []})
    rec = Recorder(tmp_path / "agent.jsonl")
    job = work(t["view"], t["hidden"]["state"], scripted(quote_steps(t, then=[extra])), "R2. When ...", rec)
    assert job["outcome"] == "draft" and len(job["drafts"]) == 1 and job["used"] == [2, 7]
    assert job["history"][-1]["result"]["error"] == "stopped"          # the call after the reply did not run
    assert "customer_reply" not in job["history"][-2]["result"]       # no scripted customer answers a draft
    assert all(c["ok"] for c in guard(job["state"], tid, job["history"]))
    events = read_events(tmp_path / "agent.jsonl")
    assert events[0]["entrance"] == "agent" and events[-1]["type"] == "episode_end"
    assert t["hidden"]["state"]["outbox"] == []                       # the shop's own state is untouched


def test_a_question_to_the_owner_waits_for_the_person_and_their_answer_lets_it_go_on():
    t = task("reorder_room")
    tid = t["view"]["ticket_id"]
    ask = [[("ask_owner", {"ticket_id": tid, "question": "May I go 6% under list?"})]]
    job = work(t["view"], t["hidden"]["state"], scripted(ask + quote_steps(t)), "R1. ...")
    assert job["outcome"] == "asked" and job["questions"][0]["question"].startswith("May I") and not job["drafts"]
    answers = {tid: {"says": "Yes, down to the floor.", "policy": "floor_ok"}}
    job = work(t["view"], t["hidden"]["state"], scripted(ask + quote_steps(t, off=0.06)), "R1. ...", answers=answers)
    assert job["outcome"] == "draft" and job["history"][0]["result"] == {"owner_reply": "Yes, down to the floor."}
    assert all(c["ok"] for c in guard(job["state"], tid, job["history"]))


def test_guards_stop_a_discount_beyond_authority_and_amounts_the_desk_never_gave():
    t = task("reorder_room")  # authority 3%
    tid = t["view"]["ticket_id"]
    job = work(t["view"], t["hidden"]["state"], scripted(quote_steps(t, off=0.06)))
    bad = [c for c in guard(job["state"], tid, job["history"]) if not c["ok"]]
    assert [c["name"] for c in bad] == [f"견적 Q-{tid.split('-')[-1]}-1"] and "권한" in bad[0]["detail"]
    steps = quote_steps(t)
    steps[-1] = [("send_reply", {"ticket_id": tid, "message": "It comes to $1.00 each."})]
    job = work(t["view"], t["hidden"]["state"], scripted(steps))
    bad = {c["name"] for c in guard(job["state"], tid, job["history"]) if not c["ok"]}
    assert bad == {"가격은 견적으로만", "금액은 데스크 값"}


def test_finishing_without_a_draft_or_question_is_stopped_not_done():
    t = task("standard")
    job = work(t["view"], t["hidden"]["state"], scripted([]), "R1. ...")
    assert job["outcome"] == "stopped" and job["used"] is None
    assert not guard(job["state"], t["view"]["ticket_id"], job["history"])[0]["ok"]


def test_the_price_rule_agrees_with_the_scorer():
    for family, off in itertools.product(("reorder_room", "standard", "reorder_tight"), (-0.02, 0, 0.02, 0.05, 0.2)):
        t = task(family)
        s, tid, want = t["hidden"]["state"], t["view"]["ticket_id"], t["grader"]["wants"][0]
        price = round(unit_price_for(s["catalog"][want["sku"]]["unit_cost"], want["qty"]) * (1 - off), 2)
        lines = [{"sku": want["sku"], "qty": want["qty"], "unit_price": price}]
        ours = price_problem({**s, "owner_asked": []}, tid, lines[0], lines)
        theirs = score2.price_problem(t, lines[0], lines, owner_ok=False)
        assert (ours is None) == (theirs is None), (family, off, ours, theirs)


def test_every_written_amount_and_every_named_quote_is_checked():
    from workalike.product import amounts
    assert amounts("Total 1,200.50 USD, or ＄3.10 each, $.50 off, 12 dollars") == ["1,200.50", "3.10", ".50", "12"]
    t = task("reorder_room")  # authority 3%
    tid = t["view"]["ticket_id"]
    qid = f"Q-{tid.split('-')[-1]}-1"
    steps = quote_steps(t, off=0.06)
    steps[-1] = [("send_reply", {"ticket_id": tid, "message": f"Please see {qid}, total 1.00 USD."})]
    job = work(t["view"], t["hidden"]["state"], scripted(steps))
    bad = {c["name"] for c in guard(job["state"], tid, job["history"]) if not c["ok"]}
    assert bad == {"가격은 견적으로만", "금액은 데스크 값", f"견적 {qid}"}   # named, not attached, still checked
