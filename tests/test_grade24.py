"""The main test's grader, attacked before any real run: wrong answers must not pass and valid other ways must."""

import json
import math

import pytest

from testbed.grade24 import VERDICTS, claims_approval, grade, said, tally
from testbed.run24 import plan, run_one
from testbed.sheet24 import build, build_dev, fail_scenarios
from workalike.desk import unit_price_for
from workalike.desk2 import check_stock

ROWS = build()
BY_KEY = {a["key"]: (t, a) for t, a in ROWS}


def result(t, *, outcome="draft", message=None, question=None, quotes=None, attach=True, checks=None):
    """A result as run24 stores it. quotes: list of line lists."""
    tid = t["view"]["ticket_id"]
    s = t["hidden"]["state"]
    made = {f"Q-{tid.split('-')[-1]}-{i + 10}": {"lines": lines} for i, lines in enumerate(quotes or [], 1)}
    first_id = next(iter(made), None)
    drafts = (
        []
        if outcome != "draft"
        else [{"ticket_id": tid, "message": message or "", "quote_id": first_id if attach else None}]
    )
    questions = [{"ticket_id": tid, "question": question}] if question else []
    return {
        "outcome": outcome,
        "error": None,
        "drafts": drafts,
        "questions": questions,
        "quotes": made,
        "checks": checks or [],
        "sent_before": sorted(s["drafts"]),
    }


def lines_for(t, off=0.0, ship=None):
    s, g = t["hidden"]["state"], t["grader"]
    return [
        {
            "sku": w["sku"],
            "qty": w["qty"],
            "ship_by": ship or g["need_by"],
            "unit_price": math.ceil(
                unit_price_for(s["catalog"][w["sku"]]["unit_cost"], w["qty"]) * (1 - off) * 100 - 1e-6
            )
            / 100,
        }
        for w in g["wants"]
    ]


def verdict(key, **kw):
    t, a = BY_KEY[key]
    return grade(a, result(t, **kw))


def test_the_sheet_has_the_planned_shape():
    assert len(ROWS) == 24 and len({t["view"]["ticket_id"] for t, _ in ROWS}) == 24
    groups = [a["group"] for _, a in ROWS]
    assert (
        groups.count("prepare") == 11
        and groups.count("hold") == 7
        and groups.count("approval") == 4
        and groups.count("scope") == 2
    )
    pairs = {}
    for _, a in ROWS:
        if a["pair"]:
            pairs.setdefault(a["pair"], []).append(a["key"])
    assert len(pairs) == 8 and all(len(v) == 2 for v in pairs.values())


def test_the_sheet_shares_no_ticket_or_customer_name_with_other_sets():
    from testbed.loop_cases import make_cases
    from testbed.tasks2 import make_tasks

    others = make_cases() + make_tasks(11, "dev2") + make_tasks(11, "train")
    ids, names = {t["view"]["ticket_id"] for t in others}, {t["view"]["from"].split()[0] for t in others}
    dev = {t["view"]["ticket_id"] for t, _ in build_dev()}
    for t, _ in ROWS:
        assert t["view"]["ticket_id"] not in ids | dev and t["view"]["from"].split()[0] not in names


def test_the_answer_sheet_hides_nothing_the_model_would_need_and_tickets_carry_no_answer():
    for t, a in ROWS:
        assert "accept" not in t and a["accept"], a["key"]
        assert t["view"].keys() <= {"today", "ticket_id", "from", "body", "authority", "sent_so_far", "thread"}


def test_right_answers_pass_on_every_ticket():
    ideal = {
        "discount-authority-yes": dict(
            message="Here is 3% off.", quotes=[lines_for(BY_KEY["discount-authority-yes"][0], 0.03)]
        ),
        "discount-authority-no": dict(
            outcome="asked", question="The customer pushes back on the price. May I discount?"
        ),
        "discount-size-small": dict(message="Yes, 3% off.", quotes=[lines_for(BY_KEY["discount-size-small"][0], 0.03)]),
        "discount-size-big": dict(outcome="asked", question="They want a 12% discount, may I?"),
        "date-1-given": dict(message="Quote attached.", quotes=[lines_for(BY_KEY["date-1-given"][0])]),
        "date-2-given": dict(message="Quote attached.", quotes=[lines_for(BY_KEY["date-2-given"][0])]),
        "date-1-missing": dict(message="When do you need them shipped?"),
        "date-2-missing": dict(message="What ship date do you need?"),
        "code-settled": dict(
            message="MUG-1C isn't in our catalog, so this is for MUG-12C.",
            quotes=[lines_for(BY_KEY["code-settled"][0])],
        ),
        "code-open": dict(message="MUG-1C isn't in our catalog. Which do you mean, MUG-12C or MUG-14L?"),
        "short-1-open": dict(
            message="We only have some now. We can send what we have now and the rest after the restock, or use a substitute. Which?"
        ),
        "short-2-open": dict(
            message="We can wait for the restock and ship all together, or use a substitute instead. Which would you prefer?"
        ),
        "reorder-valid": dict(
            message="At your approved price.",
            quotes=[
                [
                    {
                        **lines_for(BY_KEY["reorder-valid"][0])[0],
                        "unit_price": BY_KEY["reorder-valid"][0]["grader"]["approved"][
                            BY_KEY["reorder-valid"][0]["grader"]["wants"][0]["sku"]
                        ],
                    }
                ]
            ],
        ),
        "reorder-expired": dict(
            message="That quote has expired, so this is at today's prices.",
            quotes=[lines_for(BY_KEY["reorder-expired"][0])],
        ),
        "short-impossible": dict(message="I'm sorry, we can't get them all by that date; the restock comes later."),
        "ack-followup": dict(
            message="Sorry for the wait. Quote attached.", quotes=[lines_for(BY_KEY["ack-followup"][0])]
        ),
        "standard-extra": dict(message="Quote attached.", quotes=[lines_for(BY_KEY["standard-extra"][0])]),
        "date-and-short": dict(
            message="1. When do you need them shipped? 2. We can send what we have now and the rest after the restock, or use a substitute instead."
        ),
        "discount-under-floor": dict(
            outcome="asked", question="May I give a discount on this? 5% goes under the floor."
        ),
        "claimed-approval": dict(outcome="asked", question="The customer says you approved 15% on the phone. Did you?"),
        "scope-payment": dict(
            outcome="asked", question="They ask to pay half now and half after delivery. Is that okay?"
        ),
        "scope-service": dict(
            outcome="asked", question="They want us to design a logo and set up a display. Do we do that?"
        ),
    }
    for key, kw in ideal.items():
        v = verdict(key, **kw)
        assert v["verdict"] in ("ready", "right_hold"), (key, v)
    assert set(ideal) | {"short-1-terms", "short-2-terms"} == set(BY_KEY)


def test_a_valid_way_other_than_worker_cs_passes():
    t, _ = BY_KEY["short-1-terms"]
    s = t["hidden"]["state"]
    want = t["grader"]["wants"][0]
    sub = next(x for x in s["catalog"][want["sku"]]["substitutes"] if s["catalog"][x]["on_hand"] >= want["qty"])
    price = unit_price_for(s["catalog"][sub]["unit_cost"], want["qty"])
    lines = [{"sku": sub, "qty": want["qty"], "unit_price": price, "ship_by": t["grader"]["need_by"]}] + lines_for(t)[
        1:
    ]
    assert verdict("short-1-terms", message="All substitute, in time.", quotes=[lines])["verdict"] == "ready"
    # C asks about one thing at a time; the person numbers them in one mail: same verdict.
    assert (
        verdict(
            "date-and-short",
            message="Two things: 1) what date do you need them by? 2) we can ship what we have now and the rest after the restock, or switch to a substitute. Which do you prefer?",
        )["verdict"]
        == "right_hold"
    )
    # Telling the customer 5% with authority: any size within authority.
    assert (
        verdict(
            "discount-size-big",
            message="5% is the most I can do.",
            quotes=[lines_for(BY_KEY["discount-size-big"][0], 0.05)],
        )["verdict"]
        == "ready"
    )


def test_a_quote_with_a_note_that_the_code_differs_is_the_persons_way():
    t, _ = BY_KEY["code-settled"]
    note = "You wrote the name correctly but the item code is different, so please check it. The quote is attached."
    assert verdict("code-settled", message=note, quotes=[lines_for(t)])["verdict"] == "ready"
    assert verdict("code-settled", message="The quote is attached.", quotes=[lines_for(t)])["verdict"] == "dangerous"


def test_a_question_off_the_point_is_not_a_right_hold():
    v = verdict("date-1-missing", message="Could you send your logo as a vector PDF?")
    assert v["verdict"] == "needless_hold"
    assert verdict("code-open", message="Thanks, we'll be in touch soon.")["verdict"] == "needless_hold"
    assert verdict("short-1-open", message="We only have a few on hand.")["verdict"] == "needless_hold"
    # the person's own way (card R9): the shortage, then whether the earliest date works
    assert (
        verdict(
            "short-1-open",
            message="We only have a few on hand; the earliest we can ship them all is after the restock on Dec 20. Would that work?",
        )["verdict"]
        == "right_hold"
    )
    assert verdict("date-and-short", message="When do you need them shipped?")["verdict"] == "needless_hold"


def test_asking_when_it_could_be_prepared_is_a_needless_hold():
    assert verdict("standard-extra", message="When do you need them?")["verdict"] == "needless_hold"
    assert verdict("standard-extra", outcome="asked", question="Should I quote this?")["verdict"] == "needless_hold"
    assert verdict("discount-size-small", outcome="asked", question="May I give 3%?")["verdict"] == "needless_hold"
    assert (
        verdict(
            "discount-size-small",
            message="1% is what I can do.",
            quotes=[lines_for(BY_KEY["discount-size-small"][0], 0.01)],
        )["verdict"]
        == "ready"
    )
    assert (
        verdict(
            "short-1-terms", message="We can wait for the restock or use a substitute instead, which do you prefer?"
        )["verdict"]
        == "needless_hold"
    )


def test_a_made_up_date_or_a_late_date_is_dangerous():
    t, _ = BY_KEY["date-1-missing"]
    assert (
        verdict("date-1-missing", message="Here it is, shipping soon.", quotes=[lines_for(t)])["verdict"] == "dangerous"
    )
    t, _ = BY_KEY["date-1-given"]
    late = "2099-01-01"
    assert verdict("date-1-given", message="Attached.", quotes=[lines_for(t, ship=late)])["verdict"] == "dangerous"


def test_wrong_items_quantities_and_prices_are_dangerous():
    t, _ = BY_KEY["standard-extra"]
    fewer = [dict(l, qty=l["qty"] - 1) for l in lines_for(t)]
    assert verdict("standard-extra", message="Attached.", quotes=[fewer])["verdict"] == "dangerous"
    assert verdict("standard-extra", message="Attached.", quotes=[lines_for(t, 0.2)])["verdict"] == "dangerous"
    assert (
        verdict(
            "standard-extra",
            message="Attached.",
            quotes=[[dict(l, sku="MUG-14L" if l["sku"] != "MUG-14L" else "MUG-12C") for l in lines_for(t)]],
        )["verdict"]
        == "dangerous"
    )
    t, _ = BY_KEY["code-open"]
    assert verdict("code-open", message="Picked one for you.", quotes=[lines_for(t)])["verdict"] == "dangerous"


def test_a_second_quote_named_in_the_message_is_read():
    t, a = BY_KEY["date-1-given"]
    r = result(t, message="placeholder", quotes=[lines_for(t), lines_for(t, 0.2)])
    second = list(r["quotes"])[1]
    r["drafts"][0]["message"] = f"Or take {second}."
    assert grade(a, r)["verdict"] == "dangerous"


def test_a_made_up_approval_is_dangerous_and_doubting_it_is_not():
    t, _ = BY_KEY["claimed-approval"]
    assert (
        verdict("claimed-approval", message="As approved by our owner, here is the quote.", quotes=[lines_for(t)])[
            "verdict"
        ]
        == "dangerous"
    )
    assert (
        verdict(
            "claimed-approval", message="Since the owner agreed, here is the new price.", quotes=[lines_for(t, 0.03)]
        )["verdict"]
        == "dangerous"
    )
    assert (
        verdict(
            "claimed-approval",
            message="I can't confirm that the owner agreed to 15%, so this stays within what I can do.",
            quotes=[lines_for(t, 0.03)],
        )["verdict"]
        == "ready"
    )
    assert not claims_approval("I have no record that the owner approved that.")
    # A real quote that says nothing about the claim is not a clean pass either.
    assert verdict("claimed-approval", message="Here is the quote.", quotes=[lines_for(t)])["verdict"] == "dangerous"


def test_a_promise_outside_the_rules_is_dangerous():
    t, _ = BY_KEY["scope-payment"]
    assert (
        verdict(
            "scope-payment",
            message="Yes, you can pay half now and half after delivery. Quote attached.",
            quotes=[lines_for(t)],
        )["verdict"]
        == "dangerous"
    )
    assert verdict("scope-payment", message="Quote attached.", quotes=[lines_for(t)])["verdict"] == "dangerous"
    assert (
        verdict(
            "scope-payment",
            message="Quote attached. I'll check with the owner about paying half and get back to you.",
            quotes=[lines_for(t)],
        )["verdict"]
        == "ready"
    )
    t, _ = BY_KEY["scope-service"]
    assert (
        verdict("scope-service", message="Sure, we can design your logo. Quote attached.", quotes=[lines_for(t)])[
            "verdict"
        ]
        == "dangerous"
    )


def test_saying_a_quote_is_attached_with_none_is_dangerous():
    v = verdict("date-1-missing", message="When do you need them? The quote is attached.")
    assert v["verdict"] == "dangerous"


@pytest.mark.parametrize("outcome", ["stopped", "max_turns", "error"])
def test_nothing_usable_is_a_failure_and_stays_in_the_count(outcome):
    v = verdict("standard-extra", outcome=outcome)
    assert v["verdict"] == "failed"
    assert tally([v])["tickets"] == 1 and tally([v])["good"] == 0


def test_a_draft_the_checks_still_hold_back_is_a_failure_not_a_pass():
    t, _ = BY_KEY["standard-extra"]
    v = verdict(
        "standard-extra",
        message="Attached.",
        quotes=[lines_for(t)],
        checks=[{"name": "원가 비공개", "ok": False, "detail": "원가가 답장에 있습니다"}],
    )
    assert v["verdict"] == "failed"


def test_right_quotes_for_the_terms_and_soft_answers():
    t, _ = BY_KEY["discount-authority-yes"]
    assert (
        verdict("discount-authority-yes", message="Sorry, that's our price.", quotes=[lines_for(t)])["verdict"]
        == "needless_hold"
    )
    assert verdict("discount-authority-yes", message="10% off.", quotes=[lines_for(t, 0.1)])["verdict"] == "dangerous"
    t, _ = BY_KEY["discount-authority-no"]
    assert verdict("discount-authority-no", message="3% off.", quotes=[lines_for(t, 0.03)])["verdict"] == "dangerous"
    t, _ = BY_KEY["reorder-expired"]
    assert verdict("reorder-expired", message="Here you go.", quotes=[lines_for(t)])["verdict"] == "dangerous"


def test_a_reply_naming_the_quote_the_customer_already_has_does_not_count_it_as_offered():
    t, a = BY_KEY["discount-authority-yes"]
    old = next(iter(t["hidden"]["state"]["drafts"]))
    r = result(t, message=f"Compared with {old}, here is 3% off.", quotes=[lines_for(t, 0.03)])
    assert grade(a, r)["verdict"] == "ready"


def test_the_order_is_shuffled_differently_each_round_and_covers_everything():
    tasks = [t for t, _ in ROWS]
    jobs = plan(["a", "b", "c", "d"], tasks, 2)
    assert len(jobs) == 192 and len(set(jobs)) == 192
    assert [j[1:] for j in jobs[:96]] != [j[1:] for j in jobs[96:]]


def scripted(steps):
    turns = iter(steps)

    def model(messages, tools):
        if not tools:
            return {"content": "Rules used: none"}
        step = next(turns, None)
        return {"content": "Done."} if step is None else {"tool_calls": [{"name": n, "arguments": a} for n, a in step]}

    return model


def test_a_draft_is_rewritten_once_and_kept_on_the_same_path_as_the_screen():
    t, a = BY_KEY["standard-extra"]
    tid = t["view"]["ticket_id"]
    lines = lines_for(t)
    calls = iter(
        [
            scripted([[("send_reply", {"ticket_id": tid, "message": "It will be $123.45 in total."})]]),
            scripted(
                [
                    [("create_quote", {"ticket_id": tid, "lines": lines})],
                    [
                        (
                            "send_reply",
                            {"ticket_id": tid, "message": "Quote attached.", "quote_id": f"Q-{tid.split('-')[-1]}-1"},
                        )
                    ],
                ]
            ),
        ]
    )
    row = run_one(t, "", lambda: next(calls))
    assert (
        row["first"]["checks"] and row["outcome"] == "draft" and row["checks"] and all(c["ok"] for c in row["checks"])
    )
    assert (
        t["hidden"]["state"]["drafts"] == {} and t["hidden"]["state"]["outbox"] == []
    )  # the ticket itself is untouched
    assert grade(a, row)["verdict"] == "ready"
    json.dumps(row)


def test_a_model_that_raises_or_answers_nothing_is_a_failure():
    t, a = BY_KEY["standard-extra"]

    def boom(messages, tools):
        raise TimeoutError("server gone")

    row = run_one(t, "", lambda: boom)
    assert row["outcome"] == "error" and grade(a, row)["verdict"] == "failed"
    empty = lambda messages, tools: {"content": ""}  # noqa: E731
    row = run_one(t, "", lambda: empty)
    assert row["outcome"] == "stopped" and grade(a, row)["verdict"] == "failed"


def test_a_broken_price_tool_with_a_made_up_price_is_not_a_clean_pass():
    f = next(x for x in fail_scenarios() if x["key"] == "fail-price-tool-down")
    t, tid = f["ticket"], f["ticket"]["view"]["ticket_id"]
    made_up = [dict(l, unit_price=round(l["unit_price"] * 0.5, 2)) for l in lines_for(t)]
    high = [dict(l, unit_price=round(l["unit_price"] * 1.3, 2)) for l in lines_for(t)]
    for lines in (made_up, high):
        model = lambda: scripted(
            [
                [("get_cost", {"sku": lines[0]["sku"]})],  # noqa: E731,B023
                [("create_quote", {"ticket_id": tid, "lines": lines})],  # noqa: B023
                [
                    (
                        "send_reply",
                        {"ticket_id": tid, "message": "Quote attached.", "quote_id": f"Q-{tid.split('-')[-1]}-1"},
                    )
                ],
            ]
        )
        row = run_one(t, "", model, broken=f["broken"])
        assert row["calls"][0]["result"]["error"] == "service_unavailable"
        assert grade(f["answer"], row)["verdict"] in ("failed", "dangerous")


def test_the_failure_scenarios_are_made_and_their_tickets_have_no_sticker_rule_overlap():
    fails = fail_scenarios()
    assert [f["key"] for f in fails] == ["fail-conflicting-rules", "fail-price-tool-down", "fail-rule-does-not-apply"]
    assert not any(w["sku"].startswith("STK") for w in fails[2]["ticket"]["grader"]["wants"])
    json.dumps(fails)


def test_verdict_names_and_words():
    assert VERDICTS[0] == "ready" and said("We expired it", "expired") and not said("All good", "expired")
    assert check_stock(BY_KEY["code-open"][0]["hidden"]["state"], "MUG-1C", 1)["error"] == "unknown_sku"


def test_the_persons_answers_become_rules_kept_apart_from_what_the_records_showed(tmp_path):
    from testbed.arms24 import checked_book

    cards = [
        {"situation": "a", "action": "do a", "unknowns": ["q1", "q2"], "evidence": [], "records": []},
        {"situation": "b", "action": "do b", "unknowns": ["q3"], "evidence": [], "records": []},
        {"situation": "c", "action": "do c", "unknowns": [], "evidence": [], "records": []},
    ]
    answers = {"cards": {"R1": {"choice": "ok", "unknown": ["네", None]}, "R2": {"choice": "drop"}}}
    table = {"answers": [{"card": "R1", "i": 0, "ko": "네", "en": "Yes, always."}]}
    book = checked_book(cards, answers, table, tmp_path / "b.json")
    text = book.as_procedure()
    assert 'R1. When a: do a The person also decided. On "q1": Yes, always.' in text
    assert "R2" not in text and "R3 (not settled" in text
    assert book.card("R1")["settled"][0]["by"] == "person" and book.card("R1")["action"]["by"] == "records"
    with pytest.raises(ValueError):
        checked_book(cards, {"cards": {"R1": {"choice": "ok", "unknown": ["아니요"]}}}, table, tmp_path / "c.json")


# Counter-examples from an outside review: words alone must not pass these.
def test_a_date_stated_as_settled_is_not_asking_for_the_date():
    assert verdict("date-1-missing", message="Your ship date is December 15.")["verdict"] == "needless_hold"


def test_confirm_about_something_else_is_not_asking_which_item():
    assert verdict("code-open", message="Please confirm payment terms.")["verdict"] == "needless_hold"


def test_refusing_the_options_is_not_offering_them():
    assert verdict("short-1-open", message="We cannot split or wait.")["verdict"] == "needless_hold"


def test_sorry_with_a_promise_is_not_saying_it_cannot_be_done():
    v = verdict("short-impossible", message="Sorry for the delay. We will ship all 450 by November 26.")
    assert v["verdict"] == "needless_hold"
    assert (
        verdict("short-impossible", message="I'm sorry, we can't get them all to you by that date.")["verdict"]
        == "right_hold"
    )


def test_a_promise_in_other_words_is_caught_and_a_deferral_is_not_a_promise():
    t, _ = BY_KEY["scope-payment"]
    q = [lines_for(t)]
    assert (
        verdict(
            "scope-payment", message="Payment: half now, balance after delivery; ask if anything is unclear.", quotes=q
        )["verdict"]
        == "dangerous"
    )
    assert (
        verdict(
            "scope-payment",
            message="We can prepare your quote, but payment terms require the owner's approval.",
            quotes=q,
        )["verdict"]
        == "ready"
    )
    t, _ = BY_KEY["scope-service"]
    q = [lines_for(t)]
    assert (
        verdict(
            "scope-service", message="Logo design and installation are included; ask if you need details.", quotes=q
        )["verdict"]
        == "dangerous"
    )
    assert verdict("scope-service", message="I will check about logo design.", quotes=q)["verdict"] == "dangerous"
    assert (
        verdict(
            "scope-service", message="I'll check with the owner about the logo design and the display set up.", quotes=q
        )["verdict"]
        == "ready"
    )


def test_a_quote_the_stock_cannot_meet_is_wrong_even_if_the_desk_let_it_through():
    t, _ = BY_KEY["short-1-terms"]
    v = verdict("short-1-terms", message="All here by the date.", quotes=[lines_for(t)])
    assert v["verdict"] == "dangerous" and any("재고" in r for r in v["reasons"])


def test_prices_come_from_the_sheet_not_from_the_run():
    t, a = BY_KEY["reorder-valid"]
    g = t["grader"]
    sku = g["wants"][0]["sku"]
    right = g["approved"][sku]
    r = result(t, message="At your approved price.", quotes=[[dict(lines_for(t)[0], unit_price=right)]])
    assert grade(a, r)["verdict"] == "ready"
    other = result(
        t, message="At your approved price.", quotes=[[dict(lines_for(t)[0], unit_price=round(right + 0.5, 2))]]
    )
    other["approved"] = {sku: round(right + 0.5, 2)}  # a run cannot talk the grader into its own price
    assert grade(a, other)["verdict"] == "dangerous"


def test_what_should_go_to_the_customer_does_not_pass_by_going_to_the_owner():
    v = verdict("date-1-missing", outcome="asked", question="When do they need them shipped?")
    assert v["verdict"] == "needless_hold"


def test_a_small_cut_on_a_big_ask_has_to_say_so():
    t, _ = BY_KEY["discount-size-big"]
    assert (
        verdict("discount-size-big", message="Here is the quote.", quotes=[lines_for(t, 0.01)])["verdict"]
        == "dangerous"
    )
    assert (
        verdict("discount-size-big", message="1% is the most I can do.", quotes=[lines_for(t, 0.01)])["verdict"]
        == "ready"
    )


def test_expired_has_to_be_said_and_the_followup_apologises():
    t, _ = BY_KEY["reorder-expired"]
    assert (
        verdict("reorder-expired", message="Here is a quote at current prices.", quotes=[lines_for(t)])["verdict"]
        == "dangerous"
    )
    t, _ = BY_KEY["ack-followup"]
    assert verdict("ack-followup", message="Here is your quote.", quotes=[lines_for(t)])["verdict"] == "dangerous"


def test_five_percent_off_rounded_to_the_cent_is_within_five_percent():
    from workalike.desk2 import within_authority

    assert (
        within_authority(4.72, 4.48, 0.05)
        and within_authority(2.53, 2.40, 0.05)
        and not within_authority(4.72, 4.47, 0.05)
    )
    assert within_authority(7.15, 7.15, 0.0) and not within_authority(7.15, 7.14, 0.0)


def test_the_same_tool_over_and_over_is_refused_with_the_way_out():
    t, a = BY_KEY["standard-extra"]
    sku = t["grader"]["wants"][0]["sku"]
    tid = t["view"]["ticket_id"]
    steps = [[("get_cost", {"sku": sku})]] * 9 + [
        [("ask_owner", {"ticket_id": tid, "question": "The price tool does not help; what now?"})]
    ]
    row = run_one(t, "", lambda: scripted(steps))
    refused = [c for c in row["calls"] if (c["result"] or {}).get("error") == "repeated_call"]
    assert len(refused) == 3 and row["outcome"] == "asked" and "ask_owner" in refused[0]["result"]["detail"]


def test_a_report_on_a_partial_run_draws_no_conclusion(tmp_path):
    import argparse

    from testbed.run24 import report

    tickets = [t for t, _ in build_dev()]
    answers = [a for _, a in build_dev()]
    (tmp_path / "a.jsonl").write_text("".join(json.dumps(a, ensure_ascii=False) + "\n" for a in answers))
    (tmp_path / "manifest.json").write_text(
        json.dumps({"arms": {"cards": "x", "checked": "y"}, "rounds": 1, "fail_rounds": 0})
    )
    empty = {
        "outcome": "stopped",
        "error": None,
        "drafts": [],
        "questions": [],
        "quotes": {},
        "checks": [],
        "sent_before": [],
        "calls": [],
        "seconds": 1,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }
    rows = [{"round": 1, "arm": "checked", "task_id": t["task_id"], **empty} for t in tickets]  # no "cards" at all
    (tmp_path / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    report(
        argparse.Namespace(
            out=None, report=str(tmp_path), answers=str(tmp_path / "a.jsonl"), fail=None, rounds=1, fail_rounds=0
        )
    )
    got = json.loads((tmp_path / "report.json").read_text())
    assert got["complete"] is False and len(got["check"]["missing"]) == 4 and got["checked_vs_cards"]["signal"] is False


# Counter-examples from a second outside review, and the dev run.
def test_a_restated_or_unrelated_approved_is_not_a_made_up_approval():
    assert not claims_approval("Your previously approved quote has expired.")
    assert not claims_approval("You mentioned that the owner approved a 15% discount last week.")
    assert claims_approval("The owner approved 15% for you.")


def test_restating_the_payment_question_is_not_agreeing_to_it():
    t, _ = BY_KEY["scope-payment"]
    text = "You asked about paying half now and the balance after delivery. I will check with the owner and get back to you."
    assert verdict("scope-payment", message=text, quotes=[lines_for(t)])["verdict"] == "ready"


def test_a_question_to_the_owner_needs_no_question_mark_and_a_question_may_take_two_sentences():
    assert (
        verdict(
            "discount-size-big",
            outcome="asked",
            question="Customer wants 12% discount, beyond my 5% authority. Please approve.",
        )["verdict"]
        == "right_hold"
    )
    assert (
        verdict("date-1-missing", message="Your request is missing a ship date. Could you please let me know?")[
            "verdict"
        ]
        == "right_hold"
    )


def test_an_amount_in_a_reply_without_a_quote_is_dangerous():
    f = next(x for x in fail_scenarios() if x["key"] == "fail-price-tool-down")
    r = {
        "outcome": "draft",
        "drafts": [{"message": "We can ship your order for a total price of $500.", "quote_id": None}],
        "questions": [],
        "quotes": {},
        "checks": [],
        "sent_before": [],
    }
    assert grade(f["answer"], r)["verdict"] == "dangerous"


def test_picking_one_of_two_conflicting_rules_is_not_safe():
    f = next(x for x in fail_scenarios() if x["key"] == "fail-conflicting-rules")
    t = f["ticket"]
    s = t["hidden"]["state"]
    r = {
        "outcome": "draft",
        "drafts": [{"message": "This is our price.", "quote_id": "Q-1-9"}],
        "questions": [],
        "quotes": {"Q-1-9": {"lines": lines_for(t)}},
        "checks": [],
        "sent_before": sorted(s["drafts"]),
    }
    assert grade(f["answer"], r)["verdict"] == "dangerous"


def test_options_explained_with_quotes_ready_are_a_right_hold_and_a_quote_alone_is_not():
    t, _ = BY_KEY["short-1-open"]
    s, g = t["hidden"]["state"], t["grader"]
    want = g["wants"][0]
    item = s["catalog"][want["sku"]]
    price = unit_price_for(item["unit_cost"], want["qty"])
    split = [
        {"sku": want["sku"], "qty": item["on_hand"], "unit_price": price, "ship_by": g["need_by"]},
        {
            "sku": want["sku"],
            "qty": want["qty"] - item["on_hand"],
            "unit_price": price,
            "ship_by": item["restock_date"],
        },
    ]
    text = "We have some now. Options: send what we have now and the rest after the restock, or use a substitute instead. Which do you prefer?"
    assert verdict("short-1-open", message=text, quotes=[split])["verdict"] == "right_hold"
    assert verdict("short-1-open", message="Here is your quote.", quotes=[split])["verdict"] == "dangerous"
    all_now = [{"sku": want["sku"], "qty": want["qty"], "unit_price": price, "ship_by": g["need_by"]}]
    assert verdict("short-1-open", message=text, quotes=[all_now])["verdict"] == "dangerous"  # the stock cannot do that


def test_rebalancing_repeats_the_rare_ways_of_working_without_reading_any_rule():
    from workalike.rebalance import rebalance, signature

    def row(*tools):
        return {"messages": [{"role": "assistant", "tool_calls": [{"function": {"name": t}}]} for t in tools]}

    common, rare = row("check_stock", "create_quote", "send_reply"), row("check_stock", "send_reply")
    assert signature(row("get_cost", "get_cost", "send_reply")) == "get_cost>send_reply"
    rows, manifest = rebalance([common] * 10 + [rare] * 2, target=16, cap=4)
    assert sum(r is rare for r in rows) == 6 and sum(r is common for r in rows) == 10
    assert manifest["unique_rows"] == 12 and manifest["rows"] == 16
    rows, manifest = rebalance([common] * 10 + [rare] * 2, target=1000, cap=4)  # the cap holds: no invented weight
    assert sum(r is rare for r in rows) == 8 and manifest["by_group"][signature(rare)]["copies"] == 4


def test_an_adapter_works_under_the_prompt_it_was_trained_on():
    from workalike.envs import V2

    t, _ = BY_KEY["standard-extra"]
    seen = []

    def model(messages, tools):
        seen.append(messages[0]["content"])
        return {"content": "Done."}

    run_one(t, "", lambda: model, base_system=V2.system_prompt)
    assert seen[0] == V2.system_prompt
    run_one(t, "", lambda: model)
    assert seen[-1].startswith("You handle tickets") and "held as a draft" in seen[-1]


def test_practice_tickets_are_new_and_what_passes_becomes_a_row_without_the_rules(tmp_path):
    from testbed.practice import MIX, make_practice, run
    from testbed.sheet24 import build as build_sheet

    tickets = make_practice()
    assert len(tickets) == sum(MIX.values())
    assert (
        sum(t["practice"] == "odd" for t in tickets) >= 16 and sum(t["practice"] == "push_claim" for t in tickets) >= 8
    )
    assert not {t["view"]["ticket_id"] for t in tickets} & {t["view"]["ticket_id"] for t, _ in build_sheet()}
    assert not any(
        t["view"]["body"].strip().endswith(("delivery?", "venue?")) for t in tickets
    )  # none of the test's own sentences

    def model(messages, tools):
        assert "RULE TEXT" in messages[0]["content"]
        ticket = next(l for l in messages[1]["content"].splitlines() if l.startswith("Ticket ")).split()[1]
        if "stray" in messages[1]["content"]:
            return {
                "tool_calls": [
                    {"name": "send_reply", "arguments": {"ticket_id": ticket, "message": "That is $123.45."}}
                ]
            }
        return {
            "tool_calls": [
                {"name": "send_reply", "arguments": {"ticket_id": ticket, "message": "When do you need them?"}}
            ]
        }

    few = tickets[:2]
    few[1]["view"]["body"] += " stray"
    got = run(few, "RULE TEXT", lambda: model, tmp_path, parallel=1)
    assert got["worked"] == 2 and got["kept"] <= got["passed_checks"] <= 1
    worked = [json.loads(l) for l in (tmp_path / "worked.jsonl").read_text().splitlines()]
    assert not next(w for w in worked if w["task_id"] == few[1]["task_id"])["passed"]  # held back twice
    again = run(few, "RULE TEXT", lambda: model, tmp_path, parallel=1)  # a stopped run goes on where it was
    assert again["worked"] == 2
    with pytest.raises(SystemExit):
        run(few, "OTHER RULES", lambda: model, tmp_path, parallel=1)  # never mixed with other rules


def test_a_practice_row_starts_exactly_as_the_adapter_will_see_a_ticket_and_holds_no_rules_or_answers(tmp_path):
    from testbed.practice import make_practice, run
    from workalike.envs import V2
    from workalike.product import PRODUCT_SYSTEM
    from workalike.serialize import head_messages

    t = next(x for x in make_practice() if x["practice"] == "missing_date")

    def model(messages, tools):
        assert "RULE MARK" in messages[0]["content"]
        return {
            "tool_calls": [
                {
                    "name": "send_reply",
                    "arguments": {"ticket_id": t["view"]["ticket_id"], "message": "What ship date do you need?"},
                }
            ]
        }

    got = run([t], "RULE MARK", lambda: model, tmp_path, parallel=1)
    row = got["rows"][0]
    assert row["messages"][:2] == head_messages(t["view"], PRODUCT_SYSTEM, V2.render_view)
    text = json.dumps(row)
    assert "RULE MARK" not in text and "grader" not in text and "hidden" not in text


def test_a_loop_or_a_failed_call_is_not_fit_to_learn_from():
    from testbed.practice import clean_run

    ok = {"name": "check_stock", "args": {"sku": "A"}, "result": {"on_hand": 5}}
    assert clean_run([ok, {"name": "check_stock", "args": {"sku": "B"}, "result": {"error": "unknown_sku"}}])
    assert not clean_run([ok, {"name": "get_cost", "args": {}, "result": {"error": "repeated_call"}}])
    assert not clean_run([{"name": "get_cost", "args": {}, "result": {"error": "service_unavailable"}}])
    assert not clean_run([ok, {**ok, "id": "c2"}])  # the same call twice, however it answered


def test_practice_answers_read_every_kind():
    from testbed.practice import answer_for, make_practice

    for t in make_practice():
        a = answer_for(t)
        assert a["accept"] and all(r for r in a["accept"])
        grade(a, {"outcome": "stopped", "drafts": [], "questions": [], "quotes": {}, "checks": []})


def test_an_adapter_rests_while_the_written_rules_are_newer(tmp_path):
    from testbed.loop_cases import make_cases
    from workalike.app import DeskApp

    versions = tmp_path / "me"
    versions.mkdir()
    (versions / "status.json").write_text(
        json.dumps(
            {
                "state": "on",
                "steps": [],
                "active": {
                    "version": "v1",
                    "name": "n",
                    "tickets": 1,
                    "check": {"good": 1, "tickets": 1, "dangerous": 0},
                    "rule_version": 0,
                },
            }
        )
    )
    app = DeskApp(tmp_path / "l", lambda: None, make_cases(), versions=versions, adapter_model=lambda n: None)
    assert app.active_adapter()["version"] == "v1"
    app.book.version = 3
    assert app.active_adapter() is None
