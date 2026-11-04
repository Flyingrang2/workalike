"""The loop's fixed cases: their own tickets, and a push-back that follows the quote the shop already sent."""

import copy

from testbed.loop_cases import ORDER, make_cases
from workalike.desk import lookup_quote, unit_price_for
from workalike.person_desk import describe_en, describe_ko
from workalike.product import guard, work


def test_cases_are_fixed_and_use_their_own_tickets():
    a, b = make_cases(), make_cases()
    assert a == b and [t["case"] for t in a] == list(ORDER)
    assert all(400000 <= int(t["view"]["ticket_id"].split("-")[1]) <= 449999 for t in a)
    assert "pay half now" in a[ORDER.index("out_of_scope")]["view"]["body"]


def test_the_push_back_continues_from_the_sent_quote():
    t = make_cases()[ORDER.index("pushback")]
    tid, want, s = t["view"]["ticket_id"], t["grader"]["wants"][0], t["hidden"]["state"]
    sent = t["grader"]["sent_quote"]
    assert sent in t["view"]["thread"][-1]["text"] and "more than last time" in t["view"]["body"]
    off = min(t["grader"]["authority"], 0.03)
    price = round(unit_price_for(s["catalog"][want["sku"]]["unit_cost"], want["qty"]) * (1 - off), 2)
    calls = iter(
        [
            [
                (
                    "create_quote",
                    {
                        "ticket_id": tid,
                        "lines": [
                            {
                                "sku": want["sku"],
                                "qty": want["qty"],
                                "unit_price": price,
                                "ship_by": t["grader"]["need_by"],
                            }
                        ],
                    },
                )
            ],
            [
                (
                    "send_reply",
                    {
                        "ticket_id": tid,
                        "message": f"Instead of {sent}, here is a better price.",
                        "quote_id": f"Q-{tid.split('-')[1]}-2",
                    },
                )
            ],
        ]
    )

    def model(messages, tools):
        step = next(calls, None) if tools else None
        return {"tool_calls": [{"name": n, "arguments": a} for n, a in step]} if step else {"content": "Done."}

    job = work(t["view"], s, model)
    assert job["history"][0]["result"]["quote_id"] == f"Q-{tid.split('-')[1]}-2" and job["outcome"] == "draft"
    assert all(c["ok"] for c in guard(job["state"], tid, job["history"]))


def test_the_quote_the_shop_sent_can_be_looked_up_and_is_no_approved_price():
    t = make_cases()[ORDER.index("pushback")]
    s, want, sent = t["hidden"]["state"], t["grader"]["wants"][0], t["grader"]["sent_quote"]
    before = copy.deepcopy(s)
    found = lookup_quote(s, sent.lower())
    assert s == before and found["status"] == "issued" and found["valid_until"] is None
    assert [(l["sku"], l["qty"]) for l in found["lines"]] == [(want["sku"], want["qty"])] and found["lines"][0]["name"]
    assert lookup_quote(s, "Q-404")["error"] == "unknown_quote"
    assert "issued from this desk" in describe_en("lookup_quote", {}, found) and "만료" not in describe_ko(
        "lookup_quote", {}, found
    )


def test_a_quote_made_in_this_sitting_is_a_draft_and_looking_one_up_clears_nothing():
    t = make_cases()[ORDER.index("pushback")]
    tid, want, s = t["view"]["ticket_id"], t["grader"]["wants"][0], t["hidden"]["state"]
    listed = unit_price_for(s["catalog"][want["sku"]]["unit_cost"], want["qty"])
    new = f"Q-{tid.split('-')[1]}-2"

    def line(price):
        return {"sku": want["sku"], "qty": want["qty"], "unit_price": price, "ship_by": t["grader"]["need_by"]}

    steps = iter(
        [
            ("lookup_quote", {"quote_id": t["grader"]["sent_quote"]}),
            ("create_quote", {"ticket_id": tid, "lines": [line(round(listed * 0.5, 2))]}),
            ("create_quote", {"ticket_id": tid, "lines": [line(listed)]}),
            ("lookup_quote", {"quote_id": new}),
            ("send_reply", {"ticket_id": tid, "message": "Here it is.", "quote_id": new}),
        ]
    )

    def model(messages, tools):
        step = next(steps, None) if tools else None
        return {"tool_calls": [{"name": step[0], "arguments": step[1]}]} if step else {"content": "Done."}

    job = work(t["view"], s, model)
    sent, halved, made, again = [c["result"] for c in job["history"][:4]]
    assert sent["status"] == "issued" and made["status"] == "draft" and again["status"] == "draft"
    assert halved["error"] == "below_margin_floor"  # having looked up a quote buys no lower price
    assert "만료" not in describe_ko("lookup_quote", {}, again) and "not sent" in describe_en("lookup_quote", {}, again)
