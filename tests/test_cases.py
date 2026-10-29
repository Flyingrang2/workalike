"""The loop's fixed cases: their own tickets, and a push-back that follows the quote the shop already sent."""
from island.loop_cases import ORDER, make_cases
from workalike.desk import unit_price_for
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
    calls = iter([[("create_quote", {"ticket_id": tid, "lines": [{"sku": want["sku"], "qty": want["qty"],
                                                                     "unit_price": price, "ship_by": t["grader"]["need_by"]}]})],
                  [("send_reply", {"ticket_id": tid, "message": f"Instead of {sent}, here is a better price.",
                                   "quote_id": f"Q-{tid.split('-')[1]}-2"})]])

    def model(messages, tools):
        step = next(calls, None) if tools else None
        return {"tool_calls": [{"name": n, "arguments": a} for n, a in step]} if step else {"content": "Done."}
    job = work(t["view"], s, model)
    assert job["history"][0]["result"]["quote_id"] == f"Q-{tid.split('-')[1]}-2" and job["outcome"] == "draft"
    assert all(c["ok"] for c in guard(job["state"], tid, job["history"]))
