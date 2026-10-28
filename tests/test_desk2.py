"""quote-desk/2: stock by date, the floor price, notices, the owner, and the customer's fixed answers."""
from workalike import desk2
from workalike.desk import Desk
from workalike.envs import V1, V2
from workalike.sft import sft_row

T = "T-1001"


def state(**customer):
    s = desk2.empty_state("2026-10-01")
    s["tickets"].append(T)
    s["catalog"] = {
        "PST-2436": {"name": "24x36 poster", "unit_cost": 4.06, "on_hand": 56, "restock_date": "2026-10-04",
                     "restock_qty": 300, "substitutes": ["PST-1824"]},
        "PST-1824": {"name": "18x24 poster", "unit_cost": 2.62, "on_hand": 1034, "restock_date": None,
                     "restock_qty": 0, "substitutes": ["PST-2436"]},
    }
    s["authority"][T] = 0.05
    s["owner"][T] = {"says": "Fine to go down to the floor price on this one, your call."}
    s["customers"][T] = {
        "wants": [{"sku": "PST-2436", "qty": 80, "substitutes": ["PST-1824"]}], "need_by": "2026-10-02",
        "split_ok": True, "later_ok": False, "substitute_ok": False,
        "says": {"answer": "Send what you have now and the rest after the restock, that's fine.",
                 "decline": "We need at least part of it by Oct 2.",
                 "push_back": "That's more than we paid last time. Anything you can do?"},
        **customer,
    }
    return s


def line(sku, qty, price, ship):
    return {"sku": sku, "qty": qty, "unit_price": price, "ship_by": ship}


def quote(desk, *lines):
    return desk.call("create_quote", {"ticket_id": T, "lines": list(lines)})


def test_stock_is_counted_per_item_by_date():
    d = Desk(state(), env=V2)
    two_early = quote(d, line("PST-2436", 40, 7.31, "2026-10-02"), line("PST-2436", 40, 7.31, "2026-10-03"))
    assert two_early["error"] == "stock_short_for_ship_date" and two_early["available_by_then"] == 56
    split = quote(d, line("PST-2436", 40, 7.31, "2026-10-02"), line("PST-2436", 40, 7.31, "2026-10-04"))
    assert split["quote_id"] and [l["ship_by"] for l in split["lines"]] == ["2026-10-02", "2026-10-04"]
    assert quote(d, line("PST-2436", 400, 7.31, "2026-10-05"))["error"] == "stock_short_for_ship_date"
    three = quote(d, *[line("PST-2436", 10, 7.31, f"2026-10-0{i}") for i in (4, 5, 6)])
    assert three["error"] == "too_many_lines_for_item"


def test_mixing_items_on_one_date():
    d = Desk(state(), env=V2)
    q = quote(d, line("PST-2436", 56, 7.31, "2026-10-02"), line("PST-1824", 24, 4.72, "2026-10-02"))
    assert "error" not in q and q["total"] == round(56 * 7.31 + 24 * 4.72, 2)


def test_floor_price_and_notices():
    d = Desk(state(), env=V2)
    p = d.call("calc_price", {"sku": "PST-1824", "qty": 100, "unit_cost": 2.62})
    assert p["unit_price"] == 4.33 and p["floor_unit_price"] == 3.75
    assert quote(d, line("PST-1824", 100, 3.74, "2026-10-02"))["error"] == "below_margin_floor"
    within = quote(d, line("PST-1824", 100, 4.12, "2026-10-02"))
    assert "within your discount authority" in within["notices"][0]
    beyond = quote(d, line("PST-1824", 100, 3.75, "2026-10-02"))
    assert "owner approval needed" in beyond["notices"][0]
    assert "notices" not in quote(d, line("PST-1824", 100, 4.33, "2026-10-02"))


def test_owner_gives_the_fixed_policy():
    d = Desk(state(), env=V2)
    assert d.call("ask_owner", {"ticket_id": T, "question": "Can I take 10% off?"}) == {
        "owner_reply": "Fine to go down to the floor price on this one, your call."}
    assert d.call("ask_owner", {"ticket_id": "T-9", "question": "?"})["error"] == "unknown_ticket"


def test_customer_answers_a_question_then_repeats_itself():
    d = Desk(state(), env=V2)
    first = d.call("send_reply", {"ticket_id": T, "message": "Would a split shipment work for you?"})
    assert first["status"] == "sent" and first["customer_reply"].startswith("Send what you have now")
    again = d.call("send_reply", {"ticket_id": T, "message": "Just checking again."})
    assert again["customer_reply"].startswith("As I said:")


def test_customer_turns_down_a_quote_outside_the_hidden_conditions_then_accepts():
    d = Desk(state(), env=V2)
    late = quote(d, line("PST-2436", 80, 7.31, "2026-10-04"))  # all after the date, later_ok is False
    r = d.call("send_reply", {"ticket_id": T, "message": "Quote attached.", "quote_id": late["quote_id"]})
    assert r["customer_reply"] == "We need at least part of it by Oct 2." and "customer_accepted" not in r
    split = quote(d, line("PST-2436", 56, 7.31, "2026-10-02"), line("PST-2436", 24, 7.31, "2026-10-04"))
    r = d.call("send_reply", {"ticket_id": T, "message": "Here is the split.", "quote_id": split["quote_id"]})
    assert r["customer_accepted"] == split["quote_id"]
    after = d.call("send_reply", {"ticket_id": T, "message": "Thanks!"})
    assert split["quote_id"] in after["customer_reply"]


def test_a_quote_named_in_the_message_is_offered_too():
    d = Desk(state(), env=V2)
    late = quote(d, line("PST-2436", 80, 7.31, "2026-10-04"))
    split = quote(d, line("PST-2436", 56, 7.31, "2026-10-02"), line("PST-2436", 24, 7.31, "2026-10-04"))
    msg = f"Two options: {late['quote_id']} (attached) or {split['quote_id'].lower()} if you'd rather split it."
    r = d.call("send_reply", {"ticket_id": T, "message": msg, "quote_id": late["quote_id"]})
    assert r["customer_accepted"] == split["quote_id"]


def test_customer_pushes_back_once_on_price():
    d = Desk(state(push_back_over=500.0), env=V2)
    split = quote(d, line("PST-2436", 56, 7.31, "2026-10-02"), line("PST-2436", 24, 7.31, "2026-10-04"))
    msg = {"ticket_id": T, "message": "Quote attached.", "quote_id": split["quote_id"]}
    assert d.call("send_reply", msg)["customer_reply"].startswith("That's more than")
    assert d.call("send_reply", msg)["customer_accepted"] == split["quote_id"]


def test_unknown_sku_lists_similar_items():
    d = Desk(state(), env=V2)
    r = d.call("check_stock", {"sku": "PST-243", "qty": 5})
    assert r["error"] == "unknown_sku" and r["similar"][0]["sku"] == "PST-2436"


def test_same_calls_give_the_same_results():
    calls = [("check_stock", {"sku": "PST-2436", "qty": 80}),
             ("send_reply", {"ticket_id": T, "message": "Split ok?"}),
             ("create_quote", {"ticket_id": T, "lines": [line("PST-2436", 56, 7.31, "2026-10-02"),
                                                         line("PST-2436", 24, 7.31, "2026-10-04")]}),
             ("send_reply", {"ticket_id": T, "message": "Attached.", "quote_id": "Q-1001-1"})]
    a, b = Desk(state(), env=V2).call_many(calls), Desk(state(), env=V2).call_many(calls)
    assert a == b and a[-1]["result"]["customer_accepted"] == "Q-1001-1"


def test_versions_keep_their_own_tools_prompt_and_screen():
    s = state()
    assert Desk(s).call("ask_owner", {"ticket_id": T, "question": "?"})["error"] == "unknown_tool"
    view = {"today": "2026-10-01", "ticket_id": T, "from": "Mia Oh, Harbor Books", "body": "250 posters please.",
            "authority": 0.0, "sent_so_far": "an acknowledgement on 2026-09-30; no quote yet"}
    text = V2.render_view(view)
    assert "Ask the owner before offering any price under list" in text and "no quote yet" in text
    ep = {"view": view, "turns": [], "final": "Done."}
    v2, v1 = sft_row(ep, V2), sft_row({**ep, "view": {k: view[k] for k in ("today", "ticket_id", "from", "body")}})
    assert v2["messages"][0]["content"] == V2.system_prompt and v1["messages"][0]["content"] == V1.system_prompt
    assert {t["function"]["name"] for t in v2["tools"]} - {t["function"]["name"] for t in v1["tools"]} == {"ask_owner"}
