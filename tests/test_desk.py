from workalike.desk import Desk, empty_state, unit_price_for


def state():
    s = empty_state("2026-10-27")
    s["tickets"] = ["T-1001"]
    s["catalog"] = {
        "MUG-11W": {
            "name": "11 oz white ceramic mug",
            "unit_cost": 2.4,
            "on_hand": 500,
            "restock_date": None,
            "substitutes": ["MUG-15W"],
        },
        "MUG-15W": {
            "name": "15 oz white ceramic mug",
            "unit_cost": 2.9,
            "on_hand": 50,
            "restock_date": "2026-11-10",
            "substitutes": ["MUG-11W"],
        },
    }
    s["quotes"] = {
        "Q-1": {"valid_until": "2026-11-01", "lines": [{"sku": "MUG-11W", "name": "x", "qty": 10, "unit_price": 4.4}]},
        "Q-2": {"valid_until": "2026-10-01", "lines": []},
    }
    return s


def test_cost_and_unknown_sku_hint():
    d = Desk(state())
    assert d.call("get_cost", {"sku": "MUG-11W"})["unit_cost"] == 2.4
    err = d.call("get_cost", {"sku": "MUG11W"})
    assert err == {"error": "unknown_sku", "sku": "MUG11W", "did_you_mean": "MUG-11W"}


def test_stock_short_lists_restock_and_substitutes():
    d = Desk(state())
    r = d.call("check_stock", {"sku": "MUG-15W", "qty": 100})
    assert r["enough"] is False and r["restock_date"] == "2026-11-10"
    assert [s["sku"] for s in r["substitutes"]] == ["MUG-11W"]
    assert "restock_date" not in d.call("check_stock", {"sku": "MUG-11W", "qty": 100})


def test_price_rounds_up_and_meets_floor():
    assert unit_price_for(0.05, 1000) == 0.08
    r = Desk(state()).call("calc_price", {"sku": "MUG-11W", "qty": 50, "unit_cost": 2.4})
    assert r["unit_price"] == 4.32 and r["margin_pct"] > 30


def test_quote_rules():
    d = Desk(state())
    line = {"sku": "MUG-11W", "qty": 50, "unit_price": 3.0}
    assert (
        d.call("create_quote", {"ticket_id": "T-1001", "lines": [line], "ship_by": "2026-11-01"})["error"]
        == "below_margin_floor"
    )
    short = {"sku": "MUG-15W", "qty": 100, "unit_price": 5.0}
    assert (
        d.call("create_quote", {"ticket_id": "T-1001", "lines": [short], "ship_by": "2026-11-01"})["error"]
        == "stock_short_for_ship_date"
    )
    ok = d.call("create_quote", {"ticket_id": "T-1001", "lines": [short], "ship_by": "2026-11-10"})
    assert ok["quote_id"] == "Q-1001-1" and ok["total"] == 500.0
    assert d.call("send_reply", {"ticket_id": "T-1001", "message": "hi", "quote_id": "Q-9"})["error"] == "unknown_quote"
    assert (
        d.call("send_reply", {"ticket_id": "T-1001", "message": "hi", "quote_id": "Q-1001-1"})["status"]
        == "held_for_approval"
    )


def test_quote_status_follows_date():
    d = Desk(state())
    assert d.call("lookup_quote", {"quote_id": "Q-1"})["status"] == "approved"
    assert d.call("lookup_quote", {"quote_id": "Q-2"})["status"] == "expired"


def test_bad_arguments_are_results_not_crashes():
    d = Desk(state())
    assert d.call("check_stock", {"sku": "MUG-11W", "qty": "ten"})["error"] == "invalid_arguments"
    assert d.call("get_cost", {})["error"] == "invalid_arguments"
    assert d.call("refund", {})["error"] == "unknown_tool"


def test_desk_does_not_touch_caller_state():
    s = state()
    Desk(s).call(
        "create_quote",
        {"ticket_id": "T-1001", "lines": [{"sku": "MUG-11W", "qty": 5, "unit_price": 9.0}], "ship_by": "2026-11-01"},
    )
    assert s["drafts"] == {}


def test_quote_ids_are_not_case_sensitive():
    d = Desk(state())
    assert d.call("lookup_quote", {"quote_id": "q-1"})["quote_id"] == "Q-1"
