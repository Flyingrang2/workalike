"""Rule cards: replies are read however they are wrapped, and evidence comes from record numbers, not the model."""
import json
import re

import pytest

from island.tasks2 import make_tasks
from workalike.envs import V2
from workalike.rules import CardError, parse_cards, propose


def test_cards_are_read_through_fences_and_prose():
    text = 'Here you go:\n```json\n{"cards": [{"situation": "stock is short", "action": "ask first", ' \
           '"evidence": [2, 9, true, "3"], "unknowns": "how long they wait"}]}\n```\nDone.'
    [card] = parse_cards(text, allowed={1, 2, 3})
    assert card == {"situation": "stock is short", "action": "ask first", "evidence": [2],
                    "unknowns": ["how long they wait"]}
    assert parse_cards("<think>{not this}</think>" + text.split(":", 1)[1])[0]["evidence"] == [2, 9]
    for bad in ("no json", '{"rules": []}', '{"cards": [{"situation": "", "action": "x"}]}', '{"cards": [}'):
        with pytest.raises(CardError):
            parse_cards(bad)


def records(n):
    tasks = make_tasks(11, "train")[:n]
    return [(f"e-{i}", {"view": t["view"], "turns": [], "final": "Done."}) for i, t in enumerate(tasks)]


def test_batches_are_merged_and_evidence_is_the_union_of_what_the_joined_cards_saw():
    seen = []

    def chat(messages):
        user = messages[1]["content"]
        seen.append(messages[0]["content"][:20])
        if user.startswith("Record"):
            first = int(re.match(r"Record (\d+)", user).group(1))
            return json.dumps({"cards": [{"situation": "short", "action": "ask", "evidence": [first, first + 1, 999],
                                          "unknowns": []}]})
        assert "Card 1: When short: ask" in user and "Card 2:" in user
        return json.dumps({"cards": [{"situation": "short", "action": "ask", "from": [1, 2, 7], "unknowns": ["u"]}]})
    out = propose(records(5), V2, chat, batch=3)
    [card] = out["cards"]
    assert card["evidence"] == [1, 2, 4, 5] and card["records"] == ["e-0", "e-1", "e-3", "e-4"]
    assert card["from"] == [1, 2] and len(out["batches"]) == 2 and len(seen) == 3


def test_an_unreadable_reply_is_asked_again_once_with_the_reason():
    replies = iter(["sorry, here are rules: none", '{"cards": [{"situation": "s", "action": "a", "evidence": [1]}]}'])
    asked = []

    def chat(messages):
        asked.append(messages)
        return next(replies)
    [card] = propose(records(1), V2, chat)["cards"]
    assert card["records"] == ["e-0"]
    assert asked[1][-1]["role"] == "user" and "no JSON object" in asked[1][-1]["content"]
    with pytest.raises(CardError):
        propose(records(1), V2, lambda m: "still nothing")


def test_a_failed_batch_is_reported_and_the_others_go_on():
    def chat(messages):
        user = messages[-1]["content"] if messages[-1]["role"] == "user" else ""
        if "Record 1:" in messages[1]["content"]:
            raise CardError("empty reply (finish reason length)")
        if messages[1]["content"].startswith("Record"):
            return '{"cards": [{"situation": "s", "action": "a", "evidence": [4]}]}'
        return '{"cards": [{"situation": "s", "action": "a", "from": [1]}]}'
    out = propose(records(5), V2, chat, batch=3)
    assert out["failed"] == [{"records": [1, 3], "error": "empty reply (finish reason length)"}]
    assert out["cards"][0]["records"] == ["e-3"]


def proposed():
    return [{"situation": "a reorder's quote has expired", "action": "quote today's price and say it changed",
             "evidence": [1, 2], "records": ["e-0", "e-1"], "unknowns": ["how far they go down"]},
            {"situation": "stock is short", "action": "ask the customer to choose", "evidence": [3],
             "records": ["e-2"], "unknowns": []}]


def test_only_confirmed_cards_reach_the_model_and_each_version_keeps_its_text(tmp_path):
    from workalike.rules import RuleBook
    book = RuleBook(tmp_path / "rules.json")
    assert book.add_proposed(proposed(), "c60") == ["R1", "R2"] and book.as_procedure() == "" and book.version == 0
    book.set_status("R1", "confirmed")
    book.set_status("R2", "ask")
    text = book.as_procedure()
    assert "R1. When a reorder's quote has expired" in text and "R2 (not settled" in text and book.version == 2
    book.add_proposed(proposed()[:1], "again")
    again = RuleBook(tmp_path / "rules.json")
    assert [c["id"] for c in again.cards] == ["R1", "R2", "R3"] and again.card("R3")["status"] == "unconfirmed"
    assert again.as_procedure() == text and again.text_at(1).count("\nR") == 1


def test_an_edit_waits_until_applied_and_then_is_marked_as_the_persons(tmp_path):
    from workalike.rules import RuleBook
    book = RuleBook(tmp_path / "rules.json")
    book.add_proposed(proposed(), "c60")
    book.set_status("R1", "confirmed")
    before, v = book.as_procedure(), book.version
    book.edit("R1", action={"en": "offer up to 5% off at once", "ko": "바로 5%까지 깎아 준다"})
    assert book.as_procedure() == before and book.version == v                # not applied yet
    assert "offer up to 5% off at once" in book.as_procedure(preview="R1")
    book.apply_edit("R1")
    card = book.card("R1")
    assert card["action"]["by"] == "person" and card["action"]["was"]["by"] == "records"
    assert card["situation"]["by"] == "records" and book.version == v + 1 and book.text_at(v) == before
    book.forget_records({"e-1", "e-9"})
    assert book.card("R1")["gone"] == ["e-1"] and book.card("R2")["gone"] == []


def test_a_dropped_card_never_reaches_the_model(tmp_path):
    from workalike.rules import RuleBook
    book = RuleBook(tmp_path / "rules.json")
    book.add_proposed(proposed(), "c60")
    book.set_status("R1", "confirmed")
    book.set_status("R2", "dropped")
    text = book.as_procedure()
    assert "R1." in text and "R2" not in text and book.card("R2")["status"] == "dropped"
