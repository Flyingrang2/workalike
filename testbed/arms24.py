"""What the model is given in each arm of the main test, written once and hashed, before any run.

    python -m testbed.arms24 --records data/v2/records/c-train.jsonl --tasks data/v2/tasks/train.jsonl \
        --cards runs/v2-cards3/c60-s1.glimmer.merge2.json [--answers <person.json> --table <their words in English>] \
        --out baselines/test24/arms

  none      the shop's guidance only (the floor)
  records   20 of the 60 records the cards were read from, the same number from each kind of ticket as far as it goes
  cards     Glimmer's cards as proposed, confirmed for this test only
  checked   the same cards after the person confirmed, changed, dropped or left them open
The two record and card arms carry the same instruction to ask when nothing covers a choice, so a difference between
them is the summary, not that sentence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
from pathlib import Path

from workalike.envs import V2
from workalike.filter import clean_episode, episode_meta
from workalike.record import group_episodes, read_events
from workalike.rules import RuleBook, render_record
from workalike.sft import read_jsonl

RECORDS_PREAMBLE = (
    "Records of how the person at this desk handled earlier tickets. Follow their way for the choices "
    "these cover. If a choice comes up that these do not cover, or information you need is missing, do "
    "not guess: ask the owner with ask_owner."
)


def pick_records(records_path, tasks_path, shuffle="11:curve:1", pool=60, n=20) -> list[tuple[str, dict]]:
    """From the same 60 the cards read, in the same order: take in turn from each ticket family until there are n."""
    family = {t["task_id"]: t["family"] for t in read_jsonl(tasks_path)}
    eps = [(episode_meta(ev), clean_episode(ev)) for ev in group_episodes(read_events(records_path)).values()]
    random.Random(shuffle).shuffle(eps)
    groups: dict[str, list] = {}
    for meta, ep in eps[:pool]:
        groups.setdefault(family[meta["task_id"]], []).append((meta["ep"], ep))
    picked = []
    while len(picked) < n:
        for name in sorted(groups):
            if groups[name] and len(picked) < n:
                picked.append(groups[name].pop(0))
    return picked


def records_text(picked: list[tuple[str, dict]]) -> str:
    return (
        RECORDS_PREAMBLE
        + "\n\n"
        + "\n\n".join(f"Ticket {i}:\n{render_record(ep, V2)}" for i, (_, ep) in enumerate(picked, 1))
    )


def book_of(cards: list[dict], path) -> RuleBook:
    book = RuleBook(path)
    book.add_proposed(cards, "test24")
    return book


def cards_text(cards: list[dict]) -> str:
    with tempfile.TemporaryDirectory() as d:
        book = book_of(cards, Path(d) / "b.json")
        for c in book.cards:
            c["status"] = "confirmed"
        return book.as_procedure()


def checked_book(cards: list[dict], answers: dict, table: dict, path) -> RuleBook:
    """The same cards with the person's answers on the card sheet. A choice left out stays open (ask); what the person
    wrote where the records left something open is kept apart as theirs. table holds their words in English."""
    book = book_of(cards, path)
    words = {(a["card"], a["i"]): a for a in table["answers"]}
    for c in book.cards:
        a = answers["cards"].get(c["id"]) or {}
        book.set_status(
            c["id"],
            {"drop": "dropped", "ask": "ask", "ok": "confirmed", "fix": "confirmed"}.get(a.get("choice"), "ask"),
        )
        if a.get("choice") == "fix" and (a.get("text") or "").strip():
            book.edit(c["id"], action={"en": words[(c["id"], "action")]["en"], "ko": a["text"]})
            book.apply_edit(c["id"])
        for i, text in enumerate(a.get("unknown") or []):
            if text and text.strip():
                w = words[(c["id"], i)]
                if w["ko"].strip() != text.strip():
                    raise ValueError(f"{c['id']}/{i}: the table does not hold what the person wrote")
                book.settle(c["id"], c["unknowns"][i]["en"], text, w["en"])
    return book


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--cards", required=True)
    ap.add_argument("--answers", help="the person's answers on the card sheet (JSON)")
    ap.add_argument("--table", help="their words in English, line for line (baselines/v2/c60-cards.user-answers.json)")
    ap.add_argument("--out", default="baselines/test24/arms")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cards = json.loads(Path(args.cards).read_text())["cards"]
    picked = pick_records(args.records, args.tasks)
    texts = {"none": "", "records": records_text(picked), "cards": cards_text(cards)}
    if args.answers:
        answers = json.loads(Path(args.answers).read_text())
        table = json.loads(Path(args.table).read_text())
        (out / "checked-book.json").unlink(missing_ok=True)
        texts["checked"] = checked_book(cards, answers, table, out / "checked-book.json").as_procedure()
    hashes = {}
    for name, text in texts.items():
        (out / f"{name}.txt").write_text(text)
        hashes[name] = hashlib.sha256(text.encode()).hexdigest()
    (out / "records-picked.json").write_text(json.dumps([ep for ep, _ in picked]) + "\n")
    (out / "hashes.json").write_text(json.dumps(hashes, indent=1) + "\n")
    print({k: (v[:12], len(texts[k])) for k, v in hashes.items()})


if __name__ == "__main__":
    main()
