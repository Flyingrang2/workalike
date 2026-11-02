"""Rule cards: the unwritten rules a worker's records show, for the person to confirm.

    cards = propose([(ep_id, clean_episode), ...], V2, chat)   # chat(messages) -> text

Glimmer reads the records in numbered batches and writes candidate cards, and a last call merges them. A card
says when it applies, what the worker does, which records show it, and what the records alone cannot settle yet.
Which records back a card is worked out in code from record numbers (and, after the merge, from the numbers of
the batch cards it joins), so the model never copies an episode id and cannot invent one.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .record import wall_clock
from .serialize import canonical_json, order_args

CARD = ('{"cards": [{"situation": "...", "action": "...", "evidence": [record numbers], '
        '"unknowns": ["..."]}]}')
READ_CARDS = ("Below are numbered records of one worker at the quote desk of a small print shop: each ticket as the "
              "worker saw it, then every action they took and what came back. Find the rules this worker follows "
              "that are written nowhere: in which situation they do something particular, and what. For each rule "
              "give the records that show it and what these records cannot settle yet. Situations and actions are "
              "general, not about one customer or item. Answer with only this JSON object: " + CARD)
MERGE_CARDS = ("Below are rule cards about one worker at the quote desk of a small print shop, found in batches of "
               "their records. Merge them into one set with one card per situation: join the cards that state the "
               "same rule or cover the same situation (say how the worker acts there in one card), and keep the ones "
               "that differ. For each card keep at most three things the records leave unsettled, the ones that "
               "matter most for doing the work. Answer with only this JSON object: "
               '{"cards": [{"situation": "...", "action": "...", "from": [numbers of the cards it joins], '
               '"unknowns": ["..."]}]}')
AGAIN = "That could not be read as the JSON object asked for ({}). Answer again with only that JSON object."


class CardError(ValueError):
    pass


def render_record(ep: dict, env) -> str:
    """One clean episode as the worker left it: the screen, then every call and result in order."""
    lines = [env.render_view(ep["view"]), "Actions:"]
    for turn in ep["turns"]:
        for c in turn["calls"]:
            lines.append(f"- {c['name']} {canonical_json(order_args(c['name'], c['args'], env.spec_by_name))}"
                         f" -> {canonical_json(c['result'])}")
    if ep.get("final"):
        lines.append(f"Final: {ep['final']}")
    return "\n".join(lines)


def parse_cards(text: str, numbers: str = "evidence", allowed: set[int] | None = None) -> list[dict]:
    """The cards in a reply that may wrap its JSON in prose or fences. Numbers outside `allowed` are dropped."""
    text = (text or "").split("</think>")[-1]
    start = text.find("{")
    if start < 0:
        raise CardError("no JSON object")
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as e:
        raise CardError(f"invalid JSON: {e.msg} at {e.pos}") from None
    raw = obj.get("cards") if isinstance(obj, dict) else None
    if not isinstance(raw, list):
        raise CardError('no "cards" list')
    cards = []
    for c in raw:
        if not isinstance(c, dict) or not str(c.get("situation") or "").strip() or not str(c.get("action") or "").strip():
            continue
        nums = [n for n in c.get(numbers) or [] if isinstance(n, int) and not isinstance(n, bool)]
        unknowns = c.get("unknowns") or []
        unknowns = [unknowns] if isinstance(unknowns, str) else [str(u) for u in unknowns if str(u).strip()]
        cards.append({"situation": str(c["situation"]).strip(), "action": str(c["action"]).strip(),
                      numbers: sorted({n for n in nums if allowed is None or n in allowed}), "unknowns": unknowns})
    if not cards:
        raise CardError("no card with a situation and an action")
    return cards


def ask_cards(chat, system: str, user: str, **parse) -> list[dict]:
    """Ask once and, if the reply cannot be read, once more quoting why."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    text = chat(messages)
    try:
        return parse_cards(text, **parse)
    except CardError as e:
        messages += [{"role": "assistant", "content": text or ""}, {"role": "user", "content": AGAIN.format(e)}]
        return parse_cards(chat(messages), **parse)


def propose(records: list[tuple[str, dict]], env, chat, batch: int = 8, workers: int = 4) -> dict:
    """Candidate cards from (episode id, clean episode) pairs. Returns {"cards", "batches", "failed"}; every card
    carries the record numbers that show it ("evidence") and their episode ids ("records").

    Batches of 8: with 20 records Glimmer's thinking ran past 12000 tokens and no answer came back (10-29).
    A batch that still fails is reported in "failed" and the others go on; if every batch fails, it raises."""
    texts = [f"Record {i}:\n{render_record(ep, env)}" for i, (_, ep) in enumerate(records, 1)]
    spans = [range(s, min(s + batch, len(texts))) for s in range(0, len(texts), batch)]

    def read(span):
        try:
            return ask_cards(chat, READ_CARDS, "\n\n".join(texts[i] for i in span), allowed={i + 1 for i in span})
        except CardError as e:
            return e
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(read, spans))
    failed = [{"records": [span[0] + 1, span[-1] + 1], "error": str(r)}
              for span, r in zip(spans, results) if isinstance(r, CardError)]
    found = [r for r in results if not isinstance(r, CardError)]
    if not found:
        raise CardError("every batch failed: " + "; ".join(f["error"] for f in failed))
    return {"cards": merge(found, records, chat), "batches": found, "failed": failed}


def merge(found: list[list[dict]], records: list[tuple[str, dict]], chat) -> list[dict]:
    """One set of cards from the batches' cards; evidence is the union of what the joined cards saw."""
    flat = [c for cards in found for c in cards]
    if len(found) == 1:
        cards = [dict(c) for c in flat]
    else:
        listing = "\n".join(f"Card {k}: When {c['situation']}: {c['action']}"
                            + (f" Unsettled: {'; '.join(c['unknowns'])}" if c["unknowns"] else "")
                            for k, c in enumerate(flat, 1))
        merged = ask_cards(chat, MERGE_CARDS, listing, numbers="from", allowed=set(range(1, len(flat) + 1)))
        cards = [{"situation": m["situation"], "action": m["action"], "unknowns": m["unknowns"][:3],
                  "evidence": sorted({n for k in m["from"] for n in flat[k - 1]["evidence"]}), "from": m["from"]}
                 for m in merged]
    for c in cards:
        c["records"] = [records[n - 1][0] for n in c["evidence"]]
    return cards


STATES = {"unconfirmed": "확인 전", "confirmed": "확인됨", "ask": "더 물어봐야 함", "dropped": "뺌"}
PREAMBLE = ("Rules the person at this desk confirmed from their own work. Follow them for the choices they cover. "
            "If a choice comes up that no rule here covers, or information you need is missing, do not guess: "
            "ask the owner with ask_owner.")
FIELDS = ("situation", "action")


class RuleBook:
    """The cards a person keeps: proposed from records, then confirmed, edited or left to ask about.

    Each card field keeps who wrote it: "records" (found in the records) or "person" (decided by the person).
    The version goes up whenever what the model works with changes (a card confirmed, set to ask, or an edit
    applied); the procedure text of every version is kept, so a draft can say which rules it was made under.
    An edit waits as "pending" until the person applies it; nothing a model proposes replaces a confirmed card.
    """

    def __init__(self, path):
        self.path = Path(path)
        data = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.version = data.get("version", 0)
        self.cards: list[dict] = data.get("cards", [])
        self.texts: dict[str, str] = data.get("texts", {"0": ""})
        self.log: list[dict] = data.get("log", [])

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"version": self.version, "cards": self.cards, "texts": self.texts,
                                   "log": self.log}, ensure_ascii=False, indent=1) + "\n")
        tmp.replace(self.path)

    def card(self, cid: str) -> dict:
        return next(c for c in self.cards if c["id"] == cid)

    def _note(self, what: str, cid: str | None = None, **extra):
        self.log.append({"t": wall_clock(), "what": what, "card": cid, "version": self.version, **extra})

    def _bump(self):
        self.version += 1
        self.texts[str(self.version)] = self.as_procedure()

    def add_proposed(self, cards: list[dict], source: str, ko: dict | None = None) -> list[str]:
        """New cards from a proposal, numbered after the existing ones, all waiting for the person."""
        ids = []
        for c in cards:
            n = max((x["n"] for x in self.cards), default=0) + 1
            k = (ko or {}).get(len(ids), {})
            self.cards.append({
                "id": f"R{n}", "n": n, "status": "unconfirmed", "source": source, "pending": None,
                **{f: {"en": c[f], "ko": k.get(f), "by": "records"} for f in FIELDS},
                "unknowns": [{"en": u, "ko": (k.get("unknowns") or [None] * len(c["unknowns"]))[i]}
                             for i, u in enumerate(c["unknowns"])],
                "evidence": c.get("evidence", []), "records": c.get("records", []), "gone": []})
            ids.append(f"R{n}")
        self._note("proposed", None, cards=ids, source=source)
        self.save()
        return ids

    def set_status(self, cid: str, status: str):
        assert status in STATES
        card = self.card(cid)
        if card["status"] != status:
            card["status"] = status
            self._note(status, cid)
            self._bump()
            self.save()

    def edit(self, cid: str, **fields):
        """A person's change waits as pending; fields are {name: {"en", "ko"}} for situation or action."""
        card = self.card(cid)
        card["pending"] = {**(card["pending"] or {}), **{f: v for f, v in fields.items() if f in FIELDS}}
        self._note("edit pending", cid, fields=sorted(fields))
        self.save()

    def drop_edit(self, cid: str):
        self.card(cid)["pending"] = None
        self._note("edit dropped", cid)
        self.save()

    def apply_edit(self, cid: str):
        """The pending change becomes the card, written by the person, and the card counts as confirmed."""
        card = self.card(cid)
        if not card["pending"]:
            return
        for f, v in card.pop("pending").items():
            card[f] = {"en": v["en"], "ko": v.get("ko"), "by": "person", "was": card[f]}
        card["pending"] = None
        card["status"] = "confirmed"
        self._note("edit applied", cid)
        self._bump()
        self.save()

    def settle(self, cid: str, question: str, ko: str, en: str):
        """The person answers something the records left open. It is theirs, not found in the records."""
        card = self.card(cid)
        card.setdefault("settled", []).append({"question": question, "ko": ko, "en": en, "by": "person"})
        self._note("settled", cid, question=question)
        self._bump()
        self.save()

    def forget_records(self, eps: set[str]):
        """Records were deleted: cards that cited them say so (the cards stay; the person decides)."""
        for c in self.cards:
            gone = sorted(set(c["records"]) & eps)
            if gone:
                c["gone"] = sorted(set(c["gone"]) | set(gone))
        self.save()

    def as_procedure(self, preview: str | None = None) -> str:
        """The confirmed rules as the model reads them; with preview, as if that card's pending edit applied."""
        lines = []
        for c in self.cards:
            fields = {f: c[f]["en"] for f in FIELDS}
            status = c["status"]
            if c["id"] == preview and c["pending"]:
                fields.update({f: v["en"] for f, v in c["pending"].items()})
                status = "confirmed"
            settled = " ".join(f'On "{x["question"]}": {x["en"]}' for x in c.get("settled", []))
            settled = f" The person also decided. {settled}" if settled else ""
            if status == "confirmed":
                lines.append(f"{c['id']}. When {fields['situation']}: {fields['action']}{settled}")
            elif status == "ask":
                lines.append(f"{c['id']} (not settled: do not decide this yourself, ask the owner). "
                             f"When {fields['situation']}: {fields['action']}{settled}")
        return (PREAMBLE + "\n" + "\n".join(lines)) if lines else ""

    def text_at(self, version: int) -> str:
        return self.texts.get(str(version), "")
