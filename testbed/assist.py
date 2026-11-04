"""Assist window: send a request, let the model work it at the desk, approve what comes back.

A request is a customer message. One worker thread hands requests to the agent loop
with the chosen model (the base model or one of a person's adapter versions), one at a
time; "every model" sends the same request to each, so their drafts can be compared. The
agent works on a private copy of the shop's state, so its quotes and replies are
only drafts. check() reads the drafts against the desk, and the page shows the
drafts, what the agent looked up and the check results. Nothing is released until
the owner presses Approve; approved replies go to sent.jsonl, where a mail or chat
connector would pick them up.

    python -m testbed.assist --shop data/shop.json --base-url http://localhost:8000/v1 \
        --model me-v1 --model muse-glimmer --out runs/assist
"""

from __future__ import annotations

import argparse
import copy
import html
import json
import queue
import re
import secrets
import threading
from pathlib import Path

from workalike import agent
from workalike.desk import Desk, cents, customer_prices, meets_floor
from workalike.model_client import ChatModel
from workalike.record import Recorder, wall_clock
from workalike.web import CSS, describe, make_server

MONEY = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)")
PRICE = re.compile(r"\$\s?\d|\b\d[\d,]*(?:\.\d+)?\s?(?:USD|dollars?)\b", re.IGNORECASE)  # as testbed/score.py
QUOTE_ID = re.compile(r"\bQ-[A-Z0-9]+(?:-\d+)?\b", re.IGNORECASE)


def _quote_problems(state: dict, draft: dict) -> list[str]:
    problems = []
    for line in draft["lines"]:
        item = state["catalog"][line["sku"]]
        if not meets_floor(line["unit_price"], item["unit_cost"]):
            problems.append(f"{line['sku']} is under the margin floor")
        if line["qty"] > item["on_hand"] and (item["restock_date"] is None or draft["ship_by"] < item["restock_date"]):
            problems.append(f"{line['sku']} is short for {draft['ship_by']}")
    return problems


def check(state: dict, ticket_id: str, history: list[dict], status: str) -> list[dict]:
    """Read the drafts of one ticket against the desk. Each check is {"name", "ok", "detail"}."""
    replies = [r for r in state["outbox"] if r["ticket_id"] == ticket_id]
    prices = customer_prices(c["result"] for c in history)
    stray = sorted({m for r in replies for m in MONEY.findall(r["message"]) if cents(m) not in prices})
    costs = {
        cents(c["result"]["unit_cost"])
        for c in history
        if c["name"] == "get_cost" and isinstance(c["result"], dict) and "unit_cost" in c["result"]
    }
    shown = sorted({m for r in replies for m in MONEY.findall(r["message"]) if cents(m) in costs - prices})
    # A price only binds on a quote the owner can approve; a price in plain text is a promise nobody checked.
    loose = sum(1 for r in replies if not r.get("quote_id") and PRICE.search(r["message"]))
    known = {q.upper() for q in [*state["quotes"], *state["drafts"]]}
    unknown = sorted({q for r in replies for q in QUOTE_ID.findall(r["message"]) if q.upper() not in known})
    checks = [
        {"name": "finished", "ok": status == "done", "detail": status},
        {"name": "reply drafted", "ok": bool(replies), "detail": f"{len(replies)} held for approval"},
        {
            "name": "prices go out on a quote",
            "ok": not loose,
            "detail": f"{loose} reply quotes a price with no quote attached" if loose else "no price outside a quote",
        },
        {
            "name": "costs stay inside",
            "ok": not shown,
            "detail": ("our cost shown: " + ", ".join(f"${s}" for s in shown)) if shown else "no unit cost in a reply",
        },
        {
            "name": "amounts come from the desk",
            "ok": not stray,
            "detail": ("not a desk price: " + ", ".join(f"${s}" for s in stray))
            if stray
            else "every amount is a desk price",
        },
        {
            "name": "quote ids exist",
            "ok": not unknown,
            "detail": ("unknown: " + ", ".join(unknown)) if unknown else "every quote id mentioned exists",
        },
    ]
    for qid in sorted({r["quote_id"] for r in replies if r.get("quote_id")}):
        problems = _quote_problems(state, state["drafts"][qid])
        checks.append(
            {
                "name": f"quote {qid} holds",
                "ok": not problems,
                "detail": "; ".join(problems) or "margin and stock hold at today's cost",
            }
        )
    return checks


ASSIST_CSS = """
main{display:grid;grid-template-columns:380px 1fr;gap:12px;padding:12px;height:auto;align-items:start}
.jobs{display:flex;flex-direction:column;gap:12px}.req{white-space:pre-wrap;margin:4px 0 8px}
.draft{background:#f6f8fa;border:1px solid #d8dee4;border-radius:4px;padding:8px;white-space:pre-wrap;margin:6px 0}
table{border-collapse:collapse;margin:6px 0}td,th{border:1px solid #d8dee4;padding:3px 8px;text-align:left;font-size:14px}
.ok{color:#1a7f37}.bad{color:#a40000}.badge{font-size:13px;padding:1px 8px;border-radius:10px;background:#dde3ea;margin-left:8px}
.waiting{background:#fff1c2}.approved{background:#d2f4dc}.rejected,.failed{background:#ffd8d8}
details{margin-top:6px;font-size:14px}details p{margin:3px 0}.reject{background:#fff;color:#a40000;border-color:#a40000}
textarea.ask{height:180px}input.wide{width:100%}.checks p{margin:2px 0}
"""


class Assist:
    """Requests in, drafts out, nothing released without the owner's approval."""

    def __init__(self, shop: dict, models: dict, out, max_turns: int = 24):
        """models maps a served name to a model function; the first one is the default."""
        self.shop = shop
        self.models = dict(models)
        self.default = next(iter(self.models))
        self.out = Path(out)
        self.recorder = Recorder(self.out / "records.jsonl")
        self.token = secrets.token_urlsafe(16)
        self.max_turns = max_turns
        self.lock = threading.Lock()
        self.jobs: list[dict] = []
        self.todo: queue.Queue = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()

    def ask(self, sender: str, body: str, model: str | None = None) -> dict:
        with self.lock:
            n = len(self.jobs) + 1
            view = {
                "today": self.shop["today"],
                "ticket_id": f"T-{1000 + n}",
                "from": sender.strip() or "Customer",
                "body": body.strip(),
                "thread": [],
            }
            job = {
                "id": n,
                "model": model if model in self.models else self.default,
                "view": view,
                "status": "working",
                "replies": [],
                "quotes": {},
                "calls": [],
                "checks": [],
            }
            self.jobs.append(job)
        self.todo.put(job)
        return job

    def _worker(self) -> None:
        while True:
            job = self.todo.get()
            self._work(job)
            self.todo.task_done()

    def _work(self, job: dict) -> None:
        view = job["view"]
        state = {**copy.deepcopy(self.shop), "tickets": [view["ticket_id"]], "drafts": {}, "outbox": []}
        desk = Desk(state, self.recorder)
        self.recorder.start(f"a-{secrets.token_hex(5)}", view, actor=f"model:{job['model']}", entrance="assist")
        try:
            _, status = agent.run(desk, view, self.models[job["model"]], self.max_turns)
        except Exception as e:  # keep the window alive; the job shows what went wrong
            status = f"error: {type(e).__name__}: {e}"
        self.recorder.end(status)
        replies = [r for r in desk.state["outbox"] if r["ticket_id"] == view["ticket_id"]]
        with self.lock:
            job.update(
                status="waiting" if replies else "failed",
                run_status=status,
                replies=replies,
                quotes={r["quote_id"]: desk.state["drafts"][r["quote_id"]] for r in replies if r.get("quote_id")},
                calls=desk.history,
                checks=check(desk.state, view["ticket_id"], desk.history, status),
            )

    def decide(self, job_id: int, approve: bool) -> None:
        with self.lock:
            job = next((j for j in self.jobs if j["id"] == job_id), None)
            if job is None or job["status"] != "waiting":
                return
            job["status"] = "approved" if approve else "rejected"
            entry = {
                "t": wall_clock(),
                "ticket_id": job["view"]["ticket_id"],
                "model": job["model"],
                "decision": job["status"],
                "replies": job["replies"],
                "quotes": job["quotes"],
                "checks": job["checks"],
            }
            self.out.mkdir(parents=True, exist_ok=True)
            with (self.out / "decisions.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            if approve:
                with (self.out / "sent.jsonl").open("a", encoding="utf-8") as f:
                    for r in job["replies"]:
                        f.write(
                            json.dumps({**r, "quote": job["quotes"].get(r.get("quote_id"))}, ensure_ascii=False) + "\n"
                        )

    def post(self, path: str, form: dict[str, str]) -> None:
        if path == "/ask" and form.get("body", "").strip():
            chosen = list(self.models) if form.get("model") == "*" else [form.get("model")]
            for name in chosen:
                self.ask(form.get("from", ""), form["body"], name)
        elif path in ("/approve", "/reject") and form.get("job", "").isdigit():
            self.decide(int(form["job"]), path == "/approve")

    def status(self) -> dict:
        with self.lock:
            return {
                "models": list(self.models),
                "jobs": [
                    {
                        "id": j["id"],
                        "model": j["model"],
                        "ticket_id": j["view"]["ticket_id"],
                        "status": j["status"],
                        "checks_ok": all(c["ok"] for c in j["checks"]) if j["checks"] else None,
                    }
                    for j in self.jobs
                ],
            }

    def _card(self, job: dict) -> str:
        e = html.escape
        v = job["view"]
        token = f"<input type='hidden' name='csrf' value='{self.token}'><input type='hidden' name='job' value='{job['id']}'>"
        parts = [
            f"<h2>{e(v['ticket_id'])} from {e(v['from'])}<span class='badge {job['status']}'>{job['status']}</span>"
            f"<span class='badge'>drafted by {e(job['model'])}</span></h2>",
            f"<div class='req'>{e(v['body'])}</div>",
        ]
        if job["status"] == "working":
            parts.append("<p><small>The model is working on this request.</small></p>")
        for r in job["replies"]:
            parts.append(f"<b>Draft reply</b><div class='draft'>{e(r['message'])}</div>")
            quote = job["quotes"].get(r.get("quote_id"))
            if quote:
                rows = "".join(
                    f"<tr><td>{e(l['sku'])}</td><td>{l['qty']:,}</td><td>${l['unit_price']:,.2f}</td>"
                    f"<td>${l['line_total']:,.2f}</td></tr>"
                    for l in quote["lines"]
                )
                parts.append(
                    f"<b>Quote {e(r['quote_id'])}</b>, ship by {e(quote['ship_by'])}<table>"
                    f"<tr><th>SKU</th><th>Qty</th><th>Unit price</th><th>Line total</th></tr>{rows}</table>"
                )
        if job["checks"]:
            items = "".join(
                f"<p class='{'ok' if c['ok'] else 'bad'}'>{'&#10003;' if c['ok'] else '&#10007;'} "
                f"{e(c['name'])}: {e(c['detail'])}</p>"
                for c in job["checks"]
            )
            parts.append(f"<b>Checks</b><div class='checks'>{items}</div>")
        if job["calls"]:
            steps = "".join(
                f"<p>{i}. {e(describe(c['name'], c['args'], c['result']))}</p>" for i, c in enumerate(job["calls"], 1)
            )
            parts.append(f"<details><summary>What it looked up ({len(job['calls'])} steps)</summary>{steps}</details>")
        if job["status"] == "waiting":
            all_ok = all(c["ok"] for c in job["checks"])
            parts.append(
                f"<form method='post' class='row'>{token}"
                f"<button class='finish' formaction='/approve'>{'Approve and send' if all_ok else 'Approve anyway'}</button>"
                f"<button class='reject' formaction='/reject'>Reject</button></form>"
            )
        return f"<div class='card'>{''.join(parts)}</div>"

    def page(self) -> str:
        with self.lock:
            jobs = [copy.deepcopy(j) for j in reversed(self.jobs)]
        refresh = "<meta http-equiv='refresh' content='3'>" if any(j["status"] == "working" for j in jobs) else ""
        e = html.escape
        choices = "".join(f"<option value='{e(m)}'>{e(m)}</option>" for m in self.models)
        if len(self.models) > 1:
            choices += "<option value='*'>every model (compare)</option>"
        ask = (
            f"<form method='post' action='/ask' class='card'><input type='hidden' name='csrf' value='{self.token}'>"
            "<h2>New request</h2><label>From <input name='from' class='wide' autocomplete='off'></label>"
            "<label style='display:block'>Message<textarea name='body' class='ask'></textarea></label>"
            f"<label>Draft with <select name='model'>{choices}</select></label> <button>Ask</button><p><small>Replies and quotes come back as drafts. "
            "Nothing reaches the customer until you approve it.</small></p></form>"
        )
        cards = "".join(self._card(j) for j in jobs) or "<div class='card'><small>No requests yet.</small></div>"
        return (
            f"<!doctype html><html><head><meta charset='utf-8'>{refresh}<title>Assist</title>"
            f"<style>{CSS}{ASSIST_CSS}</style></head><body><header><b>Assist</b>"
            f"<span>Today {html.escape(self.shop['today'])}</span>"
            f"</header><main><div class='col'>{ask}</div><div class='jobs'>{cards}</div></main></body></html>"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shop", required=True, help="shop state JSON: today, catalog, quotes")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument(
        "--model",
        action="append",
        help="served name of the base model or an adapter version; "
        "repeat for several (the first is the default; default muse-glimmer)",
    )
    ap.add_argument("--out", default="runs/assist")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--max-tokens", type=int, default=8192)
    args = ap.parse_args()
    params = {"max_tokens": args.max_tokens}
    if args.temperature is not None:
        params["temperature"] = args.temperature
    shop = json.loads(Path(args.shop).read_text())
    names = args.model or ["muse-glimmer"]
    app = Assist(shop, {n: ChatModel(args.base_url, n, **params) for n in names}, args.out)
    server = make_server(app, port=args.port)
    print(f"http://127.0.0.1:{args.port}  drafting with {', '.join(names)}, Ctrl+C to stop", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
