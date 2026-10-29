"""Web entrance to the quote desk: the same six tools as forms on one page.

A person, or a model that only sees screenshots, works tickets in a browser.
Every form submission becomes exactly one Desk call, recorded like a call from
the tool entrance, so tasks, records and the scorer are shared by both entrances.
Results are shown as plain sentences in an activity log, newest first.
The page is laid out to fit a 1280x800 viewport without scrolling.
"""
from __future__ import annotations

import html
import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

from .desk import Desk
from .record import Recorder

FINAL = "Done."


def _money(v) -> str:
    if not isinstance(v, (int, float)):
        return str(v)
    return f"${v:,.2f}" if v >= 1 or round(v, 2) == v else f"${v}"


def describe(name: str, args: dict, r: dict) -> str:
    """A result in the words a person at the desk would read."""
    if "error" in r:
        hint = f" Did you mean {r['did_you_mean']}?" if r.get("did_you_mean") else ""
        detail = {k: v for k, v in r.items() if k not in ("error", "did_you_mean")}
        extra = f" ({', '.join(f'{k} {v}' for k, v in detail.items())})" if detail else ""
        return f"{name} failed: {r['error']}{extra}.{hint}"
    if name == "lookup_quote":
        lines = "; ".join(f"{l['qty']} x {l['sku']} ({l['name']}) at {_money(l['unit_price'])}" for l in r["lines"])
        return f"Quote {r['quote_id']} is {r['status']}, valid until {r['valid_until']}. Lines: {lines}."
    if name == "get_cost":
        return f"{r['sku']} ({r['name']}): unit cost {_money(r['unit_cost'])} as of {r['as_of']}."
    if name == "check_stock":
        text = f"{r['sku']} ({r['name']}), {r['qty']:,} needed: {r['on_hand']:,} on hand, "
        if r["enough"]:
            return text + "enough stock."
        subs = ", ".join(f"{s['sku']} ({s['name']}, {s['on_hand']:,} on hand)" for s in r["substitutes"]) or "none"
        return text + f"short. Restock on {r['restock_date']}. In-stock substitutes: {subs}."
    if name == "calc_price":
        return (f"{r['sku']} x {r['qty']:,} at cost {_money(r['unit_cost'])}: unit price {_money(r['unit_price'])}, "
                f"line total {_money(r['line_total'])}, margin {r['margin_pct']}%.")
    if name == "create_quote":
        lines = "; ".join(f"{l['qty']:,} x {l['sku']} at {_money(l['unit_price'])}" for l in r["lines"])
        return f"Draft quote {r['quote_id']} created: {lines}. Total {_money(r['total'])}, ship by {r['ship_by']}."
    if name == "send_reply":
        attached = f" with quote {args['quote_id']}" if args.get("quote_id") else ""
        return f"Reply #{r['reply_no']}{attached} is held for the owner's approval."
    return json.dumps(r)


def _num(text: str, kind):
    cleaned = text.replace("$", "").replace(",", "").strip()
    try:
        return kind(cleaned)
    except ValueError:
        return text


def form_to_call(form: dict[str, str], ticket_id: str) -> tuple[str, dict]:
    """One submitted form to (tool, args). Empty fields are left out, as an agent would."""
    tool = form.get("tool", "")
    f = {k: v.strip() for k, v in form.items()}
    if tool == "lookup_quote":
        return tool, {"quote_id": f.get("quote_id", "")}
    if tool == "get_cost":
        return tool, {"sku": f.get("sku", "")}
    if tool == "check_stock":
        return tool, {"sku": f.get("sku", ""), "qty": _num(f.get("qty", ""), int)}
    if tool == "calc_price":
        return tool, {"sku": f.get("sku", ""), "qty": _num(f.get("qty", ""), int),
                      "unit_cost": _num(f.get("unit_cost", ""), float)}
    if tool == "create_quote":
        lines = [{"sku": f[f"sku{i}"], "qty": _num(f.get(f"qty{i}", ""), int),
                  "unit_price": _num(f.get(f"price{i}", ""), float)}
                 for i in (1, 2, 3) if f.get(f"sku{i}")]
        return tool, {"ticket_id": ticket_id, "lines": lines, "ship_by": f.get("ship_by", "")}
    if tool == "send_reply":
        args = {"ticket_id": ticket_id, "message": form.get("message", "")}
        if f.get("quote_id"):
            args["quote_id"] = f["quote_id"]
        return tool, args
    return tool, {}


CSS = """
*{box-sizing:border-box}body{margin:0;font:15px/1.35 system-ui,Arial,sans-serif;color:#111;background:#eef0f3}
header{height:48px;display:flex;align-items:center;gap:24px;padding:0 16px;background:#1f2a37;color:#fff}
header b{font-size:17px}header form{margin-left:auto}
main{display:grid;grid-template-columns:430px 1fr;gap:12px;padding:12px;height:752px}
.col{display:flex;flex-direction:column;gap:12px;min-height:0}
.card{background:#fff;border:1px solid #cfd5dd;border-radius:6px;padding:10px 12px}
h2{font-size:14px;margin:0 0 6px;text-transform:uppercase;letter-spacing:.04em;color:#44505e}
.ticket{flex:0 0 420px;overflow:auto;white-space:pre-wrap}.ticket p{margin:4px 0;font-size:14px}.log{flex:1;overflow:auto}
.log p{margin:0 0 8px;padding-bottom:6px;border-bottom:1px solid #eee}.log small{color:#5a6573}
.err{color:#a40000}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.wide{grid-column:1/3}
label{display:inline-block;margin:2px 10px 2px 0}input,textarea{font:inherit;padding:4px 6px;border:1px solid #9aa5b1;border-radius:4px}
input{width:120px}input.short{width:80px}textarea{width:100%;height:64px}
button{font:inherit;padding:5px 12px;border-radius:4px;border:1px solid #1f2a37;background:#2f4a6d;color:#fff;cursor:pointer}
.finish{background:#1a7f37;border-color:#1a7f37}.row{display:flex;gap:8px;margin:3px 0}
"""


class WebDesk:
    """Serves one ticket at a time from a queue; Finish moves to the next."""

    def __init__(self, sessions: list[dict], recorder: Recorder):
        self.queue = list(sessions)
        self.recorder = recorder
        self.token = secrets.token_urlsafe(16)  # forms carry it, so other sites cannot post into the record
        self.lock = threading.Lock()
        self.current, self.desk, self.log, self.finished = None, None, [], []
        self._next()

    def _next(self) -> None:
        self.current = self.queue.pop(0) if self.queue else None
        self.log = []
        if self.current:
            s = self.current
            self.recorder.start(s["episode"], s["view"], task_id=s.get("task_id"), actor=s.get("actor"),
                                entrance="web")
            self.desk = Desk(s["state"], self.recorder)

    def act(self, form: dict[str, str]) -> None:
        with self.lock:
            if not self.current:
                return
            name, args = form_to_call(form, self.current["view"]["ticket_id"])
            done = self.desk.call_many([(name, args)])[0]
            self.log.insert(0, done)

    def finish(self, status: str = "done", final: str | None = FINAL) -> None:
        with self.lock:
            if not self.current:
                return
            if final is not None:
                self.recorder.final(final)
            self.recorder.end(status)
            self.finished.append(self.current.get("task_id"))
            self._next()

    def post(self, path: str, form: dict[str, str]) -> None:
        if path == "/act":
            self.act(form)
        elif path == "/finish":
            self.finish()

    def status(self) -> dict:
        return {"task_id": self.current.get("task_id") if self.current else None,
                "episode": self.current["episode"] if self.current else None, "finished": self.finished}

    def page(self) -> str:
        e = html.escape
        if not self.current:
            return f"<!doctype html><title>Quote desk</title><style>{CSS}</style><header><b>Quote desk</b></header>" \
                   "<main><div class='card'>No more tickets.</div></main>"
        v = self.current["view"]
        thread = "".join(f"<p><small>{e(m['date'])}, {e(m['who'])}:</small><br>{e(m['text'])}</p>"
                         for m in v.get("thread", []))
        ticket = (f"<b>From:</b> {e(v['from'])}\n<h2 style='margin-top:8px'>New message</h2>{e(v['body'])}"
                  + (f"<h2 style='margin-top:10px'>Earlier in this thread</h2>{thread}" if thread else ""))
        log = "".join(
            f"<p class='{'err' if 'error' in c['result'] else ''}'><small>{e(c['name'])} "
            f"{e(json.dumps(c['args'], ensure_ascii=False))}</small><br>{e(describe(c['name'], c['args'], c['result']))}</p>"
            for c in self.log) or "<p><small>Nothing yet.</small></p>"

        def inp(name, label, cls=""):
            return f"<label>{label} <input name='{name}' class='{cls}' autocomplete='off'></label>"

        token = f"<input type='hidden' name='csrf' value='{self.token}'>"

        def form(tool_buttons, body, cls="card"):
            buttons = "".join(f"<button name='tool' value='{t}'>{lab}</button> " for t, lab in tool_buttons)
            return f"<form method='post' action='/act' class='{cls}'>{token}{body}<div>{buttons}</div></form>"

        quote_rows = "".join(f"<div class='row'>{inp(f'sku{i}', f'SKU {i}')}{inp(f'qty{i}', 'Qty', 'short')}"
                             f"{inp(f'price{i}', 'Unit price', 'short')}</div>" for i in (1, 2, 3))
        forms = (
            form([("lookup_quote", "Look up quote")], "<h2>Earlier quote</h2>" + inp("quote_id", "Quote id"))
            + form([("get_cost", "Get cost"), ("check_stock", "Check stock")],
                   "<h2>Item</h2>" + inp("sku", "SKU") + inp("qty", "Qty", "short"))
            + form([("calc_price", "Calculate price")],
                   "<h2>Price a line</h2>" + inp("sku", "SKU") + inp("qty", "Qty", "short") + inp("unit_cost", "Unit cost", "short"),
                   "card wide")
            + form([("create_quote", "Create quote")],
                   "<h2>Create quote</h2>" + quote_rows + inp("ship_by", "Ship by (YYYY-MM-DD)"), "card wide")
            + form([("send_reply", "Send reply")],
                   "<h2>Reply to customer</h2><textarea name='message'></textarea>" + inp("quote_id", "Attach quote id"),
                   "card wide")
        )
        return (f"<!doctype html><html><head><meta charset='utf-8'><title>Quote desk</title><style>{CSS}</style></head>"
                f"<body><header><b>Quote desk</b><span>Ticket {e(v['ticket_id'])}</span><span>Today {e(v['today'])}</span>"
                f"<form method='post' action='/finish'>{token}<button class='finish'>Finish ticket</button></form></header>"
                f"<main><div class='col'><div class='card ticket'><h2>Ticket</h2>{ticket}</div>"
                f"<div class='card log'><h2>Activity (newest first)</h2>{log}</div></div>"
                f"<div class='grid'>{forms}</div></main></body></html>")


def make_server(webdesk, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    """Serve any page object with .token, .page(), .status() and .post(path, form), local requests only.
    Optional .page_for(path) serves pages by path, and .after_post(path) says where a post returns to."""
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="text/html; charset=utf-8"):
            data = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _local(self) -> bool:
            """Only this server's own address, which blocks DNS rebinding and cross-site posts."""
            port = self.server.server_address[1]
            hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
            origin = self.headers.get("Origin")
            return self.headers.get("Host") in hosts and (origin is None or origin in {f"http://{h}" for h in hosts})

        def do_GET(self):
            if not self._local():
                return self._send(403, "forbidden")
            if self.path == "/status":
                return self._send(200, json.dumps(webdesk.status()), "application/json")
            page_for = getattr(webdesk, "page_for", None)  # an object with several pages routes by path
            return self._send(200, page_for(self.path) if page_for else webdesk.page())

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            form = {k: v[0] for k, v in parse_qs(self.rfile.read(length).decode(), keep_blank_values=True).items()}
            if not self._local() or not secrets.compare_digest(form.get("csrf", ""), webdesk.token):
                return self._send(403, "forbidden")
            webdesk.post(self.path, form)
            after = getattr(webdesk, "after_post", None)
            self.send_response(303)
            self.send_header("Location", after(self.path) if after else "/")
            self.end_headers()

        def log_message(self, *args):
            pass

    return ThreadingHTTPServer((host, port), Handler)


def serve_in_background(webdesk, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    server = make_server(webdesk, host, port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
