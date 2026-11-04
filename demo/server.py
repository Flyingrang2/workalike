"""The try-it server: picks in, one new ticket worked with and without the manual, both told step by step.

    BASETEN_API_KEY=... python -m demo.server --port 8080

It holds the key and nothing else: no visitor state, no free text in. Only known picks are accepted, and limits say no
before a model is called. When it says no, the page plays a recorded run of the same picks and labels it recorded.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import queue
import threading
import time
import urllib.request
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from workalike.sft import read_jsonl

from .manual import DATE, PRICE, manual
from .work import cost, run_arm

TICKETS = Path(__file__).parent / "tickets.jsonl"
PER_IP, PER_IP_SECONDS = 8, 600  # runs per visitor address
AT_ONCE = (
    3  # runs at the same time; each is two tickets, about 16 calls in 25 seconds, and the key allows 100 calls a minute
)
DAY_USD, TOTAL_USD = 20.0, 80.0
BODY_LIMIT = 200
METADATA = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"


class GcsObject:
    """One small JSON object in a bucket, read and written with the instance's own identity."""

    def __init__(self, bucket: str, name: str = "meter.json"):
        self.read_url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{name}?alt=media"
        self.write_url = f"https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o?uploadType=media&name={name}"

    def _token(self) -> str:
        req = urllib.request.Request(METADATA, headers={"Metadata-Flavor": "Google"})
        return json.load(urllib.request.urlopen(req, timeout=5))["access_token"]

    def load(self) -> dict:
        req = urllib.request.Request(self.read_url, headers={"Authorization": "Bearer " + self._token()})
        try:
            return json.load(urllib.request.urlopen(req, timeout=5))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return {}
            raise

    def save(self, data: dict) -> None:
        req = urllib.request.Request(
            self.write_url,
            data=json.dumps(data).encode(),
            method="POST",
            headers={"Authorization": "Bearer " + self._token(), "Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=5).read()


class Meter:
    """Money spent so far, kept outside the process so a restart does not forget it. If it cannot be read, spend nothing."""

    def __init__(
        self, store=None, day_usd=DAY_USD, total_usd=TOTAL_USD, today=lambda: datetime.date.today().isoformat()
    ):
        self.store, self.day_usd, self.total_usd, self.today = store, day_usd, total_usd, today
        self.lock = threading.Lock()
        self.spent, self.days, self.broken = 0.0, {}, False
        if store:
            try:
                data = store.load()
                self.spent, self.days = data.get("spent", 0.0), data.get("days", {})
            except Exception as e:
                print("meter unreadable:", type(e).__name__, flush=True)
                self.broken = True

    def open(self) -> bool:
        with self.lock:
            return not self.broken and self.spent < self.total_usd and self.days.get(self.today(), 0.0) < self.day_usd

    def add(self, usd: float) -> None:
        with self.lock:
            self.spent += usd
            self.days[self.today()] = self.days.get(self.today(), 0.0) + usd
            data = {"spent": round(self.spent, 4), "days": {k: round(v, 4) for k, v in self.days.items()}}
        if self.store:
            try:
                self.store.save(data)
            except Exception as e:  # the count in memory still holds while this process lives
                print("meter not saved:", type(e).__name__, flush=True)


class Limits:
    def __init__(self, per_ip=PER_IP, per_ip_seconds=PER_IP_SECONDS, at_once=AT_ONCE, clock=time.monotonic):
        self.per_ip, self.seconds, self.clock = per_ip, per_ip_seconds, clock
        self.slots = threading.BoundedSemaphore(at_once)
        self.lock = threading.Lock()
        self.seen: dict[str, deque] = defaultdict(deque)

    def take(self, ip: str) -> str | None:
        """None if this run may go, or the reason it may not."""
        now = self.clock()
        with self.lock:
            hits = self.seen[ip]
            while hits and hits[0] < now - self.seconds:
                hits.popleft()
            if len(hits) >= self.per_ip:
                return "visitor_limit"
            if not self.slots.acquire(blocking=False):
                return "busy"
            hits.append(now)
        return None

    def give_back(self) -> None:
        self.slots.release()


def valid(body: dict, tickets: dict) -> bool:
    return (
        set(body) == {"ticket", "price", "date"}
        and body["ticket"] in tickets
        and body["price"] in PRICE
        and body["date"] in DATE
    )


def stream_run(body: dict, tickets: dict, api_key: str, meter: Meter, write, make_model=None) -> None:
    """Both tries at once on one stream. write(event) may raise if the visitor left; the work still finishes and is paid for."""
    events: queue.Queue = queue.Queue()
    task = tickets[body["ticket"]]
    arms = {"none": "", "manual": manual(body["price"], body["date"])}

    def one(arm, rules):
        done = run_arm(task, rules, api_key, lambda e: events.put({"arm": arm, **e}), make_model)
        meter.add(cost(done))

    threads = [threading.Thread(target=one, args=a, daemon=True) for a in arms.items()]
    for t in threads:
        t.start()
    finished, alive = 0, True
    while finished < len(arms):
        try:
            event = events.get(timeout=10)
        except queue.Empty:
            event = None  # a comment line keeps the connection open while the model thinks
        if event and event["type"] == "done":
            finished += 1
        if alive:
            try:
                write(event)
            except OSError:
                alive = False
    if alive:
        write({"type": "end"})


def handler(tickets: dict, api_key: str, meter: Meter, limits: Limits, origins: set[str], make_model=None):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_a):
            pass  # no addresses or picks in the logs

        def _cors(self):
            origin = self.headers.get("Origin")
            if origin in origins:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")

        def _once(self):
            """One request per connection: a body left unread can never be taken for the next request."""
            self.close_connection = True
            self.send_header("Connection", "close")

        def _json(self, code, data):
            raw = json.dumps(data).encode()
            self.send_response(code)
            self._once()
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_OPTIONS(self):
            self.send_response(204)
            self._once()
            self._cors()
            self.send_header("Access-Control-Allow-Methods", "POST, GET")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Max-Age", "3600")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            if self.path == "/health":
                return self._json(200, {"live": meter.open()})
            self._json(404, {"error": "not_found"})

        def do_POST(self):
            if self.path != "/run":
                return self._json(404, {"error": "not_found"})
            try:
                size = int(self.headers.get("Content-Length") or 0)
                if "Transfer-Encoding" in self.headers or not 0 < size <= BODY_LIMIT:
                    return self._json(400, {"error": "bad_request"})
                body = json.loads(self.rfile.read(size))
            except ValueError:
                return self._json(400, {"error": "bad_request"})
            if not isinstance(body, dict) or not valid(body, tickets):
                return self._json(400, {"error": "bad_request"})
            if not meter.open():
                return self._json(503, {"error": "resting"})
            # Cloud Run adds the address it saw at the end of the list; whatever comes before is the visitor's to fake.
            ip = (self.headers.get("X-Forwarded-For") or self.client_address[0]).split(",")[-1].strip()
            refused = limits.take(ip)
            if refused:
                return self._json(429, {"error": refused})
            try:
                self.send_response(200)
                self._once()
                self._cors()
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

                def write(event):
                    line = b": still working\n\n" if event is None else f"data: {json.dumps(event)}\n\n".encode()
                    self.wfile.write(f"{len(line):x}\r\n".encode() + line + b"\r\n")
                    self.wfile.flush()

                stream_run(body, tickets, api_key, meter, write, make_model)
                self.wfile.write(b"0\r\n\r\n")
            except OSError:
                self.close_connection = True
            finally:
                limits.give_back()

    return Handler


def load_tickets(path=TICKETS) -> dict:
    return {t["kind"]: t for t in read_jsonl(path)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8080)))
    a = p.parse_args()
    origins = set(filter(None, os.environ.get("ALLOW_ORIGINS", "").split(",")))
    bucket = os.environ.get("METER_BUCKET")
    meter = Meter(GcsObject(bucket) if bucket else None)
    app = handler(load_tickets(), os.environ["BASETEN_API_KEY"], meter, Limits(), origins)
    ThreadingHTTPServer(("0.0.0.0", a.port), app).serve_forever()


if __name__ == "__main__":
    main()
