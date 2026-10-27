"""Drive the assist window once per model with a real served model: ask, wait for the drafts, approve.

    python -m island.assist_lap --tasks data/h1/tasks/dev.jsonl --family stock_short \
        --models a b muse-glimmer --out runs/p1-lap

Each model gets its own window on the shop behind one test-bed ticket, and the ticket's
message is sent through the page form, as a person would. The page is saved (HTML, and
a screenshot when Playwright is installed) while the drafts wait and again after Approve.
lap.json keeps, per model, the job status, the checks, the drafts and the token use.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

from workalike.assist import Assist
from workalike.model_client import ChatModel
from workalike.sft import read_jsonl
from workalike.web import serve_in_background

from .shop import shop_from


def post(url: str, form: dict) -> None:
    data = urllib.parse.urlencode(form).encode()
    urllib.request.urlopen(urllib.request.Request(url, data=data, method="POST")).read()


def save_page(url: str, path: Path) -> None:
    path.with_suffix(".html").write_bytes(urllib.request.urlopen(url).read())
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(url)
        for summary in page.query_selector_all("summary"):
            summary.click()
        page.screenshot(path=str(path.with_suffix(".png")), full_page=True)
        browser.close()


def lap(task: dict, model_name: str, base_url: str, out: Path, timeout: float) -> dict:
    model = ChatModel(base_url, model_name, max_tokens=8192, temperature=0)
    app = Assist(shop_from(task), {model_name: model}, out)
    server = serve_in_background(app, port=0)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        start = time.monotonic()
        post(url + "/ask", {"csrf": app.token, "from": task["view"]["from"], "body": task["view"]["body"]})
        while app.jobs[0]["status"] == "working" and time.monotonic() - start < timeout:
            time.sleep(2)
        job = app.jobs[0]
        save_page(url, out / "waiting")
        if job["status"] == "waiting":
            post(url + "/approve", {"csrf": app.token, "job": "1"})
            save_page(url, out / "approved")
        return {"model": model_name, "task_id": task["task_id"], "family": task["family"],
                "status": job["status"], "run_status": job.get("run_status"), "seconds": round(time.monotonic() - start, 1),
                "checks": job["checks"], "replies": job["replies"], "quotes": job["quotes"],
                "trace": [c["name"] for c in job["calls"]],
                "completion_tokens": sum((c["usage"] or {}).get("completion_tokens", 0) for c in model.calls)}
    finally:
        server.shutdown()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--family", default="standard", help="use the first ticket of this family")
    ap.add_argument("--models", nargs="+", required=True, help="served names: adapters or the base model")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    task = next(t for t in read_jsonl(args.tasks) if t["family"] == args.family)
    results = []
    for name in args.models:
        out = Path(args.out) / name
        out.mkdir(parents=True, exist_ok=True)
        results.append(lap(task, name, args.base_url, out, args.timeout))
        r = results[-1]
        print(f"{name}: {r['status']}, {len(r['replies'])} replies, checks "
              f"{sum(c['ok'] for c in r['checks'])}/{len(r['checks'])}, {r['trace']}", flush=True)
    (Path(args.out) / "lap.json").write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
