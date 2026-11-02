"""Learn from your own records: review what will be learned, leave episodes out, and build a training version.

    python -m workalike.learn review --records runs/me/records.jsonl --versions versions/me   # http://127.0.0.1:8772
    python -m workalike.learn build --records runs/me/records.jsonl --versions versions/me
    python -m workalike.learn list --versions versions/me

review shows each recorded episode exactly as training will see it (the same filter:
only what was on screen and what was done), and lets you leave episodes out; the
choice is kept in versions/me/exclude.json. Unfinished episodes are left out always.
build writes the next version folder (v1, v2, ...) with the training rows in
sft/me-train.jsonl and a manifest of the episodes used and left out. Training and
serving run on a GPU box:

    python scripts/render_sft.py --tokenizer <model dir> --data versions/me/v1
    python scripts/train_lora.py --base <model dir> --data versions/me/v1/sft/me-train.tok.jsonl --out versions/me/v1/adapter

list prints each version and, for those with an adapter, the --lora-modules entry for vLLM,
so the assist window can draft with any version by its name.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import secrets
import threading
from pathlib import Path

from .filter import clean_episode, episode_meta
from .record import group_episodes, read_events, wall_clock
from .serialize import render_view
from .sft import sft_row, write_jsonl
from .web import CSS, describe, make_server


def episodes(records, actor: str | None = None) -> list[dict]:
    """Every recorded episode of one actor (or all), with its clean form and whether it finished."""
    out = []
    for events in group_episodes(read_events(records)).values():
        meta = episode_meta(events)
        if actor and meta["actor"] != actor:
            continue
        status = next((e.get("status") for e in events if e["type"] == "episode_end"), "unfinished")
        out.append({"ep": meta["ep"], "status": status, "clean": clean_episode(events)})
    return out


def load_excluded(versions: Path) -> set[str]:
    path = versions / "exclude.json"
    return set(json.loads(path.read_text())["excluded"]) if path.exists() else set()


def save_excluded(versions: Path, excluded) -> None:
    versions.mkdir(parents=True, exist_ok=True)
    (versions / "exclude.json").write_text(json.dumps({"excluded": sorted(excluded)}, indent=1) + "\n")


def build(records, versions, actor: str | None = None, name: str | None = None, env=None) -> dict:
    """env is a desk version from workalike.envs; None is quote-desk/1."""
    versions = Path(versions)
    name = name or versions.name
    excluded = load_excluded(versions)
    eps = episodes(records, actor)
    kept = [e for e in eps if e["status"] == "done" and e["ep"] not in excluded]
    if not kept:
        raise SystemExit("no finished episode left to learn from")
    n = 1 + max((int(p.name[1:]) for p in versions.glob("v*") if p.name[1:].isdigit()), default=0)
    out = versions / f"v{n}"
    write_jsonl([sft_row(e["clean"], env) for e in kept], out / "sft" / f"{name}-train.jsonl")
    manifest = {"version": f"v{n}", "name": f"{name}-v{n}", "created": wall_clock(), "actor": actor,
                "records": str(records), "records_sha256": hashlib.sha256(Path(records).read_bytes()).hexdigest(),
                "rows": len(kept), "episodes": [e["ep"] for e in kept],
                "left_out": sorted(e["ep"] for e in eps if e["ep"] in excluded),
                "unfinished": sorted(e["ep"] for e in eps if e["status"] != "done")}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def list_versions(versions) -> list[dict]:
    out = []
    for path in sorted(Path(versions).glob("v*/manifest.json"), key=lambda p: int(p.parent.name[1:])):
        m = json.loads(path.read_text())
        adapter = path.parent / "adapter"
        out.append({**{k: m[k] for k in ("version", "name", "created", "rows")},
                    "adapter": str(adapter) if (adapter / "adapter_config.json").exists() else None})
    return out


def _step(call: dict) -> str:
    if call["result"] is None:
        return f"{call['name']} {json.dumps(call['args'], ensure_ascii=False)} (no result)"
    return describe(call["name"], call["args"], call["result"])


class Review:
    """One page: every episode as training will see it, each with a box to leave it out."""

    def __init__(self, records, versions, actor: str | None = None):
        self.records, self.versions, self.actor = records, Path(versions), actor
        self.token = secrets.token_urlsafe(16)
        self.lock = threading.Lock()

    def post(self, path: str, form: dict[str, str]) -> None:
        if path == "/save":
            with self.lock:
                save_excluded(self.versions, [k[len("out:"):] for k in form if k.startswith("out:")])

    def status(self) -> dict:
        return {"excluded": sorted(load_excluded(self.versions)), "versions": list_versions(self.versions)}

    def page(self) -> str:
        e = html.escape
        excluded = load_excluded(self.versions)
        cards = []
        for ep in episodes(self.records, self.actor):
            c = ep["clean"]
            steps = "".join(f"<p>{i}. {e(_step(call))}</p>"
                            for i, call in enumerate((x for t in c["turns"] for x in t["calls"]), 1))
            done = ep["status"] == "done"
            box = (f"<label><input type='checkbox' name='out:{e(ep['ep'])}' style='width:auto'"
                   f"{' checked' if ep['ep'] in excluded else ''}> Leave out</label>" if done
                   else "<small>Unfinished, always left out</small>")
            cards.append(f"<div class='card'><h2>{e(c['view'].get('ticket_id', ''))} {box}</h2>"
                         f"<div class='req'>{e(render_view(c['view']))}</div>{steps or '<p>No actions.</p>'}</div>")
        versions = "".join(f"<p>{e(v['name'])}: {v['rows']} episodes, {e(v['created'])}"
                           f"{', adapter ready' if v['adapter'] else ''}</p>" for v in list_versions(self.versions))
        return (f"<!doctype html><html><head><meta charset='utf-8'><title>Review</title><style>{CSS}"
                "main{display:block;height:auto}.card{margin-bottom:12px}.req{white-space:pre-wrap;margin:6px 0}"
                ".card p{margin:2px 0}</style></head><body><form method='post' action='/save'>"
                f"<input type='hidden' name='csrf' value='{self.token}'><header><b>Review</b>"
                "<span>What training will see: only what was on screen and what was done.</span>"
                "<button class='finish'>Save</button></header><main>"
                f"<div class='card'><h2>Versions</h2>{versions or '<p>None yet.</p>'}</div>{''.join(cards)}"
                "</main></form></body></html>")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("review", "build", "list"):
        p = sub.add_parser(cmd)
        p.add_argument("--versions", required=True, help="folder for this person's versions, e.g. versions/me")
        if cmd != "list":
            p.add_argument("--records", required=True)
            p.add_argument("--actor", help="only this actor's episodes")
        if cmd == "review":
            p.add_argument("--port", type=int, default=8772)
    args = ap.parse_args()
    if args.cmd == "review":
        server = make_server(Review(args.records, args.versions, args.actor), port=args.port)
        print(f"http://127.0.0.1:{args.port}  Ctrl+C to stop", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    elif args.cmd == "build":
        m = build(args.records, args.versions, args.actor)
        print(f"{m['name']}: {m['rows']} episodes, {len(m['left_out'])} left out, {len(m['unfinished'])} unfinished")
    else:
        versions = list_versions(args.versions)
        for v in versions:
            print(f"{v['name']:16} {v['rows']:4} episodes  {v['created']}  {v['adapter'] or 'no adapter yet'}")
        ready = " ".join(f"{v['name']}={v['adapter']}" for v in versions if v["adapter"])
        if ready:
            print(f"--lora-modules {ready}")


if __name__ == "__main__":
    main()
