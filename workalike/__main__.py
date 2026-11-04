"""Serve the product loop: the Korean quote desk on the left, the agent pane on the right (workalike.app).

    python -m workalike --out runs/loop --tasks data/human/htrain.jsonl --korean <ko.json> \
        --translator "<your translator command>" --cases data/loop/cases.jsonl \
        --examples data/v2/records/c-train.jsonl --model muse-glimmer [--cards runs/v2-cards/c60-s1.glimmer.json]

--lang en shows both panes in English. --korean and --translator are then not needed and not used: the person reads
each ticket's English mail and types English.

Open http://127.0.0.1:8770. The person's records go to <out>/records.jsonl through the recording switch; the
agent's go to <out>/agent.jsonl. --examples puts a virtual worker's records (the first 60 in the 11:curve:1 order,
the ones the procedures were written from) beside the person's, read-only, so cards can be proposed before the
person has worked many tickets. --cards offers cards Glimmer proposed earlier, labelled as such.
"""

from __future__ import annotations

import argparse
import json
import random
import shlex
import subprocess
from pathlib import Path

from .app import DeskApp
from .filter import clean_episode, episode_meta
from .model_client import ChatModel
from .person_desk import AsWritten, Korean, PersonDesk, as_written, command_translator
from .record import Recorder, RecordSwitch, group_episodes, read_events
from .sft import read_jsonl
from .web import keep_finished, make_server, sessions_for


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", help="tickets for the person's desk (quote-desk/2)")
    ap.add_argument("--korean", help="JSON with those tickets' Korean and the fixed answers")
    ap.add_argument("--translator", help="Korean on stdin, English on stdout; '--to ko' is added for the other way")
    ap.add_argument("--cases", required=True, help="the fixed cases (testbed.loop_cases)")
    ap.add_argument("--examples", help="a virtual worker's records, shown read-only")
    ap.add_argument("--examples-label", default="가상 직원 C의 기록")
    ap.add_argument("--examples-limit", type=int, default=60)
    ap.add_argument("--cards", help="cards proposed earlier (testbed.propose_cards output)")
    ap.add_argument("--learn", help="the autolearn versions folder: show its state and work with its active adapter")
    ap.add_argument(
        "--book", help="start from this rule book (cards a person has already checked), if out has none yet"
    )
    ap.add_argument("--model", default="muse-glimmer")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--actor", default="person")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--lang", choices=("ko", "en"), default="en", help="the language of both panes")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.book and not (out / "rules.json").exists():
        (out / "rules.json").write_text(Path(args.book).read_text())

    translator = args.translator if args.lang == "ko" else None  # in English nothing is translated
    to_en = command_translator(translator) if translator else None

    def to_ko(text):
        res = subprocess.run(
            shlex.split(translator) + ["--to", "ko"],
            input=text,
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        ).stdout
        return json.loads(res)["ko"]

    desk, switch = None, None
    if args.tasks:
        done = keep_finished(out / "records.jsonl")
        todo = [t for t in read_jsonl(args.tasks) if t["task_id"] not in done]
        switch = RecordSwitch(Recorder(out / "records.jsonl"))
        if args.lang == "en":
            desk = PersonDesk(sessions_for(todo, args.actor), switch, AsWritten(), as_written, lang="en")
        elif not (args.korean and translator):
            raise SystemExit("--tasks needs --korean and --translator")
        else:
            desk = PersonDesk(
                sessions_for(todo, args.actor),
                switch,
                Korean(json.loads(Path(args.korean).read_text())),
                to_en,
                command_translator(translator + " --term"),
            )
    examples = []
    if args.examples:
        examples = [
            (episode_meta(ev)["ep"], clean_episode(ev)) for ev in group_episodes(read_events(args.examples)).values()
        ]
        random.Random("11:curve:1").shuffle(examples)
        examples = examples[: args.examples_limit]
    app = DeskApp(
        out,
        lambda: ChatModel(args.base_url, args.model, temperature=0, max_tokens=16000),
        read_jsonl(args.cases),
        desk,
        switch,
        out / "records.jsonl" if desk else None,
        examples,
        args.examples_label,
        to_en,
        to_ko if translator else None,
        args.cards,
        rule_en=command_translator(translator + " --rule") if translator else None,
        versions=args.learn,
        adapter_model=lambda name: ChatModel(args.base_url, name, temperature=0, max_tokens=16000),
        lang=args.lang,
    )
    if desk:
        desk.badge = app.badge
    server = make_server(app, port=args.port)
    print(f"http://127.0.0.1:{args.port}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
