"""Check the Glimmer chat template against our rows and write token-level training files.

    uv run --no-project --with "transformers>=5" python scripts/render_sft.py \
        --tokenizer <dir with tokenizer.json, tokenizer_config.json, chat_template.jinja> --data data/h1

For every sft/*-train.jsonl it writes sft/*-train.tok.jsonl (input_ids, attention_mask,
labels) and a preview of the first row in render/. Exits non-zero if any check fails.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workalike.render import RenderError, load_tokenizer, preview, tokenize  # noqa: E402
from workalike.sft import read_jsonl, write_jsonl  # noqa: E402

EXPECTED_EOS = [200001, 200008]
EOM = 200007


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--data", default="data/h1")
    args = ap.parse_args()
    tok = load_tokenizer(args.tokenizer)
    data = Path(args.data)
    failures = []

    gen = json.loads((Path(args.tokenizer) / "generation_config.json").read_text())
    ids = {t: tok.convert_tokens_to_ids(t) for t in ("<|eot|>", "<|eom|>", "<|end_of_text|>")}
    print("eos in generation_config:", gen["eos_token_id"], "special ids:", ids)
    if gen["eos_token_id"] != EXPECTED_EOS or EOM in gen["eos_token_id"] or ids["<|eom|>"] != EOM:
        failures.append("stop tokens differ from the documented ones")

    for path in sorted((data / "sft").glob("*-train.jsonl")):
        rows = read_jsonl(path)
        out, stats = [], []
        for i, row in enumerate(rows):
            try:
                encoded, s = tokenize(tok, row["messages"], row["tools"])
            except RenderError as e:
                failures.append(f"{path.name} row {i}: {e}")
                continue
            for key in ("roundtrip", "turns_end_with_eot"):
                if not s[key]:
                    failures.append(f"{path.name} row {i}: {key} failed")
            if s["straddle"]:
                failures.append(f"{path.name} row {i}: {s['straddle']} tokens cross a train boundary")
            out.append(encoded)
            stats.append(s)
        write_jsonl(out, path.with_suffix(".tok.jsonl"))
        (data / "render").mkdir(exist_ok=True)
        (data / "render" / f"{path.stem}-0.txt").write_text(preview(tok, rows[0]["messages"], rows[0]["tools"]))
        lengths = [s["tokens"] for s in stats]
        print(f"{path.name}: rows {len(rows)}, tokens mean {statistics.mean(lengths):.0f} max {max(lengths)}, "
              f"trained share {sum(s['trained'] for s in stats) / sum(lengths):.1%}")

    if failures:
        print(f"{len(failures)} failures")
        for f in failures[:20]:
            print(" ", f)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
