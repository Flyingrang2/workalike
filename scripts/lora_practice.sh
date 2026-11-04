#!/usr/bin/env bash
# The model practises with the rules a person checked and learns from its own work: no rules left in the adapter's prompt.
# One run, fixed settings. Then the adapter on the fixed 24 (seen, a regression check) and, beside the rules it came from,
# on a new sheet nobody has looked at (baselines/test24b).
#   M=<model dir> bash scripts/lora_practice.sh runs/lora-practice
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?usage: lora_practice.sh <out dir>}
PY=.venv-serve/bin/python
B=baselines/test24
C=baselines/test24b
T=${T:-checked}                   # a name for this run; TEACHER is the rules as plain text, by default the person's checked rules
TEACHER=${TEACHER:-$B/arms/$T.txt}
A=adapters/practice-$T
DATA=data/practice-$T
mkdir -p "$OUT" adapters "$DATA/sft"

stop_server() {
    [ -n "${SERVER:-}" ] && kill "$SERVER" 2>/dev/null && wait "$SERVER" 2>/dev/null || true
    SERVER=
    for _ in $(seq 60); do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        { ! curl -sf localhost:8000/v1/models > /dev/null 2>&1; } && [ "$used" -lt 3000 ] && return 0
        sleep 5
    done
    echo "the server did not give the card back (used ${used} MiB)"; exit 1
}
start_server() {  # log, expected model name, adapters
    stop_server
    ADAPTERS="$3" QUANT=fp8_per_channel bash scripts/serve.sh "$1" &
    SERVER=$!
    for _ in $(seq 120); do
        curl -sf localhost:8000/v1/models 2>/dev/null | grep -q "\"$2\"" && return 0
        kill -0 "$SERVER" 2>/dev/null || { echo "the server died: $1"; exit 1; }
        sleep 10
    done
    echo "the server did not come up: $1"; exit 1
}
trap 'stop_server' EXIT
sha256sum "$TEACHER" $C/*.jsonl > "$OUT/sheet.sha256"

if [ ! -f "$OUT/practice/summary.json" ]; then
    start_server "$OUT/server-teach.log" muse-glimmer ""
    $PY -m testbed.practice --teacher "$TEACHER" --out "$OUT/practice" --sft "$DATA/sft/$T-train.jsonl" | tee "$OUT/practice.log"
fi
stop_server
if [ ! -f "$A/trained.ok" ]; then
    .venv-train/bin/python scripts/render_sft.py --tokenizer "$M" --data "$DATA" > "$OUT/render.log"
    rm -rf "$A"
    .venv-train/bin/python scripts/train_lora.py --base "$M" --data "$DATA/sft/$T-train.tok.jsonl" --seed 1 --out "$A" > "$OUT/train.log" 2>&1
    touch "$A/trained.ok"
fi
start_server "$OUT/server-test.log" "p-$T" "p-$T=$A"
$PY -m testbed.run24 --tasks $B/tickets.jsonl --answers $B/answers.jsonl --arm "$T=$TEACHER" --arm p-$T=$B/arms/none.txt --served p-$T=p-$T \
    --adapter-prompt product --rounds 2 --parallel 6 --fail $B/fail.jsonl --fail-rounds 3 --fail-arm p-$T --out "$OUT/regress" > "$OUT/regress.log"
$PY -m testbed.run24 --tasks $C/tickets.jsonl --answers $C/answers.jsonl --arm none=$B/arms/none.txt --arm "$T=$TEACHER" \
    --arm p-$T=$B/arms/none.txt --served p-$T=p-$T --adapter-prompt product --rounds 2 --parallel 6 --out "$OUT/new" > "$OUT/new.log"
