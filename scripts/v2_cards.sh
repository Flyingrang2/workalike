#!/usr/bin/env bash
# Glimmer's rule cards from C's 60 fixed records (the records its 60-record procedure came from), twice:
# parsed from plain text, and with the server's structured output on. Product loop step 2, tried early.
#   M=<model dir> bash scripts/v2_cards.sh runs/v2-cards
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?usage: v2_cards.sh <out dir>}
PY=.venv-serve/bin/python
mkdir -p "$OUT"
$PY -m island.gen_v2 --out data/v2 --seed 11 > "$OUT/gen.log"
QUANT=fp8_per_channel bash scripts/serve.sh "$OUT/server.log" &
SERVER=$!
trap 'kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null || true' EXIT
for _ in $(seq 180); do curl -sf localhost:8000/v1/models > /dev/null 2>&1 && break; sleep 10; done
cards() {  # name [extra args]
    local name=$1; shift
    [ -f "$OUT/$name.json" ] && return 0
    $PY -m island.propose_cards --records data/v2/records/c-train.jsonl --env quote-desk/2 --shuffle 11:curve:1 \
        --limit 60 --model muse-glimmer "$@" --out "$OUT/$name.json" > "$OUT/$name.log" 2>&1 || echo "$name failed" >> "$OUT/failed.txt"
}
cards c60-s1.glimmer & p1=$!
cards c60-s1.glimmer.schema --json-schema & p2=$!
wait $p1 $p2
ls "$OUT"/*.json
