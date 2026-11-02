#!/usr/bin/env bash
# The main test on the served model, on the sheet fixed in baselines/test24.
#   M=<model dir> DEV=1 bash scripts/test24.sh runs/test24-dev    the short check: 4 tickets, cards and checked, once
#   M=<model dir> bash scripts/test24.sh runs/test24              all four arms, twice, then the failure scenarios
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?usage: test24.sh <out dir>}
PY=.venv-serve/bin/python
B=baselines/test24
mkdir -p "$OUT"
sha256sum $B/*.jsonl $B/arms/*.txt > "$OUT/sheet.sha256"
QUANT=fp8_per_channel bash scripts/serve.sh "$OUT/server.log" &
SERVER=$!
trap 'kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null || true' EXIT
for _ in $(seq 180); do curl -sf localhost:8000/v1/models > /dev/null 2>&1 && break; sleep 10; done
if [ -n "${DEV:-}" ]; then
    $PY -m island.run24 --tasks $B/dev.jsonl --answers $B/dev-answers.jsonl --arm cards=$B/arms/cards.txt \
        --arm checked=$B/arms/checked.txt --rounds 1 --parallel 6 --fail $B/fail.jsonl --fail-rounds 1 --out "$OUT"
else
    $PY -m island.run24 --tasks $B/tickets.jsonl --answers $B/answers.jsonl --arm none=$B/arms/none.txt \
        --arm records=$B/arms/records.txt --arm cards=$B/arms/cards.txt --arm checked=$B/arms/checked.txt \
        --rounds 2 --parallel 6 --fail $B/fail.jsonl --fail-rounds 3 --out "$OUT"
fi
