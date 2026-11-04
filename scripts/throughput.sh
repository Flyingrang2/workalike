#!/usr/bin/env bash
# How many tickets an hour does one GPU work, six at a time? One arm alone on the server, wall clock, on the main test's
# sheet (so its scores are not a new test). Writes tp.log and the two run folders in baselines/results/throughput/.
#   M=<model dir> bash scripts/throughput.sh runs/throughput
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?usage: throughput.sh <out dir>}
: "${M:?model dir}"
PY=.venv-serve/bin/python; B=baselines/test24; C=baselines/test24b
mkdir -p "$OUT"
QUANT=fp8_per_channel ADAPTERS="p-checked=adapters/practice-checked" bash scripts/serve.sh "$OUT/server.log" &
SERVER=$!
trap 'kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null || true' EXIT
for _ in $(seq 120); do curl -sf localhost:8000/v1/models 2>/dev/null | grep -q p-checked && break; sleep 5; done
t0=$(date +%s)
$PY -m testbed.run24 --tasks $C/tickets.jsonl --answers $C/answers.jsonl --arm p-checked=$B/arms/none.txt \
    --served p-checked=p-checked --adapter-prompt product --rounds 1 --parallel 6 --out "$OUT/tp-lora" > "$OUT/tp-lora.log" 2>&1
t1=$(date +%s); echo "LORA_WALL $((t1-t0))" | tee "$OUT/tp.log"
$PY -m testbed.run24 --tasks $C/tickets.jsonl --answers $C/answers.jsonl --arm checked=$B/arms/checked.txt \
    --rounds 1 --parallel 6 --out "$OUT/tp-rules" > "$OUT/tp-rules.log" 2>&1
t2=$(date +%s); echo "RULES_WALL $((t2-t1))" | tee -a "$OUT/tp.log"
echo THROUGHPUT_DONE | tee -a "$OUT/tp.log"
