#!/usr/bin/env bash
# Merge the batch cards of v2-cards2 again with the one-card-per-situation prompt (Glimmer, no new reading).
#   M=<model dir> bash experiments/v2_cards3.sh runs/v2-cards3
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?usage: v2_cards3.sh <out dir>}
PY=.venv-serve/bin/python
mkdir -p "$OUT" runs/v2-cards2
[ -f runs/v2-cards2/c60-s1.glimmer.json ] || cp ${BACKUP_DIR:-$HOME/backup}/runs/v2-cards2/c60-s1.glimmer.json runs/v2-cards2/
$PY -m testbed.gen_v2 --out data/v2 --seed 11 > "$OUT/gen.log"
QUANT=fp8_per_channel bash scripts/serve.sh "$OUT/server.log" &
SERVER=$!
trap 'kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null || true' EXIT
for _ in $(seq 180); do curl -sf localhost:8000/v1/models > /dev/null 2>&1 && break; sleep 10; done
$PY -m testbed.propose_cards --records data/v2/records/c-train.jsonl --env quote-desk/2 --shuffle 11:curve:1 \
    --limit 60 --model muse-glimmer --remerge runs/v2-cards2/c60-s1.glimmer.json --out "$OUT/c60-s1.glimmer.merge2.json"
