#!/usr/bin/env bash
# The cheap test: does a person's few-line fix to Glimmer's procedure lift it on dev2 (44)?
#   G0    base Glimmer with Glimmer's 60-record procedure, again (search_catalog came after v2-dawn)
#   G1u   the same procedure fixed by the person whose rules worker C follows
#   G1c   the same procedure fixed by a stand-in that saw only the procedure and the 60 records
#   A60   the comparison model's 60-record procedure, the reference in today's desk
# ARMS picks the runs (default all four); the person's fix can come in a later session with G0 again beside it.
#   M=<model dir> [ARMS="G0 G1c A60"] bash scripts/v2_fix.sh runs/v2-fix
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?usage: v2_fix.sh <out dir>}
PY=.venv-serve/bin/python
mkdir -p "$OUT"
$PY -m island.gen_v2 --out data/v2 --seed 11 > "$OUT/gen.log"
QUANT=fp8_per_channel bash scripts/serve.sh "$OUT/server.log" &
SERVER=$!
trap 'kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null || true' EXIT
for _ in $(seq 180); do curl -sf localhost:8000/v1/models > /dev/null 2>&1 && break; sleep 10; done
D2=data/v2/tasks/dev2.jsonl
run() {  # name procedure
    [ -f "$OUT/$1/summary.json" ] && return 0
    $PY -m island.run_agent --env quote-desk/2 --tasks $D2 --model muse-glimmer --temperature 0 --parallel 6 \
        --procedure "$2" --out "$OUT/$1" > "$OUT/$1.log" 2>&1
}
declare -A PROC=([G0]=baselines/v2/c60-s1.glimmer3.txt [G1u]=baselines/v2/c60-s1.glimmer3.fix-user.txt
                 [G1c]=baselines/v2/c60-s1.glimmer3.fix-claude.txt [A60]=baselines/v2/c60-s1.argon2.txt)
pids=()
for arm in ${ARMS:-G0 G1u G1c A60}; do run "$arm" "${PROC[$arm]}" & pids+=($!); done
for p in "${pids[@]}"; do wait "$p"; done
$PY -m island.report_v2 --tasks $D2 --ref data/v2/records/c-dev2.jsonl --out "$OUT/report-dev2.json" \
    $(for arm in ${ARMS:-G0 G1u G1c A60}; do echo "$arm=$OUT/$arm"; done)
