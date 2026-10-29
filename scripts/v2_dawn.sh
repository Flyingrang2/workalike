#!/usr/bin/env bash
# Last comparisons on dev2 (44) before the morning:
#   procG60   base Glimmer with the procedure Glimmer writes from all 60 fixed C records (against the comparison model's 60, pA60)
#   c150+pA60 the balanced 150-row adapter with the comparison model's 60-record procedure in its prompt
#   mem20     base Glimmer with the 20 records the 20-record adapters trained on in its prompt (memory baseline)
#   M=<model dir> bash scripts/v2_dawn.sh runs/v2-dawn
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?usage: v2_dawn.sh <out dir>}
PY=.venv-serve/bin/python
mkdir -p "$OUT"
$PY -m island.gen_v2 --out data/v2 --seed 11 > "$OUT/gen.log"
$PY - <<'PYEOF'
import json, random
from workalike.record import group_episodes, read_events
episodes = list(group_episodes(read_events("data/v2/records/c-train.jsonl")).values())
random.Random("11:curve:1").shuffle(episodes)
with open("data/v2/records/c-s1-first20.jsonl", "w") as f:
    for events in episodes[:20]:
        f.write("".join(json.dumps(e) + "\n" for e in events))
PYEOF
QUANT=fp8_per_channel ADAPTERS="c150=adapters/v2-c150" bash scripts/serve.sh "$OUT/server.log" &
SERVER=$!
trap 'kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null || true' EXIT
for _ in $(seq 180); do curl -sf localhost:8000/v1/models > /dev/null 2>&1 && break; sleep 10; done
D2=data/v2/tasks/dev2.jsonl
[ -f "$OUT/c60-s1.glimmer3.txt" ] || $PY -m island.extract_procedure --records data/v2/records/c-train.jsonl \
    --env quote-desk/2 --shuffle 11:curve:1 --limit 60 --chunk 8 --model muse-glimmer --out "$OUT/c60-s1.glimmer3.txt"
run() {  # name model [procedure] [memory]
    [ -f "$OUT/$1/summary.json" ] && return 0
    $PY -m island.run_agent --env quote-desk/2 --tasks $D2 --model "$2" --temperature 0 --parallel 6 \
        ${3:+--procedure "$3"} ${4:+--memory "$4" --memory-k 20} --out "$OUT/$1" > "$OUT/$1.log" 2>&1
}
pids=()
run procG60 muse-glimmer "$OUT/c60-s1.glimmer3.txt" & pids+=($!)
run c150+pA60 c150 baselines/v2/c60-s1.argon2.txt & pids+=($!)
run mem20 muse-glimmer "" data/v2/records/c-s1-first20.jsonl & pids+=($!)
wait "${pids[@]}"
$PY -m island.report_v2 --tasks $D2 --ref data/v2/records/c-dev2.jsonl --out "$OUT/report-dev2.json" \
    procG60="$OUT/procG60" c150+pA60="$OUT/c150+pA60" mem20="$OUT/mem20"
