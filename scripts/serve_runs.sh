#!/usr/bin/env bash
# Serve the base model with its adapters on one vLLM server and run the same tickets through the tool loop
# for several runs at once (one server batches them). The server stops when this script ends.
#   M=<model dir> QUANT=fp8_per_channel ADAPTERS="a=adapters/a b=adapters/b" \
#   RUNS="base=muse-glimmer a=a b=b proc-a=muse-glimmer+procedure:baselines/procedure-a.txt \
#         mem-a=muse-glimmer+memory:data/h1/records/a-train.jsonl" bash scripts/serve_runs.sh runs/e2-dev
# A run is name=model, optionally followed by +procedure:<file> (a written procedure in the system
# prompt) and +memory:<records> (earlier records in the system prompt; MEMORY_K of them, default 30).
# Server settings (M, QUANT, ADAPTERS, MAX_LEN, MAX_LORA_RANK) are those of scripts/serve.sh.
# AFTER, if set, is a command run after the runs while the server is still up.
# ENV picks the desk version the tickets are for (default quote-desk/1); PARALLEL works that many tickets of a
# run at once (default 1).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?usage: serve_runs.sh <out dir>}
: "${M:?model dir}" "${RUNS:?name=model[+procedure:file][+memory:records] ...}"
: "${TASKS:=data/h1/tasks/dev.jsonl}" "${TEMPERATURE:=0}"
mkdir -p "$OUT"
bash scripts/serve.sh "$OUT/server.log" &
SERVER=$!
trap 'kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null || true' EXIT

for _ in $(seq 180); do
    curl -sf localhost:8000/v1/models > "$OUT/models.json" 2>/dev/null && break
    kill -0 $SERVER 2>/dev/null || { echo "server exited before it was ready"; tail -40 "$OUT/server.log"; exit 1; }
    sleep 10
done
[ -s "$OUT/models.json" ] || { echo "server not ready after 30 minutes"; exit 1; }
grep -E "Loading weights took|Model loading took|Available KV cache|Maximum concurrency" "$OUT/server.log" || true

pids=()
for run in $RUNS; do
    name=${run%%=*} spec=${run#*=}
    IFS=+ read -r model parts <<< "$spec"
    extra=()
    IFS=+ read -ra parts <<< "$parts"
    for part in "${parts[@]}"; do
        case $part in
            procedure:*) extra+=(--procedure "${part#procedure:}") ;;
            memory:*) extra+=(--memory "${part#memory:}" --memory-k "${MEMORY_K:-30}") ;;
            *) echo "unknown run part $part in $run"; exit 1 ;;
        esac
    done
    .venv-serve/bin/python -m island.run_agent --tasks "$TASKS" --model "$model" --temperature "$TEMPERATURE" \
        --env "${ENV:-quote-desk/1}" --parallel "${PARALLEL:-1}" "${extra[@]}" --out "$OUT/$name" > "$OUT/$name.log" 2>&1 &
    pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
# AFTER is an optional command run once the runs are done, while the server is still up.
if [ -n "${AFTER:-}" ]; then echo "== after: $AFTER"; bash -c "$AFTER" || status=1; fi
for run in $RUNS; do
    name=${run%%=*}
    echo "== $name"; cat "$OUT/$name/summary.json" 2>/dev/null || tail -20 "$OUT/$name.log"
done
exit $status
