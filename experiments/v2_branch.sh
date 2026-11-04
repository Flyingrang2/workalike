#!/usr/bin/env bash
# Where do C's conditional steps get lost? At C's choice turns in the training rows:
#   nf4   the training stack (NF4, as train_lora loads it): log-probabilities of C's step and the routine one
#         (experiments/branch_probs.py) for base Glimmer and the adapters fA60, fB60
#   fp8   the serving stack (vLLM, online FP8): the next step the same models take from C's exact history
#         (testbed.next_action --choice-only)
# If the training stack prefers C's step and the server does not, serving loses it; if neither, it was not learned.
#   M=<model dir> bash experiments/v2_branch.sh runs/v2-branch
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?usage: v2_branch.sh <out dir>}
PY=.venv-serve/bin/python
mkdir -p "$OUT"
$PY -m testbed.gen_v2 --out data/v2 --seed 11 > "$OUT/gen.log"
.venv-train/bin/python experiments/branch_probs.py --base "$M" --rows data/v2/sft/c-s1-train.jsonl \
    --tasks data/v2/tasks/train.jsonl --adapter fA60=adapters/v2-fA60 --adapter fB60=adapters/v2-fB60 \
    --out "$OUT/nf4.json" > "$OUT/nf4.log" 2>&1
QUANT=fp8_per_channel ADAPTERS="fA60=adapters/v2-fA60 fB60=adapters/v2-fB60" bash scripts/serve.sh "$OUT/server.log" &
SERVER=$!
trap 'kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null || true' EXIT
for _ in $(seq 180); do curl -sf localhost:8000/v1/models > /dev/null 2>&1 && break; sleep 10; done
pids=()
for m in fA60 fB60 muse-glimmer; do
    $PY -m testbed.next_action --env quote-desk/2 --records data/v2/records/c-train.jsonl \
        --tasks data/v2/tasks/train.jsonl --model $m --choice-only --out "$OUT/fp8-$m.json" > "$OUT/fp8-$m.log" 2>&1 &
    pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid" || true; done
tail -30 "$OUT/nf4.log" | sed -n '/^{$/,$p'
for m in fA60 fB60 muse-glimmer; do echo "== fp8 $m"; $PY -c "import json,sys; print(json.dumps(json.load(open(sys.argv[1]))['summary']))" "$OUT/fp8-$m.json"; done
