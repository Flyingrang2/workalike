#!/usr/bin/env bash
# Can the pipeline turn natural records into a useful LoRA by itself? The same 60 records of worker C as fA60, repeated
# by what the worker did (workalike.rebalance, no rules, no labels), same training settings (3 epochs, NF4, r16, seed 1).
# Compared on dev2 with fA60 and c150 in today's desk, then on the fixed 24 through the product's path.
#   M=<model dir> bash experiments/lora_auto.sh runs/lora-auto
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?usage: lora_auto.sh <out dir>}
PY=.venv-serve/bin/python
B=baselines/test24
KEEP=${BACKUP_DIR:-$HOME/backup}/adapters
mkdir -p "$OUT" adapters runs/v2-train
for a in v2-fA60 v2-c150; do [ -d adapters/$a ] || cp -r $KEEP/$a adapters/; done
$PY -m testbed.gen_v2 --out data/v2 --seed 11 > "$OUT/gen.log"
$PY -m workalike.rebalance --rows data/v2/sft/c-s1-train.jsonl --limit 60 --target 150 --out data/v2/sft/auto60-s1-train.jsonl > "$OUT/autobalance.log"
.venv-train/bin/python scripts/render_sft.py --tokenizer "$M" --data data/v2 > "$OUT/render.log"
[ -f adapters/v2-auto60/adapter_config.json ] || .venv-train/bin/python scripts/train_lora.py --base "$M" \
    --data data/v2/sft/auto60-s1-train.tok.jsonl --seed 1 --out adapters/v2-auto60 > runs/v2-train/auto60.log 2>&1
ADAPTERS="auto60=adapters/v2-auto60 fA60=adapters/v2-fA60 c150=adapters/v2-c150" QUANT=fp8_per_channel \
    bash scripts/serve.sh "$OUT/server.log" &
SERVER=$!
trap 'kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null || true' EXIT
for _ in $(seq 180); do curl -sf localhost:8000/v1/models > /dev/null 2>&1 && break; sleep 10; done
D2=data/v2/tasks/dev2.jsonl
pids=()
for m in auto60 fA60 c150; do
    [ -f "$OUT/dev2-$m/summary.json" ] || $PY -m testbed.run_agent --env quote-desk/2 --tasks $D2 --model $m --temperature 0 \
        --parallel 6 --out "$OUT/dev2-$m" > "$OUT/dev2-$m.log" 2>&1 &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
$PY -m testbed.report_v2 --tasks $D2 --ref data/v2/records/c-dev2.jsonl --out "$OUT/report-dev2.json" \
    auto60="$OUT/dev2-auto60" fA60="$OUT/dev2-fA60" c150="$OUT/dev2-c150"
$PY -m testbed.run24 --tasks $B/tickets.jsonl --answers $B/answers.jsonl --arm auto60=$B/arms/none.txt --arm fA60=$B/arms/none.txt \
    --arm c150=$B/arms/none.txt --served auto60=auto60 --served fA60=fA60 --served c150=c150 --rounds 2 --parallel 6 \
    --fail $B/fail.jsonl --fail-rounds 3 --fail-arm auto60 --out "$OUT/test24"
