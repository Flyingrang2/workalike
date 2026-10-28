#!/usr/bin/env bash
# a reviewer's experiment 0: can this LoRA setup learn C's conditional steps when each rule comes up about 25 times?
# C's 150 balanced rows (train150), 3 epochs, the E2 settings otherwise (NF4, r16, lr 2e-4, accumulation 4, seed 1).
# Measured on dev (12) and dev2 (44), with base Glimmer and fA60 (60 rows, same settings) on dev2 for reference.
#   M=<model dir> bash scripts/v2_exp0.sh runs/v2-exp0
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?usage: v2_exp0.sh <out dir>}
PY=.venv-serve/bin/python
mkdir -p "$OUT" runs/v2-train
$PY -m island.gen_v2 --out data/v2 --seed 11 > "$OUT/gen.log"
.venv-train/bin/python scripts/render_sft.py --tokenizer "$M" --data data/v2 > "$OUT/render.log"
.venv-train/bin/python scripts/train_lora.py --base "$M" --data data/v2/sft/c150-s1-train.tok.jsonl --seed 1 \
    --out adapters/v2-c150 > runs/v2-train/c150.log 2>&1
D2=data/v2/tasks/dev2.jsonl
ENV=quote-desk/2 TASKS=data/v2/tasks/dev.jsonl QUANT=fp8_per_channel PARALLEL=6 \
ADAPTERS="c150=adapters/v2-c150 fA60=adapters/v2-fA60" RUNS="c150=c150" \
AFTER="P=$PY; for m in c150 fA60 muse-glimmer; do \$P -m island.run_agent --env quote-desk/2 --tasks $D2 --model \$m --temperature 0 --parallel 6 --out $OUT/dev2-\$m > $OUT/dev2-\$m.log 2>&1 & done; wait" \
bash scripts/serve_runs.sh "$OUT" || true
$PY -m island.report_v2 --tasks data/v2/tasks/dev.jsonl --ref data/v2/records/c-dev.jsonl --out "$OUT/report-dev.json" \
    c150="$OUT/c150" || true
$PY -m island.report_v2 --tasks $D2 --ref data/v2/records/c-dev2.jsonl --out "$OUT/report-dev2.json" \
    c150="$OUT/dev2-c150" fA60="$OUT/dev2-fA60" base="$OUT/dev2-muse-glimmer"
