#!/usr/bin/env bash
# How many examples per rule does the balanced setup need, and does experiment 0 hold across seeds?
#   b50, b100   the first 50 and 100 of C's balanced 150 rows (about 8 and 17 per rule), seed 1
#   c150s2, c150s3  all 150 rows with training seeds 2 and 3 (experiment 0 was seed 1)
# Same settings as experiment 0 (3 epochs, NF4, r16, lr 2e-4, accumulation 4); measured on dev (12) and dev2 (44).
#   M=<model dir> bash scripts/v2_curve150.sh runs/v2-curve150
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?usage: v2_curve150.sh <out dir>}
PY=.venv-serve/bin/python
mkdir -p "$OUT" runs/v2-train
$PY -m island.gen_v2 --out data/v2 --seed 11 > "$OUT/gen.log"
[ -f data/v2/sft/c150-s1-train.tok.jsonl ] || .venv-train/bin/python scripts/render_sft.py --tokenizer "$M" --data data/v2 > "$OUT/render.log"
for arm in "b50 50 1" "b100 100 1" "c150s2 150 2" "c150s3 150 3"; do
    set -- $arm
    [ -f adapters/v2-$1/adapter_model.safetensors ] || .venv-train/bin/python scripts/train_lora.py --base "$M" \
        --data data/v2/sft/c150-s1-train.tok.jsonl --limit $2 --seed $3 --out adapters/v2-$1 > runs/v2-train/$1.log 2>&1
done
D2=data/v2/tasks/dev2.jsonl
ENV=quote-desk/2 TASKS=data/v2/tasks/dev.jsonl QUANT=fp8_per_channel PARALLEL=6 MAX_LORAS=4 \
ADAPTERS="b50=adapters/v2-b50 b100=adapters/v2-b100 c150s2=adapters/v2-c150s2 c150s3=adapters/v2-c150s3" \
RUNS="b50=b50 b100=b100 c150s2=c150s2 c150s3=c150s3" \
AFTER="P=$PY; for m in b50 b100 c150s2 c150s3; do \$P -m island.run_agent --env quote-desk/2 --tasks $D2 --model \$m --temperature 0 --parallel 6 --out $OUT/dev2-\$m > $OUT/dev2-\$m.log 2>&1 & done; wait" \
bash scripts/serve_runs.sh "$OUT" || true
$PY -m island.report_v2 --tasks data/v2/tasks/dev.jsonl --ref data/v2/records/c-dev.jsonl --out "$OUT/report-dev.json" \
    b50="$OUT/b50" b100="$OUT/b100" c150s2="$OUT/c150s2" c150s3="$OUT/c150s3" || true
$PY -m island.report_v2 --tasks $D2 --ref data/v2/records/c-dev2.jsonl --out "$OUT/report-dev2.json" \
    b50="$OUT/dev2-b50" b100="$OUT/dev2-b100" c150s2="$OUT/dev2-c150s2" c150s3="$OUT/dev2-c150s3"
