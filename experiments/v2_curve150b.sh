#!/usr/bin/env bash
# The balanced curve again where the person experiment would sit: is 100 stable across seeds, and where is 125?
#   b100s2, b100s3  the first 100 of the balanced rows with training seeds 2 and 3 (b100 was seed 1)
#   b125            the first 125 rows (about 20 per rule), seed 1
# Same settings as experiment 0 (3 epochs, NF4, r16, lr 2e-4, accumulation 4); measured on dev (12) and dev2 (44).
#   M=<model dir> bash experiments/v2_curve150b.sh runs/v2-curve150b
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?usage: v2_curve150.sh <out dir>}
PY=.venv-serve/bin/python
mkdir -p "$OUT" runs/v2-train
$PY -m testbed.gen_v2 --out data/v2 --seed 11 > "$OUT/gen.log"
[ -f data/v2/sft/c150-s1-train.tok.jsonl ] || .venv-train/bin/python scripts/render_sft.py --tokenizer "$M" --data data/v2 > "$OUT/render.log"
for arm in "b100s2 100 2" "b100s3 100 3" "b125 125 1"; do
    set -- $arm
    [ -f adapters/v2-$1/adapter_model.safetensors ] || .venv-train/bin/python scripts/train_lora.py --base "$M" \
        --data data/v2/sft/c150-s1-train.tok.jsonl --limit $2 --seed $3 --out adapters/v2-$1 > runs/v2-train/$1.log 2>&1
done
D2=data/v2/tasks/dev2.jsonl
ENV=quote-desk/2 TASKS=data/v2/tasks/dev.jsonl QUANT=fp8_per_channel PARALLEL=6 MAX_LORAS=3 \
ADAPTERS="b100s2=adapters/v2-b100s2 b100s3=adapters/v2-b100s3 b125=adapters/v2-b125" \
RUNS="b100s2=b100s2 b100s3=b100s3 b125=b125" \
AFTER="P=$PY; for m in b100s2 b100s3 b125; do \$P -m testbed.run_agent --env quote-desk/2 --tasks $D2 --model \$m --temperature 0 --parallel 6 --out $OUT/dev2-\$m > $OUT/dev2-\$m.log 2>&1 & done; wait" \
bash experiments/serve_runs.sh "$OUT" || true
$PY -m testbed.report_v2 --tasks data/v2/tasks/dev.jsonl --ref data/v2/records/c-dev.jsonl --out "$OUT/report-dev.json" \
    b100s2="$OUT/b100s2" b100s3="$OUT/b100s3" b125="$OUT/b125" || true
$PY -m testbed.report_v2 --tasks $D2 --ref data/v2/records/c-dev2.jsonl --out "$OUT/report-dev2.json" \
    b100s2="$OUT/dev2-b100s2" b100s3="$OUT/dev2-b100s3" b125="$OUT/dev2-b125"
