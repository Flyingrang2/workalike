#!/usr/bin/env bash
# C's adapters again on the fixed training rows (the screen's discount authority and what was sent are now kept),
# two arms on seed 1's order and training seed 1:
#   A  3 epochs as before (only the data fix):  fA20, fA40, fA60
#   B  8 epochs (the first LoRA adjustment):     fB20, fB60
# On one server: dev 12 for every adapter, the comparison model's procedures from the fixed records (pA20, pA60) and the
# adapter with its procedure (fA60+pA60); the trained tickets (fA60, fB60 on all 60; fA20, fB20 on their 20);
# the 40 held-out tickets for the 20-record arms (fA20, fB20, pA20, fA20+pA20).
#   M=<model dir> bash scripts/v2_fixed.sh runs/v2-fixed
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?usage: v2_fixed.sh <out dir>}
PY=.venv-serve/bin/python
mkdir -p "$OUT" runs/v2-train
rm -rf data/v2 && $PY -m island.gen_v2 --out data/v2 --seed 11
.venv-train/bin/python scripts/render_sft.py --tokenizer "$M" --data data/v2
for arm in "A 3 20" "A 3 40" "A 3 60" "B 8 20" "B 8 60"; do
    set -- $arm
    .venv-train/bin/python scripts/train_lora.py --base "$M" --data data/v2/sft/c-s1-train.tok.jsonl --limit $3 --seed 1 \
        --epochs $2 --out adapters/v2-f$1$3 > runs/v2-train/f$1$3.log 2>&1
done
for n in 20; do
    $PY -m island.heldout --tasks data/v2/tasks/train.jsonl --records data/v2/records/c-train.jsonl \
        --shuffle 11:curve:1 --n $n --out data/v2/tasks/heldout-c$n-s1.jsonl
done
$PY - <<'EOF'
import json, random
from workalike.filter import episode_meta
from workalike.record import group_episodes, read_events
order = [episode_meta(e)["task_id"] for e in group_episodes(read_events("data/v2/records/c-train.jsonl")).values()]
random.Random("11:curve:1").shuffle(order)
seen = set(order[:20])
rows = [l for l in open("data/v2/tasks/train.jsonl") if json.loads(l)["task_id"] in seen]
open("data/v2/tasks/seen-c20-s1.jsonl", "w").write("".join(rows))
EOF
run() {  # name tasks model [procedure]
    $PY -m island.run_agent --env quote-desk/2 --tasks "$2" --model "$3" --temperature 0 ${4:+--procedure "$4"} \
        --out "$OUT/$1" > "$OUT/$1.log" 2>&1
}
export OUT_DIR=$OUT
P20=baselines/v2/c20-s1.argon2.txt; P60=baselines/v2/c60-s1.argon2.txt
ENV=quote-desk/2 TASKS=data/v2/tasks/dev.jsonl QUANT=fp8_per_channel MAX_LORAS=3 \
ADAPTERS="fA20=adapters/v2-fA20 fA40=adapters/v2-fA40 fA60=adapters/v2-fA60 fB20=adapters/v2-fB20 fB60=adapters/v2-fB60" \
RUNS="fA20=fA20 fA40=fA40 fA60=fA60 fB20=fB20 fB60=fB60 pA20=muse-glimmer+procedure:$P20 pA60=muse-glimmer+procedure:$P60 fA60+pA60=fA60+procedure:$P60" \
AFTER="PY=$PY; OUT=$OUT; P20=$P20
$(declare -f run)
T=data/v2/tasks; S20=\$T/seen-c20-s1.jsonl; H20=\$T/heldout-c20-s1.jsonl
run seen-fA60 \$T/train.jsonl fA60 & run seen-fB60 \$T/train.jsonl fB60 & run seen-fA20 \$S20 fA20 & run seen-fB20 \$S20 fB20 &
run h20-fA20 \$H20 fA20 & run h20-fB20 \$H20 fB20 & run h20-pA20 \$H20 muse-glimmer \$P20 & run h20-fA20+pA20 \$H20 fA20 \$P20 &
wait" \
bash scripts/serve_runs.sh "$OUT" || true
REF=data/v2/records/c-train.jsonl
$PY -m island.report_v2 --tasks data/v2/tasks/dev.jsonl --ref data/v2/records/c-dev.jsonl --out "$OUT/report-dev.json" \
    fA20="$OUT/fA20" fA40="$OUT/fA40" fA60="$OUT/fA60" fB20="$OUT/fB20" fB60="$OUT/fB60" \
    pA20="$OUT/pA20" pA60="$OUT/pA60" fA60+pA60="$OUT/fA60+pA60" || true
$PY -m island.report_v2 --tasks data/v2/tasks/train.jsonl --ref $REF --out "$OUT/report-seen60.json" \
    fA60="$OUT/seen-fA60" fB60="$OUT/seen-fB60" || true
$PY -m island.report_v2 --tasks data/v2/tasks/seen-c20-s1.jsonl --ref $REF --out "$OUT/report-seen20.json" \
    fA20="$OUT/seen-fA20" fB20="$OUT/seen-fB20" || true
$PY -m island.report_v2 --tasks data/v2/tasks/heldout-c20-s1.jsonl --ref $REF --out "$OUT/report-h20.json" \
    fA20="$OUT/h20-fA20" fB20="$OUT/h20-fB20" pA20="$OUT/h20-pA20" fA20+pA20="$OUT/h20-fA20+pA20"
