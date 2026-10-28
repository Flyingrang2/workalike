#!/usr/bin/env bash
# The written-procedure baseline against LoRA on worker C's records (quote-desk/2), on one server:
#   dev 12        base Glimmer with the procedure Argon wrote from C's first 20, 40, 60 records (procA*) and with
#                 the procedure Glimmer wrote from the same records (procG*)
#   held-out      the training tickets the seed-1 adapters never saw (40 for c20s1, 20 for c40s1): the adapter,
#                 both procedures from the same N records, and base Glimmer (on the 40)
# Needs data/v2 from gen_v2 and adapters/v2-c20-s1, v2-c40-s1 from the v2-curve2 job. Run from the repo root.
#   M=<model dir> bash scripts/v2_proc.sh runs/v2-proc
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?usage: v2_proc.sh <out dir>}
PY=.venv-serve/bin/python
mkdir -p "$OUT" baselines/v2
for n in 20 40; do
    $PY -m island.heldout --tasks data/v2/tasks/train.jsonl --records data/v2/records/c-train.jsonl \
        --shuffle 11:curve:1 --n $n --out data/v2/tasks/heldout-c$n-s1.jsonl
done
run() {  # name tasks model [procedure]
    $PY -m island.run_agent --env quote-desk/2 --tasks "$2" --model "$3" --temperature 0 ${4:+--procedure "$4"} \
        --out "$OUT/$1" > "$OUT/$1.log" 2>&1
}
AFTER=$(cat <<'EOF'
set -e
PY=.venv-serve/bin/python; OUT=$OUT_DIR
for n in 20 40 60; do
    $PY -m island.extract_procedure --records data/v2/records/c-train.jsonl --env quote-desk/2 --shuffle 11:curve:1 \
        --limit $n --chunk 8 --model muse-glimmer --out baselines/v2/c$n-s1.glimmer.txt
done
EOF
)
export OUT_DIR=$OUT
# The procedure runs on dev go through serve_runs; everything after needs the Glimmer procedures, so it runs in AFTER.
ENV=quote-desk/2 TASKS=data/v2/tasks/dev.jsonl QUANT=fp8_per_channel MAX_LORAS=2 \
ADAPTERS="c20s1=adapters/v2-c20-s1 c40s1=adapters/v2-c40-s1" \
RUNS="procA20=muse-glimmer+procedure:baselines/v2/c20-s1.argon.txt procA40=muse-glimmer+procedure:baselines/v2/c40-s1.argon.txt procA60=muse-glimmer+procedure:baselines/v2/c60-s1.argon.txt" \
AFTER="$AFTER
$(declare -f run)
H20=data/v2/tasks/heldout-c20-s1.jsonl; H40=data/v2/tasks/heldout-c40-s1.jsonl; DEV=data/v2/tasks/dev.jsonl
run procG20 \$DEV muse-glimmer baselines/v2/c20-s1.glimmer.txt & run procG40 \$DEV muse-glimmer baselines/v2/c40-s1.glimmer.txt &
run procG60 \$DEV muse-glimmer baselines/v2/c60-s1.glimmer.txt &
run h20-c20s1 \$H20 c20s1 & run h20-procA20 \$H20 muse-glimmer baselines/v2/c20-s1.argon.txt &
run h20-procG20 \$H20 muse-glimmer baselines/v2/c20-s1.glimmer.txt & run h20-base \$H20 muse-glimmer &
run h40-c40s1 \$H40 c40s1 & run h40-procA40 \$H40 muse-glimmer baselines/v2/c40-s1.argon.txt &
run h40-procG40 \$H40 muse-glimmer baselines/v2/c40-s1.glimmer.txt &
wait" \
bash scripts/serve_runs.sh "$OUT" || true
cp baselines/v2/*.glimmer.* "$OUT/" 2>/dev/null || true
REF=data/v2/records/c-train.jsonl
$PY -m island.report_v2 --tasks data/v2/tasks/dev.jsonl --ref data/v2/records/c-dev.jsonl --out "$OUT/report-dev.json" \
    procA20="$OUT/procA20" procA40="$OUT/procA40" procA60="$OUT/procA60" \
    procG20="$OUT/procG20" procG40="$OUT/procG40" procG60="$OUT/procG60"
$PY -m island.report_v2 --tasks data/v2/tasks/heldout-c20-s1.jsonl --ref $REF --out "$OUT/report-h20.json" \
    c20s1="$OUT/h20-c20s1" procA20="$OUT/h20-procA20" procG20="$OUT/h20-procG20" base="$OUT/h20-base"
$PY -m island.report_v2 --tasks data/v2/tasks/heldout-c40-s1.jsonl --ref $REF --out "$OUT/report-h40.json" \
    c40s1="$OUT/h40-c40s1" procA40="$OUT/h40-procA40" procG40="$OUT/h40-procG40"
