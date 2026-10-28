#!/usr/bin/env bash
# The procedure path with Glimmer alone: Glimmer writes a procedure (v1) from C's first 20 fixed records, base Glimmer
# follows it on the 40 held-out tickets, and Glimmer sees only the tickets where that went another way than C and
# writes v2; once more for v3. The final procedure is the one with the most "both" on the held-out tickets (fixed
# before the run), and it is measured once on dev2, which the loop never sees, next to the comparison model's procedures.
# the comparison model's 20-record procedure is also run on the held-out tickets again (it was cut short in v2-fixed).
#   M=<model dir> bash scripts/v2_refine.sh runs/v2-refine
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?usage: v2_refine.sh <out dir>}
PY=.venv-serve/bin/python
mkdir -p "$OUT"
$PY -m island.gen_v2 --out data/v2 --seed 11 > "$OUT/gen.log"
REF=data/v2/records/c-train.jsonl; H20=data/v2/tasks/heldout-c20-s1.jsonl; D2=data/v2/tasks/dev2.jsonl
$PY -m island.heldout --tasks data/v2/tasks/train.jsonl --records $REF --shuffle 11:curve:1 --n 20 --out $H20
QUANT=fp8_per_channel bash scripts/serve.sh "$OUT/server.log" &
SERVER=$!
trap 'kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null || true' EXIT
for _ in $(seq 180); do curl -sf localhost:8000/v1/models > /dev/null 2>&1 && break; sleep 10; done
run() {  # name tasks procedure
    $PY -m island.run_agent --env quote-desk/2 --tasks "$2" --model muse-glimmer --temperature 0 --parallel 8 \
        --procedure "$3" --out "$OUT/$1" > "$OUT/$1.log" 2>&1
}
report() {  # out tasks ref name=folder...
    local out=$1 tasks=$2 ref=$3; shift 3
    $PY -m island.report_v2 --tasks "$tasks" --ref "$ref" --out "$out" "$@"
}
$PY -m island.extract_procedure --records $REF --env quote-desk/2 --shuffle 11:curve:1 --limit 20 --chunk 8 \
    --model muse-glimmer --out "$OUT/proc-v1.txt"
run h20-v1 $H20 "$OUT/proc-v1.txt" & run h20-pA20 $H20 baselines/v2/c20-s1.argon2.txt & wait
for r in 1 2; do
    $PY -m island.refine_procedure --procedure "$OUT/proc-v$r.txt" --run "$OUT/h20-v$r" --ref $REF --tasks $H20 \
        --env quote-desk/2 --model muse-glimmer --out "$OUT/proc-v$((r + 1)).txt"
    run h20-v$((r + 1)) $H20 "$OUT/proc-v$((r + 1)).txt"
done
report "$OUT/report-h20.json" $H20 $REF v1="$OUT/h20-v1" v2="$OUT/h20-v2" v3="$OUT/h20-v3" pA20="$OUT/h20-pA20"
BEST=$($PY -c "
import json; r = json.load(open('$OUT/report-h20.json'))['runs']
print(max(['v1', 'v2', 'v3'], key=lambda v: (r[v]['both'], r[v]['business_ok'], -int(v[1]))))")
echo "final procedure: $BEST" | tee "$OUT/final.txt"
cp "$OUT/proc-$BEST.txt" "$OUT/proc-final.txt"
run dev2-final $D2 "$OUT/proc-final.txt" & run dev2-v1 $D2 "$OUT/proc-v1.txt" &
run dev2-pA60 $D2 baselines/v2/c60-s1.argon2.txt & run dev2-pA20 $D2 baselines/v2/c20-s1.argon2.txt & wait
report "$OUT/report-dev2.json" $D2 data/v2/records/c-dev2.jsonl final="$OUT/dev2-final" v1="$OUT/dev2-v1" \
    pA60="$OUT/dev2-pA60" pA20="$OUT/dev2-pA20"
