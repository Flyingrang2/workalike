#!/usr/bin/env bash
# Overnight on the fixed C data (after v2_fixed.sh, which leaves the adapters fA20, fA60, fB20, fB60 on the box):
#   memory baseline  base Glimmer with the same 20 records the 20-record adapters trained on, in its prompt
#                    (dev, and the 40 held-out tickets)
#   Glimmer procedures written from the fixed records (glimmer2), on dev and the held-out tickets, to set
#                    against the comparison model's (argon2) as the plan asks
#   next action      teacher-forced next step from C's exact history, for the adapters, base, base with
#                    the comparison model's procedure, and the adapter with it (dev; trained tickets for the adapters)
#   M=<model dir> bash scripts/v2_night.sh runs/v2-night
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?usage: v2_night.sh <out dir>}
PY=.venv-serve/bin/python
mkdir -p "$OUT"
$PY - <<'EOF'
import json, random
from workalike.filter import episode_meta
from workalike.record import group_episodes, read_events
episodes = list(group_episodes(read_events("data/v2/records/c-train.jsonl")).values())
random.Random("11:curve:1").shuffle(episodes)
with open("data/v2/records/c-s1-first20.jsonl", "w") as f:
    for events in episodes[:20]:
        f.write("".join(json.dumps(e) + "\n" for e in events))
print("memory records:", [episode_meta(e)["task_id"] for e in episodes[:3]], "...")
EOF
run() {  # name tasks model [procedure]
    $PY -m island.run_agent --env quote-desk/2 --tasks "$2" --model "$3" --temperature 0 ${4:+--procedure "$4"} \
        --out "$OUT/$1" > "$OUT/$1.log" 2>&1
}
mem() {  # name tasks
    $PY -m island.run_agent --env quote-desk/2 --tasks "$2" --model muse-glimmer --temperature 0 \
        --memory data/v2/records/c-s1-first20.jsonl --memory-k 20 --out "$OUT/$1" > "$OUT/$1.log" 2>&1
}
nxt() {  # name records tasks model [procedure]
    $PY -m island.next_action --env quote-desk/2 --records "$2" --tasks "$3" --model "$4" ${5:+--procedure "$5"} \
        --out "$OUT/next-$1.json" > "$OUT/next-$1.log" 2>&1
}
T=data/v2/tasks; RT=data/v2/records/c-train.jsonl; RD=data/v2/records/c-dev.jsonl
A60=baselines/v2/c60-s1.argon2.txt
ENV=quote-desk/2 TASKS=$T/dev.jsonl QUANT=fp8_per_channel MAX_LORAS=2 MEMORY_K=20 \
ADAPTERS="fA20=adapters/v2-fA20 fA60=adapters/v2-fA60 fB20=adapters/v2-fB20 fB60=adapters/v2-fB60" \
RUNS="mem20=muse-glimmer+memory:data/v2/records/c-s1-first20.jsonl" \
AFTER="PY=$PY; OUT=$OUT; T=$T; RT=$RT; RD=$RD; A60=$A60
$(declare -f run mem nxt)
for n in 20 60; do
    \$PY -m island.extract_procedure --records \$RT --env quote-desk/2 --shuffle 11:curve:1 --limit \$n --chunk 8 \
        --model muse-glimmer --out baselines/v2/c\$n-s1.glimmer2.txt
done
cp baselines/v2/*.glimmer2.* \$OUT/
G20=baselines/v2/c20-s1.glimmer2.txt; G60=baselines/v2/c60-s1.glimmer2.txt; H20=\$T/heldout-c20-s1.jsonl
run procG20 \$T/dev.jsonl muse-glimmer \$G20 & run procG60 \$T/dev.jsonl muse-glimmer \$G60 &
run h20-procG20 \$H20 muse-glimmer \$G20 & mem h20-mem20 \$H20 &
nxt fA60-train \$RT \$T/train.jsonl fA60 & nxt fB60-train \$RT \$T/train.jsonl fB60 &
nxt fA20-seen \$RT \$T/seen-c20-s1.jsonl fA20 & nxt fB20-seen \$RT \$T/seen-c20-s1.jsonl fB20 &
nxt fA60-dev \$RD \$T/dev.jsonl fA60 & nxt fB60-dev \$RD \$T/dev.jsonl fB60 & nxt base-dev \$RD \$T/dev.jsonl muse-glimmer &
nxt pA60-dev \$RD \$T/dev.jsonl muse-glimmer \$A60 & nxt fA60+pA60-dev \$RD \$T/dev.jsonl fA60 \$A60 &
wait" \
bash scripts/serve_runs.sh "$OUT" || true
REF=data/v2/records/c-train.jsonl
$PY -m island.report_v2 --tasks $T/dev.jsonl --ref $RD --out "$OUT/report-dev.json" \
    mem20="$OUT/mem20" procG20="$OUT/procG20" procG60="$OUT/procG60" || true
$PY -m island.report_v2 --tasks $T/heldout-c20-s1.jsonl --ref $REF --out "$OUT/report-h20.json" \
    procG20="$OUT/h20-procG20" mem20="$OUT/h20-mem20" || true
for f in "$OUT"/next-*.json; do echo "== $f"; $PY -c "import json,sys; print(json.dumps(json.load(open(sys.argv[1]))['summary']))" "$f"; done
