#!/usr/bin/env bash
# Run workalike on your GPU with the LoRA from the release: no rules in its prompt, and every draft waits for you.
#   bash scripts/setup_gpu.sh       # once
#   bash scripts/run_lora.sh        # then open http://127.0.0.1:8770 (ssh -L 8770:localhost:8770 if the GPU is elsewhere)
# MODEL_DIR is where setup_gpu.sh put Muse Glimmer. QUANT=fp8_per_channel fits a 48 GB card; use QUANT= on 80 GB.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${MODEL_DIR:=$HOME/models/glimmer}" "${REPO:=Flyingrang2/workalike}" "${TAG:=lora-v1}" "${OUT:=runs/loop}" "${PORT:=8770}"
QUANT=${QUANT-fp8_per_channel}
A=adapters/practice-checked
PY=.venv-serve/bin/python
[ -x "$PY" ] || { echo "run scripts/setup_gpu.sh first"; exit 1; }

if [ ! -f "$A/adapter_model.safetensors" ]; then
    echo "== downloading the LoRA (419 MB)"
    mkdir -p "$A"
    for f in adapter_model.safetensors adapter_config.json SHA256SUMS; do
        curl -fL --retry 3 -o "$A/$f" "https://github.com/$REPO/releases/download/$TAG/$f"
    done
    (cd "$A" && sha256sum -c SHA256SUMS)
fi

echo "== starting the model server (a few minutes)"
M=$MODEL_DIR QUANT=$QUANT ADAPTERS="p-checked=$A" bash scripts/serve.sh "$OUT/server.log" &
SERVER=$!
trap 'kill "$SERVER" 2>/dev/null || true' EXIT
for _ in $(seq 180); do
    curl -sf localhost:8000/v1/models > /dev/null 2>&1 && break
    kill -0 "$SERVER" 2>/dev/null || { echo "the model server stopped; see $OUT/server.log"; exit 1; }
    sleep 5
done
curl -sf localhost:8000/v1/models | grep -q p-checked || { echo "the adapter did not load; see $OUT/server.log"; exit 1; }

# The sample desk, made from the generator. The person's rule book is the one the LoRA practised from; when you change a
# rule the LoRA rests and the agent reads the written rules until it has learned the change.
[ -f data/loop/cases.jsonl ] || "$PY" -m testbed.loop_cases --out data/loop/cases.jsonl
[ -f data/human/htrain.jsonl ] || "$PY" -m testbed.human_sheet --out data/human
mkdir -p "$OUT/adapter"
cp baselines/demo/adapter/status.json "$OUT/adapter/status.json"
"$PY" -m workalike --lang en --out "$OUT" --port "$PORT" --cases data/loop/cases.jsonl --tasks data/human/htrain.jsonl \
    --book baselines/test24/arms/checked-book.json --learn "$OUT/adapter"
