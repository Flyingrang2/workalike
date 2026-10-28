#!/usr/bin/env bash
# Serve the base model and its adapters with vLLM on port 8000 until stopped, or for HOURS hours.
#   M=<model dir> QUANT=fp8_per_channel ADAPTERS="a=adapters/a b=adapters/b" HOURS=1 bash scripts/serve.sh runs/server.log
# QUANT is empty for BF16 (80 GB cards); on a 48 GB card use an online FP8 mode.
# MAX_LORA_RANK (default 16) must be at least the largest adapter rank.
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=${1:?usage: serve.sh <log file>}
: "${M:?model dir}" "${ADAPTERS:=}" "${QUANT:=}" "${MAX_LEN:=32768}" "${HOURS:=}"
mkdir -p "$(dirname "$LOG")"
# The box has no system CUDA toolkit; the pip one has nvcc for FlashInfer's JIT kernels.
CU=$(ls -d "$PWD"/.venv-serve/lib/python3*/site-packages/nvidia/cu1* 2>/dev/null | head -1 || true)
[ -x "$CU/bin/nvcc" ] && export CUDA_HOME="$CU" PATH="$CU/bin:$PATH"
# FlashInfer's sampler is the first kernel that needs JIT; greedy decoding does not need it.
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}

LORA=()
if [ -n "$ADAPTERS" ]; then
    # vLLM puts one adapter per batch by default; allow all of them so runs can share batches.
    # MAX_LORAS below the count keeps that many on the GPU and swaps the rest in from CPU memory.
    LORA=(--enable-lora --max-lora-rank "${MAX_LORA_RANK:-16}" --max-loras "${MAX_LORAS:-$(wc -w <<< "$ADAPTERS")}"
          --max-cpu-loras "$(wc -w <<< "$ADAPTERS")" --lora-modules $ADAPTERS)
fi
exec ${HOURS:+timeout "${HOURS}h"} .venv-serve/bin/vllm serve "$M" --served-model-name muse-glimmer \
    --max-model-len "$MAX_LEN" --max-num-seqs 8 ${QUANT:+--quantization "$QUANT"} --gpu-memory-utilization 0.92 \
    --enable-auto-tool-choice --tool-call-parser muse_glimmer --reasoning-parser muse_glimmer \
    "${LORA[@]}" > "$LOG" 2>&1
