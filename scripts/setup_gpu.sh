#!/usr/bin/env bash
# Set up a fresh GPU box (Ubuntu, NVIDIA driver already installed) to run workalike.
#   bash scripts/setup_gpu.sh            # run from the repo root on the GPU box
# Two separate environments so vLLM's pinned torch never fights the training stack:
#   .venv-train  torch, transformers, peft, bitsandbytes, accelerate
#   .venv-serve  vllm (with its own torch), playwright for the screen entrance
# The model goes to $MODEL_DIR (default ~/models/glimmer). When the root disk is small,
# set SCRATCH to a big disk (e.g. instance storage) and every cache, the model and temp files go there:
#   SCRATCH=/mnt/scratch MODEL_DIR=/mnt/scratch/models/glimmer bash scripts/setup_gpu.sh
set -euo pipefail
cd "$(dirname "$0")/.."
: "${MODEL_DIR:=$HOME/models/glimmer}"
if [ -n "${SCRATCH:-}" ]; then
  export HF_HOME=$SCRATCH/hf UV_CACHE_DIR=$SCRATCH/uv-cache TMPDIR=$SCRATCH/tmp PLAYWRIGHT_BROWSERS_PATH=$SCRATCH/ms-playwright
  mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$TMPDIR"
fi

echo "== machine"; nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv; df -h ~ | tail -1; free -g | head -2

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "== training env"
uv venv -q --python 3.12 .venv-train
# Pinned to the versions the published adapter was trained and served with.
uv pip install -q --python .venv-train/bin/python torch==2.15.0 transformers==5.22.1 peft==0.21.2 bitsandbytes==0.50.2 accelerate==1.15.0 jinja2 huggingface_hub
.venv-train/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu ok', torch.cuda.is_available(), torch.cuda.get_device_capability())"

echo "== serving env"
uv venv -q --python 3.12 .venv-serve
uv pip install -q --python .venv-serve/bin/python vllm==0.33.0 playwright
.venv-serve/bin/python -m playwright install --with-deps chromium >/dev/null
.venv-serve/bin/python -c "import vllm; print('vllm', vllm.__version__)"

echo "== model"
mkdir -p "$(dirname "$MODEL_DIR")"
# Plain HTTP: the xet backend stalled in our runs while plain downloads stayed fast.
HF_HUB_DISABLE_XET=1 .venv-train/bin/hf download meta-models/Muse-Glimmer-30B --local-dir "$MODEL_DIR"
du -sh "$MODEL_DIR"

echo "== versions"
for env in .venv-train .venv-serve; do echo "$env"; uv pip list --python $env/bin/python 2>/dev/null | grep -E -i "^(torch|transformers|peft|bitsandbytes|accelerate|vllm) " ; done
