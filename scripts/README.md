# Scripts

| Script | What it does |
|---|---|
| `setup_gpu.sh` | Once on a fresh GPU box: two Python environments and Muse Glimmer. About 7 minutes on ours. |
| `run_lora.sh` | Downloads the LoRA, starts the model, opens the desk on port 8770. |
| `serve.sh` | The model server, vLLM on this machine only. Used by the others. |
| `lora_practice.sh` | Makes an adapter from your rules: practice, training, then a test on the frozen sheets. About 2 hours 12 minutes on ours. |
| `train_lora.py`, `render_sft.py` | The training steps inside it. |
| `test24.sh` | The earlier test on the first sheet. |
| `throughput.sh` | Tickets an hour, one arm alone on the server. |
| `check_numbers.py` | Recomputes the public numbers from the raw results. No GPU. |

Scripts from earlier experiments are in [`../experiments/`](../experiments/).
