# Test bed

Everything here serves the tests. The product is in `../workalike/`.

| Used by | Modules |
|---|---|
| The main test | `sheet24` (writes the sheets), `arms24` (what each column reads), `run24` (runs them through the product's path), `grade24` (the grader), `blind_review` (the blind re-read) |
| Making the LoRA | `practice` (tickets the model works itself, which become training rows) |
| The sample desk | `tasks2` (ticket generator), `human_sheet`, `loop_cases`, `worker_c` and `gen_v2` (scripted worker and its records), `propose_cards` |
| One person against the LoRA's records | `person_time` |
| Driving the product once | `loop_lap`, `assist_lap`, `assist` (the first approval window, replaced by `workalike/app.py`), `shop` |
| Earlier experiments, kept because `experiments/` and their tests use them | `gen_h1`, `tasks`, `scripted`, `score`, `score2`, `judge_h1`, `run_agent`, `report_dev`, `report_v2`, `extract_procedure`, `refine_procedure`, `next_action`, `heldout`, `probe_run`, `screen_agent`, `serve_web`, `serve_web2` |
