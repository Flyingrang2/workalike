# Results

The raw rows behind the numbers in the top-level README and on the page. To recompute them, run `python3 scripts/check_numbers.py` from the top of the repo. It needs no GPU.

## Names

| In these files | In the tables |
|---|---|
| `none` | Muse Glimmer out of the box |
| `checked` | The same model reading the written rules on every ticket |
| `p-checked` | Muse Glimmer with the LoRA and no rules in its prompt |

A verdict is one of `ready` (a right draft), `right_hold` (rightly asked instead of quoting), `needless_hold` (asked when it did not have to), `dangerous` (a person could approve it, and it is wrong or promises what nothing supports) and `failed` (the desk's checks would not let a person approve it). "Right" in the tables is `ready` plus `right_hold`.

p95 is nearest-rank: the value at position ceil(0.95 × n) of the sorted times. Other formulas give slightly different values. The tests ran at temperature 0, so the two runs of a ticket are close to repeats.

The desk's check names are in Korean in these files, because the desk's checks speak the desk's default language:

| In the files | In English |
|---|---|
| 답장 초안 | Reply is a draft waiting for approval |
| 가격은 견적으로만 | Prices appear only in a quote |
| 원가 비공개 | The shop's cost stays private |
| 금액은 데스크 값 | Every amount came from the desk |
| 견적번호 확인 | Every quote number exists |
| 견적 Q-… | This quote: price, authority, floor, stock by ship date |

## Folders

- `test24b/`: the main test. 24 new tickets from a new seed, three arms, two runs each. `graded.jsonl` has one row per arm, run and ticket, with the tool calls, the final draft, the checks, the time and the tokens. `first-drafts.jsonl` has the first draft of every run the checks sent back.
- `test24/`: the earlier 24 tickets, in four arms: no rules (`none`), the 60 raw records in the prompt (`records`), the rules as found (`cards`), and the rules with the person's 29 answers (`checked`).
- `test24-lora/`: the LoRA and the written rules on those earlier tickets, and three failure scenarios run three times each (`fail` arm): a broken price tool, a rule that does not apply, conflicting rules.
- `throughput/`: one arm alone on the server, six at a time, wall clock in `tp.log`, from `scripts/throughput.sh`. The sheet is `test24b` again, so these scores are not a new test.
- `practice/`: how the LoRA was made. `worked.jsonl` is all 173 practice tickets with what was kept and why. `train-loss.jsonl` is the trainer's log, `train_args.json` its settings. We chose the mix of ticket kinds ourselves (`test24b/practice-summary.json`): it leans on what is rare in natural records and adds requests no rule covers, and the same people wrote the test, so the kinds line up with the test's.
- `rules/`: `records.jsonl` is the 60 worked tickets from the scripted worker, and `cards.json` is what Muse Glimmer wrote from them, unedited. Each card lists the records it cites. The citations are the model's and not all of them are good: of the 12 under the discount rule, at least 3 show no price complaint and no discount at all. It listed 30 open points; the person answered 29 and left one blank. Their answers are in `../v2/c60-cards.person.json`, in English, as Claude translated them line for line before any run.
- `product-run/`: the running desk on one GPU, from a clean install. Six requests with the LoRA, one approved and sent, then one rule changed and the same requests run again. `jobs.jsonl` shows the adapter turned off. Two requests finished both ways (`j002` and `j007`, `j005` and `j011`); the others were still running when we stopped the server and say `running`. `clean-install-before-publishing.log` is a later check of this copy from an empty folder.
- `person-time/`: the founder, timed by hand on the main test's tickets, from `testbed/person_time.py`. One practice ticket and one counted ticket, then the run stopped. The LoRA's side is its first-round row from `test24b/`. The grader is the one the test used, so a draft it did not read as a price (6.05$) counts as `needless_hold`.
- `demo-lora/`: the browser demo's two tickets, worked by the LoRA with no rules in its prompt on the same GPU, five times each, from `demo/lora_run.py`. At temperature 0 these are close to repeats.
- `learning-round/`: the product's own learning round, run once on a real GPU. It kept its adapter off.
- `demo-final/`: the browser demo's final check, six manuals on two new tickets, five times each, plus no manual.
- `serving/nine-adapters.log`: the base model and nine adapters on one GPU, and the tickets each one worked. Those adapters came from an early run with a data bug, so this shows serving only, not quality.
- `experiments.tsv`: the 33 GPU jobs queued through November 2, by name, how each ended and when. `FETCHED` means it finished and we took the results. 26 finished (`dummy-001` is a smoke test), 3 failed, 4 were cancelled. The main test is inside `lora-practice2`. The November 3 runs (`product-run/`, `throughput/`, `learning-round/`) were started by hand.

## First drafts and what a person sees

`first_verdict` grades the first draft. `verdict` grades what a person would actually see. When the desk's checks hold a draft back, the model gets one rewrite. The checks sent back 7, 6 and 7 of the 48 first drafts in `none`, `checked` and `p-checked`.

| First draft to what a person sees | `none` | `checked` | `p-checked` |
|---|---:|---:|---:|
| dangerous to `right_hold` | 1 | 0 | 3 |
| dangerous to `ready` | 1 | 0 | 0 |
| dangerous to `needless_hold` | 1 | 1 | 0 |
| dangerous to `failed` | 0 | 0 | 2 |

That is 9 of 144 rows. Three of the LoRA's first drafts named the shop's floor price: `date-2-missing` in both runs, and `discount-size-small` in run 2. No verdict got worse, but a rewrite can still go wrong inside the same verdict: in `date-1-missing`, run 2, the LoRA's first draft asked for the ship date and wrote a price outside a quote. The checks held it back, and the rewrite made a date up. It stayed dangerous, and it is the one dangerous draft in the LoRA's column.

## What changed after the main test

The desk gained two things after the main test: a check that holds back a reply naming the shop's inner limits (`inner_words`), and one reminder when a model ends in plain words (`one_nudge`). They are on in the desk and the browser demo, and off in the test runner, so the frozen tests stay as they ran. By word match, the check would have sent back 5, 8 and 6 of the final drafts counted as right, in `none`, `checked` and `p-checked`.

## The earlier sheet's raw-records arm

Reading the 60 raw records in the prompt is a strong baseline. On the earlier sheet, `records` got 46 of 48 right with 2 dangerous drafts, at 115 seconds and 92,301 tokens a ticket. `checked` got 46 with none, at 197 seconds and 32,544 tokens. We did not carry it into the main test.

## What the hashes show

`hashes.json` and `sheet.sha256` show that the sheets have not changed since those files were written. The run scripts write `sheet.sha256` before the first ticket. The order itself rests on our own commit dates and records, not on an outside timestamp.

## The blind re-read

`blind-read.json` is a second reading of the 98 verdicts in the main test that rested on words, by a fresh Claude agent that could not see which arm wrote them. `blind-items.json` is what it was shown, and `blind-key.json` says which draft was which. It agreed on 96. The two it changed were both against the arm with no rules, in `code-settled`, and it marked itself unsure. The tables use the automatic verdict, so they do not include those two changes.

## Two adapters, kept apart

| | The adapter you can download | The learning round's adapter |
|---|---|---|
| Made by | `scripts/lora_practice.sh`, from a person's answers and practice | `workalike.autolearn`, from 60 scripted records |
| Tested on | 24 new tickets, twice: 41 of 48 right, 1 dangerous | 4 fixed tickets: 3 right, 1 dangerous |
| Status | Published as a sample. It has not been through the learning round's gate. | Kept off by that gate. Not published. |

The status file says "worse than before" in Korean. That is the round's stock wording for not passing; there was no earlier adapter to compare with.

## Other notes

- **About $8** is 132 minutes (92 of practice, 40 of training) at $3.69 an hour, or $8.12. One step was rerun for 32 minutes, so the real bill was about $10. That is practice and training only, not the cost of the project.
- **The browser demo** also flagged one no-manual draft for wording: it said the shop's cost "has gone up", with no amount. The flag is a simple word match, and other drafts, some with a manual, say the same in other words. One run with a manual stopped without a draft, which is the one miss in 59 of 60.
- **The rows in the learning round:** 154 rows are 60 distinct tickets, rebalanced.
- **`test24/` and `test24-lora/`** are not the main test: we looked at them while shaping the method.
