# Baselines

The frozen test sheets, the rules, and the results behind every public number.

Some labels on the frozen sheets are in Korean. They stay that way on purpose: each sheet was hashed before any run, and translating a label would change the hash.

| Folder | What it is |
|---|---|
| [`results/`](results/) | Every run behind the numbers, with [notes on how to read them](results/README.md). |
| [`test24b/`](test24b/) | The main test: 24 new tickets and their answers, written and hashed before any run. |
| [`test24/`](test24/) | The earlier 24 tickets. We used them to shape the method, so they are a regression check now. `arms/` holds the prompts each column ran with: `none.txt` is empty, `checked.txt` is the written rules. |
| [`v2/`](v2/) | The person's answers (`c60-cards.person.json`, `c60-cards.user-answers.json`), and the procedures from early comparisons. The answers were given in Korean; in this published copy the `ko` fields hold the English too. |
| [`demo/`](demo/) | What the desk shows when you run the sample adapter. |
| `procedure-a.txt`, `procedure-b.txt` | Written procedures from the first experiments, explained below. |

## Written procedures (first experiments)


`procedure-a.txt` and `procedure-b.txt` were written by another model (Gemini) that was given only the 150 training records of one worker, in the same form the memory baseline uses (what was seen, each tool call and its result), and asked to write the procedure that worker follows. They are used unedited: `testbed.run_agent --procedure baselines/procedure-a.txt` adds one to the system prompt of the base model. The adapters learn from the same records.

How they were made (2026-10-27): the records were rendered with `testbed.run_agent.memory_block(records, 150)`, whose heading was changed to "Records of one worker at the quote desk:", and attached under this instruction:

> The attached records show how one worker handled tickets at a quote desk: what they saw and every tool call they made, with its result. Write the procedure this worker follows, as instructions for an assistant who will handle the next tickets at the same desk with the same tools and should work the way this worker does. Cover the order of steps and how exceptions are handled, without copying particular tickets. Answer with the instructions only.

Source records (sha256, first 16 hex): `data/h1/records/a-train.jsonl` 74bfe345573ae4c8, `data/h1/records/b-train.jsonl` bd7e9c239a9225e1. No dev or eval ticket was shown, and the outputs were not edited after any run.
