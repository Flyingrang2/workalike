# Credits

Who and what made each part. The product itself runs only on Meta's models.

| Part | Made with |
|---|---|
| The product at run time | Muse Glimmer (self-hosted) and Muse Spark Open (Baseten). No other model is called. |
| The code | Written by one person with AI coding agents, Claude Code among them. Commits carry a co-author line. |
| The person's answers | Given in Korean. Claude put them into English line for line before any run, and the LoRA practised from that English. |
| Reading drafts for the test | The automatic grader in `testbed/grade24.py`, plus a blind re-read of 98 word-based verdicts by a fresh Claude agent that could not see which arm wrote them. |
| Outside reviews | Gemini and GPT models reviewed plans, the grader and public text. Their counter-examples are in `tests/test_grade24.py`. |
| Older comparisons | `baselines/procedure-*.txt` and the `argon` files in `baselines/v2/` are procedures written by another model (Gemini 4 Argon), used as a reference in early experiments. `fix-claude.txt` is one Claude revised. Nothing in the product uses them. |
| The film's picture and the page's drawings | Code, drawn in Remotion 4 (free license for an organization of up to 3 people). Every line is a generated pencil stroke. No image model. The real screens are recordings of this product and this page. |
| The film's voice | Gemini 3.1 Flash TTS, voice Algieba. |
| The film's music | Lyria 2 on Vertex AI, solo piano. |
| The film's sound effects | Synthesized with numpy. No samples. |
| Lettering in the film and on the page | Gochi Hand by HT Fonts, SIL Open Font License 1.1. |
| Page type | Bricolage Grotesque and Instrument Sans, SIL Open Font License 1.1, from Google Fonts. |
| Tickets, customers and shops | Made up by `testbed/` generators. Any match with a real name is by chance. |
