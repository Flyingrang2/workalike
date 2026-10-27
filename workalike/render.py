"""Render training rows with Muse Glimmer's own chat template and mark what to train on.

Only the tokenizer files are needed (tokenizer.json, tokenizer_config.json,
chat_template.jinja), not the weights. Each row becomes input_ids and labels,
with loss only on the model's own turns. Rendering is checked to be prefix
stable: at every assistant turn, the prompt the server would build at inference
time is exactly the text that precedes that turn in the training example.
"""
from __future__ import annotations

GEN_PROMPT = "<|start|>assistant"
IGNORE = -100


class RenderError(Exception):
    pass


def load_tokenizer(path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path)


def render(tok, messages, tools, add_generation_prompt=False) -> str:
    return tok.apply_chat_template(messages, tools=tools, tokenize=False,
                                   add_generation_prompt=add_generation_prompt)


def segments(tok, messages, tools) -> list[tuple[str, bool]]:
    """Split the full rendering into (text, train) pieces."""
    prev = render(tok, messages[:1], tools)
    out = [(prev, False)]
    for k in range(1, len(messages)):
        cur = render(tok, messages[: k + 1], tools)
        if not cur.startswith(prev):
            raise RenderError(f"message {k} changes the text before it")
        piece = cur[len(prev):]
        if messages[k]["role"] == "assistant":
            prompt = render(tok, messages[:k], tools, add_generation_prompt=True)
            if not prompt.startswith(prev):
                raise RenderError(f"generation prompt at message {k} does not extend the history")
            head = prompt[len(prev):]
            if not piece.startswith(head):
                raise RenderError(f"assistant message {k} does not start with the generation prompt")
            out += [(head, False), (piece[len(head):], True)]
        else:
            out.append((piece, False))
        prev = cur
    return out


def tokenize(tok, messages, tools) -> tuple[dict, dict]:
    segs = segments(tok, messages, tools)
    text = "".join(t for t, _ in segs)
    spans, pos = [], 0
    for t, train in segs:
        if train:
            spans.append((pos, pos + len(t)))
        pos += len(t)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, labels, straddle = enc["input_ids"], [], 0
    for tid, (s, e) in zip(ids, enc["offset_mapping"]):
        inside = any(a <= s and e <= b and s < b for a, b in spans)
        if not inside and any(s < b and a < e for a, b in spans):
            straddle += 1
        labels.append(tid if inside else IGNORE)
    stats = {
        "tokens": len(ids),
        "trained": sum(l != IGNORE for l in labels),
        "straddle": straddle,
        "roundtrip": tok.decode(ids, skip_special_tokens=False) == text,
        "turns_end_with_eot": all(t.endswith("<|eot|>") for t, train in segs if train),
    }
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}, stats


def preview(tok, messages, tools) -> str:
    """The rendered text with trained spans wrapped in [[ ]]."""
    return "".join(f"[[{t}]]" if train else t for t, train in segments(tok, messages, tools))
