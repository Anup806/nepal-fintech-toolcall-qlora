"""Compute training labels that supervise ONLY assistant turns (incl. <|im_end|>)."""
import re

# Assistant content = everything after "assistant\n" up to and including <|im_end|>.
# We include <|im_end|> so the model learns to STOP. The trailing "\n" is excluded.
ASSISTANT_RE = re.compile(r"<\|im_start\|>assistant\n(.*?<\|im_end\|>)", re.S)


def build_labels(text: str, tokenizer):
    """Return (input_ids, labels). labels = -100 everywhere except assistant spans."""
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    spans = [m.span(1) for m in ASSISTANT_RE.finditer(text)]
    input_ids = enc["input_ids"]
    labels = []
    for tid, (s, e) in zip(input_ids, enc["offset_mapping"]):
        inside = any(a <= s and e <= b for a, b in spans)
        labels.append(tid if inside else -100)
    return input_ids, labels