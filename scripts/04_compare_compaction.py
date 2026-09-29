import json
from transformers import AutoTokenizer
from src.tools import schemas
from src.tools.compact import compact_tools

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

try:
    raw = schemas.to_openai_tools()
except TypeError:
    raw = schemas.to_openai_tools(None)
compact = compact_tools(raw)

msgs = [{"role": "user", "content": "hi"}]


def n_tokens(tools):
    return len(tok(tok.apply_chat_template(msgs, tools=tools, tokenize=False,
                                           add_generation_prompt=True))["input_ids"])


for k in (4, 12):
    a, b = n_tokens(raw[:k]), n_tokens(compact[:k])
    print(f"{k:2d} tools: raw={a:5d}  compact={b:5d}  saved={1 - b/a:.0%}")

print("\nget_transactions, compacted:")
gt = next(t for t in compact if t["function"]["name"] == "get_transactions")
print(json.dumps(gt, indent=2))