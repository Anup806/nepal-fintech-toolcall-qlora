"""Measure real token lengths + supervised ratio on seed examples."""
import json
import statistics as st
from collections import defaultdict

from transformers import AutoTokenizer

from src.data.masking import build_labels
from src.tools import schemas

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SEED_PATH = "data/seed/seed_examples.json"

tok = AutoTokenizer.from_pretrained(MODEL)


def get_tools(names=None):
    """Adapter: return OpenAI-shaped tool list, in the order of `names`."""
    try:
        tools = schemas.to_openai_tools(names) if names is not None else schemas.to_openai_tools()
    except TypeError:
        tools = schemas.to_openai_tools()
    by_name = {t["function"]["name"]: t for t in tools}
    if names is None:
        return list(by_name.values())
    return [by_name[n] for n in names]


def build_messages(ex):
    msgs = [{"role": "user", "content": ex["user"]}]
    t = ex["target"]
    if t["type"] == "call":
        msgs.append({"role": "assistant", "content": "", "tool_calls": [{
            "type": "function",
            "function": {"name": t["name"], "arguments": t["arguments"]}}]})
    else:  # ask / reply
        msgs.append({"role": "assistant", "content": t["content"]})
    return msgs


def measure(ex, tool_names=None):
    tools = get_tools(tool_names)
    text = tok.apply_chat_template(build_messages(ex), tools=tools, tokenize=False)
    ids, labels = build_labels(text, tok)
    sup = sum(1 for l in labels if l != -100)
    return len(ids), sup


with open(SEED_PATH, encoding="utf-8") as f:
    seeds = json.load(f)

by_type = defaultdict(list)
lengths, worst = [], []
for ex in seeds:
    total, sup = measure(ex, ex["tools_shown"])
    total_all, _ = measure(ex, None)  # all 12 tools
    assert sup > 0, f"{ex['id']}: nothing supervised!"
    by_type[ex["target"]["type"]].append((total, sup))
    lengths.append(total)
    worst.append(total_all)

print(f"\nSeeds: {len(seeds)}")
print("\n=== Per target type ===")
for typ, rows in by_type.items():
    tot = [r[0] for r in rows]
    sup = [r[1] for r in rows]
    print(f"{typ:6s} n={len(rows):2d}  tokens mean={st.mean(tot):.0f} max={max(tot)}  "
          f"supervised mean={st.mean(sup):.0f}  ratio={sum(sup)/sum(tot):.1%}")

print("\n=== As-shown (4 tools) length histogram ===")
lo, hi, step = 0, max(lengths) + 250, 250
for b in range(lo, hi, step):
    n = sum(1 for x in lengths if b <= x < b + step)
    if n:
        print(f"{b:5d}-{b+step-1:<5d} | {'#' * n} {n}")

print("\n=== Worst case: same examples with ALL 12 tools ===")
print(f"min={min(worst)}  mean={st.mean(worst):.0f}  max={max(worst)}")
print("\nRule of thumb: pick max_seq_length above the max of the length you will actually train on.")