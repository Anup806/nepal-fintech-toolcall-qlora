import json
import random
import statistics as st

from transformers import AutoTokenizer

from src.data.render import pick_tools, render

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
with open("data/seed/seed_examples.json", encoding="utf-8") as f:
    seeds = json.load(f)
rng = random.Random(42)
IM_END = tok.convert_tokens_to_ids("<|im_end|>")

for n in (4, 8, 12):
    lens, ratios = [], []
    for ex in seeds:
        tools = pick_tools(ex, n, rng)
        names = [t["function"]["name"] for t in tools]
        assert set(ex["tools_shown"]) <= set(names), ex["id"]
        assert len(names) == max(n, len(set(ex["tools_shown"]))), ex["id"]
        if ex["target"]["type"] == "call":
            assert ex["target"]["name"] in names, ex["id"]
        text, ids, labels = render(ex, tok, tools)
        sup = [t for t in labels if t != -100]
        assert sup and sup[-1] == IM_END, ex["id"]
        lens.append(len(ids))
        ratios.append(len(sup) / len(ids))
    print(f"n_tools={n:2d}  tokens min={min(lens)} mean={st.mean(lens):.0f} "
          f"max={max(lens)}  supervised mean={st.mean(ratios):.1%}")

pos = []
for _ in range(20):
    for ex in seeds:
        if ex["target"]["type"] == "call":
            names = [t["function"]["name"] for t in pick_tools(ex, 12, rng)]
            pos.append(names.index(ex["target"]["name"]))
print("\nCorrect-tool position at 12 tools (0 = first):")
print({i: pos.count(i) for i in range(12)})
print("Goal: roughly flat. A spike at 0 means the positional leak is back.")
print("\nAll assertions passed.")