import random
import re
from collections import Counter

from src.data.scenarios import build_scenarios
from src.tools.schemas import TOOLS

sc = build_scenarios(720, seed=7)
print("total:", len(sc))
print("split:", dict(Counter(s["split"] for s in sc)))
print("kind :", dict(Counter(s["kind"] for s in sc)))
print("style:", dict(Counter(s["style"] for s in sc)))

# 1) every CALL target must pass the real Pydantic schema
bad = 0
for s in sc:
    t = s["target"]
    if t["type"] == "call":
        try:
            TOOLS[t["name"]]["args_model"].model_validate(t["arguments"])
        except Exception as e:
            bad += 1
            print("SCHEMA FAIL:", s["sid"], t, str(e)[:120])
print("schema failures:", bad)

# 2) invalid phones must really be invalid
for s in sc:
    if s["group"] == "topup_reply":
        assert not re.fullmatch(r"(97|98)\d{8}", s["must"][0]), s["must"][0]

# 3) the test split must cover every tool with all three kinds where they exist
def tool_of(s):
    return s["target"]["name"] if s["kind"] == "call" else (s["tools"][0] if s["tools"] else "general")

table = Counter((tool_of(s), s["kind"]) for s in sc if s["split"] == "test")
print("\nTEST split counts (tool, kind):")
for k in sorted(table):
    print(f"  {k[0]:22s} {k[1]:6s} {table[k]}")

# 4) eyeball samples
print("\nSAMPLES:")
rng = random.Random(1)
for s in rng.sample(sc, 8):
    print(f"\n[{s['sid']} {s['group']} style={s['style']} today={s['today']}]")
    print("  intent:", s["intent"])
    print("  must  :", s["must"])
    print("  target:", s["target"])