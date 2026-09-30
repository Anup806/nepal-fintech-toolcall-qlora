"""Single source of truth: example -> chat text -> (input_ids, labels).
Training, evaluation and the API must ALL build prompts through this module."""
from functools import lru_cache
from src.data.prompts import build_system
from src.data.masking import build_labels
from src.tools import schemas
from src.tools.compact import compact_tools


@lru_cache(maxsize=1)
def tool_registry():
    """name -> compacted OpenAI-shaped tool definition (all 12)."""
    try:
        raw = schemas.to_openai_tools()
    except TypeError:
        raw = schemas.to_openai_tools(None)
    return {t["function"]["name"]: t for t in compact_tools(raw)}


def build_messages(ex):
    msgs = [
        {"role": "system", "content": build_system(ex.get("today"), ex.get("account"))},
        {"role": "user", "content": ex["user"]},
    ]
    t = ex["target"]
    if t["type"] == "call":
        msgs.append({"role": "assistant", "content": "", "tool_calls": [{
            "type": "function",
            "function": {"name": t["name"], "arguments": t["arguments"]}}]})
    else:  # ask / reply
        msgs.append({"role": "assistant", "content": t["content"]})
    return msgs


def pick_tools(ex, n_tools, rng):
    """Keep the example's own tools_shown, pad with random others, shuffle order."""
    registry = tool_registry()
    names = list(dict.fromkeys(ex["tools_shown"]))
    extras = [n for n in registry if n not in names]
    rng.shuffle(extras)
    names += extras[: max(0, n_tools - len(names))]
    rng.shuffle(names)
    return [registry[n] for n in names]


def render(ex, tokenizer, tools):
    """Return (text, input_ids, labels) with only assistant tokens supervised."""
    text = tokenizer.apply_chat_template(build_messages(ex), tools=tools, tokenize=False)
    ids, labels = build_labels(text, tokenizer)
    return text, ids, labels