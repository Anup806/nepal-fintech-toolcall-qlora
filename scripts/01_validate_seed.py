import json
from pathlib import Path

from transformers import AutoTokenizer

from src.tools.schemas import TOOLS, to_openai_tools

SEED_PATH = Path("data/seed/seed_examples.json")
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def build_messages(example: dict) -> list[dict]:
    target = example["target"]
    user_msg = {"role": "user", "content": example["user"]}
    if target["type"] == "call":
        assistant_msg = {
            "role": "assistant",
            "content": f'<tool_call>\n{json.dumps({"name": target["name"], "arguments": target["arguments"]})}\n</tool_call>',
        }
    else:  # ask / reply — both are plain assistant text, no tool call
        assistant_msg = {"role": "assistant", "content": target["content"]}
    return [user_msg, assistant_msg]


def main() -> None:
    examples = json.loads(SEED_PATH.read_text())
    print(f"Loaded {len(examples)} seed examples.")

    # --- 1. Validate every CALL example's arguments against the real schema ---
    call_count, fail_count = 0, 0
    for ex in examples:
        if ex["target"]["type"] != "call":
            continue
        call_count += 1
        name = ex["target"]["name"]
        args = ex["target"]["arguments"]
        model = TOOLS[name]["args_model"]
        try:
            model(**args)
        except Exception as e:
            fail_count += 1
            print(f"  [FAIL] {ex['id']}: {e}")
    print(f"Validated {call_count} CALL examples, {fail_count} failed schema validation.")
    assert fail_count == 0, "Fix the failing examples above before continuing."

    # --- 2. Render one CALL, one ASK, one REFUSE through the real chat template ---
    tok = AutoTokenizer.from_pretrained(MODEL)
    by_type = {"call": None, "ask": None, "reply": None}
    for ex in examples:
        t = ex["target"]["type"]
        if by_type.get(t) is None:
            by_type[t] = ex

    for t, ex in by_type.items():
        messages = build_messages(ex)
        tools = to_openai_tools(ex["tools_shown"])
        rendered = tok.apply_chat_template(messages, tools=tools, tokenize=False)
        mask_result = tok.apply_chat_template(
            messages, tools=tools, tokenize=True, return_dict=True, return_assistant_tokens_mask=True
        )
        n_assistant_tokens = sum(mask_result["assistant_masks"])
        print(f"\n{'='*70}\n[{t.upper()}] example: {ex['id']}\n{'='*70}")
        print(rendered)
        print(f"--- assistant-masked tokens: {n_assistant_tokens} ---")


if __name__ == "__main__":
    main()