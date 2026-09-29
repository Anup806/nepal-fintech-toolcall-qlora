from transformers import AutoTokenizer
from src.data.masking import build_labels

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

tools = [{"type": "function", "function": {
    "name": "get_balance",
    "description": "Get the current balance of a specific account.",
    "parameters": {"type": "object",
                   "properties": {"account_id": {"type": "string"}},
                   "required": ["account_id"]}}}]

cases = {
    "CALL": [
        {"role": "user", "content": "What's my balance in account 8821345?"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "type": "function",
            "function": {"name": "get_balance",
                         "arguments": {"account_id": "8821345"}}}]},
    ],
    "ASK": [
        {"role": "user", "content": "Send money to Sandesh"},
        {"role": "assistant", "content": "How much should I send, and from which account?"},
    ],
}

for name, msgs in cases.items():
    text = tok.apply_chat_template(msgs, tools=tools, tokenize=False)
    ids, labels = build_labels(text, tok)
    sup = [t for t in labels if t != -100]
    print(f"\n[{name}] total={len(ids)} supervised={len(sup)} ({len(sup)/len(ids):.1%})")
    print("SUPERVISED TEXT:", repr(tok.decode(sup)))
    assert sup[-1] == tok.convert_tokens_to_ids("<|im_end|>"), "must end on <|im_end|>"
    assert "<|im_start|>" not in tok.decode(sup), "leaked non-assistant tokens"
print("\nMasking OK")