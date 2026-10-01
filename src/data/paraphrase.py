"""Prompt building, parsing and validation for LLM-written user messages."""
import json
import re

STYLE_GUIDE = {
    "plain": "clear everyday English, one natural sentence",
    "casual": "informal texting English: lowercase is fine, light slang or abbreviations (pls, plz, bro), maybe one small typo",
    "romanized": "Nepali written in English letters (Roman Nepali) the way people really chat, e.g. 'mero', 'kati cha', 'dekhaau', 'pathaideu', 'lai', 'ko'; mostly Nepali with some English words",
    "mixed": "English mixed with a few Nepali words (code-switching), natural for a Kathmandu user",
    "terse": "extremely short, 2-6 words, keyword style, no politeness",
    "formal": "polite, complete, formal English",
}

NUMBER_WORDS = re.compile(r"\b(hundred|thousand|lakhs?|hajar|hazar|crore)\b", re.I)
COMPARATORS = re.compile(r"\b(over|above|more than|greater than|below|under|less than|exceeding|exceeds?)\b", re.I)


def build_prompt(items, n_each):
    styles = "\n".join(f"- {k}: {v}" for k, v in STYLE_GUIDE.items())
    lines = "\n".join(json.dumps({
        "id": it["sid"],
        "style": it["style"],
        "what_the_user_wants": it["intent"],
        "must_appear_verbatim": it["must"],
        "messages_needed": n_each[it["sid"]],
    }, ensure_ascii=False) for it in items)
    return (
        "You write realistic chat messages from customers of a Nepali mobile-banking / digital-wallet app, "
        "typed to the app's chat assistant. For each item below, write the requested number of DIFFERENT "
        "messages the customer might send.\n\n"
        "HARD RULES\n"
        "1. Every string in must_appear_verbatim has to appear in EVERY message exactly as written (same spelling and digits).\n"
        "2. Do not mention any number, amount, date, account, card, phone or person that is not in must_appear_verbatim. Never invent details.\n"
        "3. Do not mention account numbers unless they are in must_appear_verbatim; the app already knows the user's own account.\n"
        "4. Say only what what_the_user_wants says. Do not add extra requests, and do not supply information the description says is NOT given.\n"
        "5. If an item mentions a minimum or maximum amount, use words like 'at least', 'minimum', 'or more' / 'at most', 'maximum', 'or less'. Never use over, above, more than, below, under or less than.\n"
        "6. One line per message, under 200 characters, no quotation marks around the whole message, no emojis.\n"
        "7. Messages for the same item must differ in wording and structure, not only punctuation.\n\n"
        f"STYLES (each item has one)\n{styles}\n\n"
        f"ITEMS (one JSON object per line)\n{lines}\n\n"
        'OUTPUT: only a JSON object mapping each id to a list of messages, e.g. {"s0001": ["...", "..."], "s0002": ["..."]}. '
        "No markdown, no commentary."
    )


def parse_json(text):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in response")
    data = json.loads(text[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("JSON is not an object")
    return data


def _norm(s):
    return re.sub(r"\s+", " ", s.strip()).lower()


def validate(utt, s):
    """Return None if the message is acceptable, else a short rejection reason."""
    if not isinstance(utt, str):
        return "not_string"
    u = re.sub(r"\s+", " ", utt.strip().strip('"').strip())
    if not (3 <= len(u) <= 220):
        return "length"
    if re.search(r"[{}\[\]]", u):
        return "markup"
    for m in s["must"]:
        if _norm(m) not in _norm(u):
            return "missing_fact"
    must_text = " ".join(s["must"])
    allowed = {t.replace(",", "") for t in re.findall(r"\d[\d,]*", must_text)}
    allowed |= set(re.findall(r"\d+", must_text))
    for run in re.findall(r"\d+", u.replace(",", "")):
        if run not in allowed:
            return "stray_number"
    for m in NUMBER_WORDS.finditer(u):
        if m.group(0).lower() not in must_text.lower():
            return "number_word"
    if s.get("ban_comparators") and COMPARATORS.search(u):
        return "comparator"
    return None


if __name__ == "__main__":
    s = {"must": ["Sandesh", "Rs. 1,500"], "ban_comparators": False}
    assert validate("Send Rs. 1,500 to Sandesh", s) is None
    assert validate("send rs. 1,500 to sandesh pls", s) is None
    assert validate("Send 1500 to Sandesh", s) == "missing_fact"
    assert validate("Send Rs. 1,500 to Sandesh from 998877", s) == "stray_number"
    assert validate("Send Rs. 1,500 to Sandesh, two thousand total", s) == "number_word"
    s2 = {"must": ["5k"], "ban_comparators": True}
    assert validate("transactions above 5k", s2) == "comparator"
    assert validate("transactions of at least 5k", s2) is None
    assert validate("hi", s2) == "length"
    print("paraphrase self-test OK")