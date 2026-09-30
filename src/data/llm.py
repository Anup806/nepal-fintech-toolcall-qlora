"""Minimal REST clients for Gemini and Groq (no SDKs -> no SDK version drift)."""
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


class LLMError(RuntimeError):
    pass


def _key(name):
    val = os.getenv(name)
    if not val:
        raise LLMError(f"{name} is not set in .env")
    return val


def _post(url, headers, payload, retries=4):
    for attempt in range(retries):
        r = requests.post(url, headers=headers, json=payload, timeout=90)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504) and attempt < retries - 1:
            time.sleep(min(60, 5 * 2 ** attempt))  # back off on rate limits
            continue
        raise LLMError(f"HTTP {r.status_code}: {r.text[:400]}")
    raise LLMError("retries exhausted")


def gemini_complete(prompt, temperature=1.0, max_tokens=4096):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    data = _post(
        url,
        {"x-goog-api-key": _key("GEMINI_API_KEY"), "Content-Type": "application/json"},
        {"contents": [{"parts": [{"text": prompt}]}],
         "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens}},
    )
    try:
        parts = data["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError):
        raise LLMError(f"Unexpected Gemini response: {str(data)[:300]}")
    return "".join(p.get("text", "") for p in parts if not p.get("thought"))


def groq_complete(prompt, temperature=1.0, max_tokens=4096):
    data = _post(
        "https://api.groq.com/openai/v1/chat/completions",
        {"Authorization": f"Bearer {_key('GROQ_API_KEY')}", "Content-Type": "application/json"},
        {"model": GROQ_MODEL,
         "messages": [{"role": "user", "content": prompt}],
         "temperature": temperature,
         "max_completion_tokens": max_tokens},
    )
    try:
        return data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError):
        raise LLMError(f"Unexpected Groq response: {str(data)[:300]}")