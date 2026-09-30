from src.data.llm import GEMINI_MODEL, GROQ_MODEL, LLMError, gemini_complete, groq_complete

for name, model, fn in [("Gemini", GEMINI_MODEL, gemini_complete),
                        ("Groq", GROQ_MODEL, groq_complete)]:
    try:
        out = fn("Reply with exactly the word: OK")
        print(f"[{name}] {model} -> {out.strip()[:60]!r}")
    except LLMError as e:
        print(f"[{name}] {model} FAILED -> {e}")