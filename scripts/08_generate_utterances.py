"""Generate user messages for every scenario. Resumable: re-run to continue."""
import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

from src.data.llm import LLMError, gemini_complete, groq_complete
from src.data.paraphrase import build_prompt, parse_json, validate
from src.data.scenarios import build_scenarios

OUT = Path("data/raw/utterances.jsonl")
SCEN = Path("data/raw/scenarios.jsonl")
BATCH = 6
GEMINI = ("gemini", gemini_complete, 4)
GROQ = ("groq", groq_complete, 20)  # (name, fn, pause seconds after each call)


def keep_n(s):
    return 2 if s["split"] == "train" else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=0, help="pilot: only the first N scenarios")
    ap.add_argument("--show", type=int, default=0, help="print N random results at the end")
    args = ap.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    all_scen = build_scenarios(args.n, seed=7)
    with SCEN.open("w", encoding="utf-8") as f:
        for s in all_scen:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    scenarios = all_scen[: args.limit] if args.limit else all_scen

    done = set()
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["sid"])

    todo = [s for s in scenarios if s["sid"] not in done]
    batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
    print(f"{len(scenarios)} scenarios, {len(done)} already done, {len(todo)} to do, {len(batches)} batches")

    rejects, by_provider, failed_batches = Counter(), Counter(), 0
    for bi, batch in enumerate(batches):
        primary, backup = (GROQ, GEMINI) if bi % 3 == 2 else (GEMINI, GROQ)
        prompt = build_prompt(batch, {s["sid"]: keep_n(s) + 1 for s in batch})
        data, used = None, None
        for name, fn, pause in (primary, backup):
            try:
                data = parse_json(fn(prompt, temperature=1.0))
                used = name
                time.sleep(pause)
                break
            except (LLMError, ValueError) as e:
                print(f"  batch {bi}: {name} failed: {str(e)[:120]}")
                time.sleep(pause)
        if data is None:
            failed_batches += 1
            continue

        kept_total = 0
        with OUT.open("a", encoding="utf-8") as f:
            for s in batch:
                got = data.get(s["sid"], [])
                seen, keep = set(), []
                for u in got if isinstance(got, list) else []:
                    reason = validate(u, s)
                    if reason:
                        rejects[reason] += 1
                        continue
                    clean = " ".join(u.split()).strip('"')
                    if clean.lower() in seen:
                        rejects["duplicate"] += 1
                        continue
                    seen.add(clean.lower())
                    keep.append(clean)
                keep = keep[: keep_n(s)]
                if keep:
                    f.write(json.dumps({"sid": s["sid"], "provider": used, "utterances": keep},
                                       ensure_ascii=False) + "\n")
                    kept_total += len(keep)
                    by_provider[used] += len(keep)
                else:
                    rejects["scenario_empty"] += 1
        print(f"batch {bi + 1}/{len(batches)} via {used}: kept {kept_total} messages")

    print("\n=== summary ===")
    print("failed batches:", failed_batches)
    print("messages kept by provider:", dict(by_provider))
    print("rejections:", dict(rejects))

    if args.show:
        by_sid = {s["sid"]: s for s in all_scen}
        rows = [json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()]
        for r in random.Random(3).sample(rows, min(args.show, len(rows))):
            s = by_sid[r["sid"]]
            print(f"\n[{r['sid']} {s['group']} style={s['style']} via {r['provider']}]")
            print("  want  :", s["intent"])
            for u in r["utterances"]:
                print("  ->", u)


if __name__ == "__main__":
    main()