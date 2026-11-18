#!/usr/bin/env python3
"""Lab 2.2 — see speculative decoding from the outside.

You can't toggle speculative decoding on serverless, but you can observe it. A draft
model (or MTP head / EAGLE) proposes k tokens; the target model verifies them in one
pass and keeps the prefix it agrees with. Acceptance depends on how *predictable* the
output is, so:

  copy task      "repeat this passage verbatim"  -> drafts are almost always right
  creative task  "write an unusual story"        -> drafts get rejected more often

If the model runs speculative decoding you should see, on the copy task:
  - higher decode TPS
  - more tokens per streamed chunk (several verified tokens arrive together)
If both tasks look the same, the model is probably not speculating (or the server
re-chunks the stream). Either way, write down what you observed and why.

Example:
  python3 labs/spec_probe.py --reps 3
  MODEL=moonshotai/Kimi-K3.1-Pro python3 labs/spec_probe.py
"""
import argparse

from common import MODEL, chat, filler, fmt, pct, save_csv

PASSAGE = ("Inference engines batch requests together so that every weight read from GPU memory "
           "produces a token for many users at once. ") * 12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=400)
    args = ap.parse_args()
    tasks = {
        "copy": f"Repeat the following passage exactly, word for word, with no commentary:\n\n{PASSAGE}",
        "creative": "Write a surreal, unpredictable short story about a lighthouse keeper who collects "
                    "forgotten passwords. Use unusual words and avoid cliches.",
        "random-words": "Continue this list with 300 more random unrelated words, comma separated: "
                        + ", ".join(filler(40, seed='w').split()[:20]),
    }
    rows = []
    print(f"model={MODEL} max_tokens={args.max_tokens} reps={args.reps}")
    for name, prompt in tasks.items():
        res = [chat([{"role": "user", "content": prompt}], max_tokens=args.max_tokens, temperature=0.7)
               for _ in range(args.reps)]
        ok = [r for r in res if r["ok"] and r.get("tps")]
        row = {"model": MODEL, "task": name, "decode_tps_p50": pct([r["tps"] for r in ok], 50),
               "tok_per_chunk_p50": pct([r["tokens_per_chunk"] for r in ok], 50),
               "completion_tokens_p50": pct([r["completion_tokens"] for r in ok], 50), "ok": len(ok)}
        rows.append(row)
        print(f"  {name:<13} decode TPS p50={fmt(row['decode_tps_p50'], '', 1):>7}  "
              f"tokens/chunk={fmt(row['tok_per_chunk_p50'], '', 2):>5}  out_tokens={row['completion_tokens_p50']}")
    c, k = rows[0]["decode_tps_p50"], rows[1]["decode_tps_p50"]
    if c and k:
        print(f"\ncopy / creative decode speed = {c / k:.2f}x  "
              f"({'consistent with speculative decoding' if c / k > 1.25 else 'no clear speculation signal'})")
    print(f"saved -> {save_csv('spec_probe', rows)}")


if __name__ == "__main__":
    main()
