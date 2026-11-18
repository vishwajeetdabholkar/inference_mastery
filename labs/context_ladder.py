#!/usr/bin/env python3
"""Lab 1.3 — how prompt length drives TTFT (prefill) but barely touches decode speed.

Sends unique prompts (no cache hits) at increasing lengths, 2 requests per size, and
reports real prompt_tokens (from usage), TTFT, implied prefill throughput
(prompt_tokens / TTFT) and decode TPS.

What you should see: TTFT grows roughly linearly with prompt length (prefill is
compute-bound), while decode TPS stays roughly flat until very long contexts, where
attention over a huge KV cache starts to cost (decode is memory-bandwidth-bound).

Examples:
  python3 labs/context_ladder.py
  python3 labs/context_ladder.py --sizes 1000,16000,64000,128000
"""
import argparse
import uuid

from common import MODEL, chat, filler, fmt, pct, save_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="1000,8000,32000,64000", help="target prompt tokens")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=128)
    args = ap.parse_args()
    print(f"model={MODEL} unique prompts (cache defeated), max_tokens={args.max_tokens}")
    print(f"{'target':>8} {'prompt':>8} {'TTFT p50':>9} {'prefill tok/s':>14} {'decode TPS':>11} {'cached':>7}")
    rows = []
    for size in (int(s) for s in args.sizes.split(",")):
        res = [chat([{"role": "user", "content": f"[{uuid.uuid4().hex}] " + filler(size, seed=uuid.uuid4().hex)
                      + "\n\nName three words that repeat often above."}],
                    max_tokens=args.max_tokens, timeout=900) for _ in range(args.reps)]
        ok = [r for r in res if r["ok"] and r.get("ttft")]
        if not ok:
            print(f"{size:>8}  FAILED: {[(r['status'], r.get('error', '')[:80]) for r in res]}")
            continue
        pt = pct([r["prompt_tokens"] for r in ok], 50)
        ttft = pct([r["ttft"] for r in ok], 50)
        row = {"model": MODEL, "target": size, "prompt_tokens": pt, "ttft_p50": ttft,
               "prefill_tok_s": pt / ttft if pt and ttft else None,
               "decode_tps": pct([r["tps"] for r in ok], 50),
               "cached": sum(r["cached_tokens"] for r in ok)}
        rows.append(row)
        print(f"{size:>8} {pt or '-':>8} {fmt(ttft, 's'):>9} {fmt(row['prefill_tok_s'], '', 0):>14} "
              f"{fmt(row['decode_tps'], '', 1):>11} {row['cached']:>7}")
    print(f"saved -> {save_csv('context_ladder', rows)}")
    print("Note: on shared serverless, TTFT also includes queueing. Re-run at a quiet hour and compare.")


if __name__ == "__main__":
    main()
