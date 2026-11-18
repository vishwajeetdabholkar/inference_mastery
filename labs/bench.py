#!/usr/bin/env python3
"""Lab 1.1 / 1.2 — benchmark an endpoint: TTFT, decode TPS, e2e, throughput, errors.

Two load models, because they answer different questions:
  closed loop  (--conc 1,4,8): N users, each sends the next request when the last finishes.
                               Finds the concurrency "knee" where TTFT starts climbing.
  open loop    (--rps 0.5,1,2): requests arrive on a Poisson clock no matter how slow the
                               server is. This is how real traffic behaves; queueing shows up.

Prompts get a random salt at the START by default (--cache unique) so every request pays
full prefill. Use --cache shared to measure the cached path instead.

Examples:
  python3 labs/bench.py --conc 1,4,8 --n 8
  python3 labs/bench.py --rps 0.5,1,2 --duration 30 --prompt-tokens 4000
  MODEL=Qwen/Qwen3.5-9B python3 labs/bench.py --conc 1,8 --max-tokens 512
"""
import argparse
import random
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from common import MODEL, chat, filler, fmt, pct, save_csv

SHARED_DOC = filler(100_000, seed="shared-doc")  # sliced to --prompt-tokens


def make_messages(prompt_tokens, cache_mode):
    doc = SHARED_DOC[: int(prompt_tokens * 5)]  # ~5 chars/token for this filler; usage has the truth
    salt = f"[run {uuid.uuid4().hex}] " if cache_mode == "unique" else ""
    return [{"role": "user", "content": salt + doc + "\n\nSummarize the text above in 3 sentences."}]


def summarize(label, results, wall_s):
    ok = [r for r in results if r["ok"] and r.get("ttft")]
    errs = Counter(r["status"] or "transport" for r in results if not r["ok"])
    out_tok = sum(r["completion_tokens"] or 0 for r in ok)
    row = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": MODEL, "load": label,
        "ok": len(ok), "n": len(results),
        "ttft_p50": pct([r["ttft"] for r in ok], 50), "ttft_p95": pct([r["ttft"] for r in ok], 95),
        "tps_p50": pct([r["tps"] for r in ok], 50), "e2e_p50": pct([r["e2e"] for r in ok], 50),
        "agg_out_tok_s": out_tok / wall_s if wall_s else None,
        "tok_per_chunk": pct([r["tokens_per_chunk"] for r in ok], 50),
        "cache_hit": (sum(r["cached_tokens"] for r in ok) / max(1, sum(r["prompt_tokens"] or 0 for r in ok))),
        "errors": dict(errs) or "",
    }
    print(f"{label:>12}  ok={row['ok']}/{row['n']}  TTFT p50={fmt(row['ttft_p50'],'s')} "
          f"p95={fmt(row['ttft_p95'],'s')}  TPS/req p50={fmt(row['tps_p50'],'',1)}  "
          f"e2e p50={fmt(row['e2e_p50'],'s')}  agg={fmt(row['agg_out_tok_s'],' tok/s',0)}  "
          f"cache={row['cache_hit']:.0%}" + (f"  errors={dict(errs)}" if errs else ""))
    return row


def closed_loop(conc, n, args):
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        res = list(ex.map(lambda _: chat(make_messages(args.prompt_tokens, args.cache),
                                         max_tokens=args.max_tokens), range(n)))
    return summarize(f"conc={conc}", res, time.time() - t0)


def open_loop(rps, args):
    results, lock, threads = [], threading.Lock(), []

    def fire():
        r = chat(make_messages(args.prompt_tokens, args.cache), max_tokens=args.max_tokens)
        with lock:
            results.append(r)

    t0 = time.time()
    while time.time() - t0 < args.duration:
        t = threading.Thread(target=fire)
        t.start()
        threads.append(t)
        time.sleep(random.expovariate(rps))  # Poisson arrivals
    for t in threads:
        t.join()
    return summarize(f"rps={rps}", results, time.time() - t0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conc", default="", help="closed-loop concurrency levels, e.g. 1,4,8")
    ap.add_argument("--rps", default="", help="open-loop arrival rates, e.g. 0.5,1,2")
    ap.add_argument("--n", type=int, default=8, help="requests per closed-loop level")
    ap.add_argument("--duration", type=float, default=30, help="seconds per open-loop level")
    ap.add_argument("--prompt-tokens", type=int, default=2000)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--cache", choices=["unique", "shared"], default="unique")
    args = ap.parse_args()
    if not args.conc and not args.rps:
        args.conc = "1,4,8"
    print(f"model={MODEL} prompt~{args.prompt_tokens} tok max_tokens={args.max_tokens} cache={args.cache}")
    chat(make_messages(200, "unique"), max_tokens=8)  # warm the connection, not the cache
    rows = [closed_loop(int(c), args.n, args) for c in args.conc.split(",") if c]
    rows += [open_loop(float(r), args) for r in args.rps.split(",") if r]
    print(f"saved -> {save_csv('bench', rows)}")


if __name__ == "__main__":
    main()
