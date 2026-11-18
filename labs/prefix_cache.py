#!/usr/bin/env python3
"""Lab 2.1 — prefix caching: prove it works, then break it on purpose.

Four experiments, each measured by cached_tokens / prompt_tokens and TTFT:

  warm      same long system prompt sent repeatedly -> 1st request cold, rest should hit
  key       same as warm, with and without prompt_cache_key (routing affinity on a
            shared fleet: without a key, repeats can land on replicas that never saw you)
  timestamp current time at the TOP of the system prompt vs at the BOTTOM
            -> top kills the cache (the earliest change invalidates everything after it)
  tools     tool definitions in random order vs sorted order
            -> unordered tools change the prefix every call

Uses a model with cached-input pricing by default (see 00_setup.md). Cache on shared
serverless is best-effort, so expect noise; the *direction* of each result is the lesson.

Examples:
  python3 labs/prefix_cache.py
  python3 labs/prefix_cache.py --only timestamp --reps 4
  MODEL=deepseek-ai/DeepSeek-V4.1-Flash python3 labs/prefix_cache.py
"""
import argparse
import random
import time
import uuid

from common import MODEL, chat, filler, fmt, pct, save_csv

TOOLS = [{"type": "function", "function": {
    "name": f"tool_{name}", "description": f"Look up {name} for an account. " + filler(60, seed=name),
    "parameters": {"type": "object", "properties": {"account_id": {"type": "string"}}, "required": ["account_id"]}}}
    for name in ["billing", "usage", "invoices", "limits", "keys", "models", "endpoints", "teams"]]


def system_prompt(run_id, tokens):
    return f"You are a support assistant (session {run_id}). Policies:\n" + filler(tokens, seed=run_id)


def run(label, build, reps, **kw):
    """build(i) -> messages; reports hit% per request so cold-vs-warm is visible."""
    res = []
    for i in range(reps):
        r = chat(build(i), max_tokens=16, **kw)
        res.append(r)
        hit = r["cached_tokens"] / r["prompt_tokens"] if r.get("prompt_tokens") else 0
        print(f"  {label:<22} req {i + 1}: prompt={r.get('prompt_tokens')} cached={r['cached_tokens']} "
              f"hit={hit:.0%} TTFT={fmt(r.get('ttft'), 's')}" + ("" if r["ok"] else f"  ERROR {r['status']}"))
    ok = [r for r in res if r["ok"]]
    warm = ok[1:] or ok  # request 1 is cold by design
    return {"model": MODEL, "experiment": label,
            "hit_after_first": sum(r["cached_tokens"] for r in warm) / max(1, sum(r["prompt_tokens"] or 0 for r in warm)),
            "ttft_first": ok[0]["ttft"] if ok else None, "ttft_rest_p50": pct([r.get("ttft") for r in warm], 50)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--prefix-tokens", type=int, default=4000)
    ap.add_argument("--only", choices=["warm", "key", "timestamp", "tools"])
    args = ap.parse_args()
    q = ["How long do refunds take?", "What is the rate limit?", "How do I rotate a key?",
         "Can I change my plan?", "Where are invoices?", "How do I add a teammate?"]
    rows = []
    print(f"model={MODEL} prefix~{args.prefix_tokens} tok, {args.reps} reps per variant\n")

    if args.only in (None, "warm"):
        rid = uuid.uuid4().hex
        print("[warm] identical long prefix, different questions")
        rows.append(run("warm", lambda i: [{"role": "system", "content": system_prompt(rid, args.prefix_tokens)},
                                           {"role": "user", "content": q[i % len(q)]}], args.reps))

    if args.only in (None, "key"):
        for use_key in (False, True):
            rid = uuid.uuid4().hex
            extra = {"prompt_cache_key": f"lab-{rid}"} if use_key else {}
            label = "key=on" if use_key else "key=off"
            print(f"[key] {label}")
            rows.append(run(label, lambda i: [{"role": "system", "content": system_prompt(rid, args.prefix_tokens)},
                                              {"role": "user", "content": q[i % len(q)]}], args.reps, **extra))

    if args.only in (None, "timestamp"):
        for where in ("top", "bottom"):
            rid = uuid.uuid4().hex

            def build(i, where=where, rid=rid):
                stamp = f"Current time: {time.strftime('%H:%M:%S')}.{time.time_ns() % 1000000}\n"
                base = system_prompt(rid, args.prefix_tokens)
                content = stamp + base if where == "top" else base + "\n" + stamp
                return [{"role": "system", "content": content}, {"role": "user", "content": q[i % len(q)]}]
            print(f"[timestamp] at {where}")
            rows.append(run(f"timestamp@{where}", build, args.reps))

    if args.only in (None, "tools"):
        for order in ("shuffled", "sorted"):
            rid = uuid.uuid4().hex

            def build(i, rid=rid):
                return [{"role": "system", "content": system_prompt(rid, args.prefix_tokens // 2)},
                        {"role": "user", "content": q[i % len(q)]}]
            print(f"[tools] {order}")
            # tools are rendered into the prompt by the chat template, so their order is part of the prefix
            hits = []
            for i in range(args.reps):
                tools = random.sample(TOOLS, len(TOOLS)) if order == "shuffled" else TOOLS
                hits.append(run(f"tools:{order}", build, 1, tools=tools, tool_choice="none"))
            rows.append({"model": MODEL, "experiment": f"tools:{order}",
                         "hit_after_first": sum(h["hit_after_first"] for h in hits[1:]) / max(1, len(hits) - 1),
                         "ttft_first": hits[0]["ttft_first"], "ttft_rest_p50": pct([h["ttft_first"] for h in hits[1:]], 50)})

    print("\nSummary (hit rate after the first, cold request):")
    for r in rows:
        print(f"  {r['experiment']:<20} hit={r['hit_after_first']:.0%}  TTFT cold={fmt(r['ttft_first'], 's')} "
              f"warm p50={fmt(r['ttft_rest_p50'], 's')}")
    print(f"saved -> {save_csv('prefix_cache', rows)}")


if __name__ == "__main__":
    main()
