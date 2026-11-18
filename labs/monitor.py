#!/usr/bin/env python3
"""Lab 4.1 — a synthetic monitor: the outside-in view of an endpoint, with an SLO verdict.

Sends one small streaming probe every --interval seconds for --duration seconds, logs every
probe to labs/results/monitor.csv, then evaluates the window against SLO thresholds and
prints per-minute buckets so a burst stands out against a healthy baseline.

This is what a customer sees. Dashboards inside the platform can look green while this
is red (edge/network problems, a dead replica still in rotation, an auto-resolved alert).

Examples:
  python3 labs/monitor.py --duration 300 --interval 10
  python3 labs/monitor.py --duration 600 --slo-success 0.995 --slo-ttft-p95 2.0
  MODEL=MiniMaxAI/MiniMax-M3.1-Flash python3 labs/monitor.py
"""
import argparse
import time
from collections import Counter, defaultdict

from common import MODEL, chat, fmt, pct, save_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=180)
    ap.add_argument("--interval", type=float, default=10)
    ap.add_argument("--slo-success", type=float, default=0.99)
    ap.add_argument("--slo-ttft-p95", type=float, default=3.0, help="seconds")
    args = ap.parse_args()
    print(f"model={MODEL} probing every {args.interval}s for {args.duration}s")
    rows, t_end = [], time.time() + args.duration
    while time.time() < t_end:
        start = time.time()
        r = chat([{"role": "user", "content": "Reply with the single word: pong"}], max_tokens=16, timeout=60)
        row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": MODEL, "ok": r["ok"],
               "status": r["status"] or "transport", "ttft": r.get("ttft"), "e2e": r["e2e"],
               "request_id": r["headers"].get("x-request-id", ""), "error": (r.get("error") or "")[:60].replace(",", ";")}
        rows.append(row)
        print(f"  {row['ts']} {'ok ' if r['ok'] else 'ERR'} status={row['status']} "
              f"ttft={fmt(row['ttft'], 's')} e2e={fmt(row['e2e'], 's')} {row['error']}")
        time.sleep(max(0.0, args.interval - (time.time() - start)))
    save_csv("monitor", rows)

    ok = [r for r in rows if r["ok"]]
    success = len(ok) / len(rows) if rows else 0
    p95 = pct([r["ttft"] for r in ok], 95)
    print("\nPer-minute buckets:")
    buckets = defaultdict(list)
    for r in rows:
        buckets[r["ts"][:16]].append(r)
    for minute, rs in sorted(buckets.items()):
        good = [x for x in rs if x["ok"]]
        print(f"  {minute}  n={len(rs)} success={len(good) / len(rs):.0%} ttft_p50={fmt(pct([x['ttft'] for x in good], 50), 's')}")
    print(f"\nWindow: success={success:.2%} (SLO {args.slo_success:.2%}) "
          f"ttft_p95={fmt(p95, 's')} (SLO {args.slo_ttft_p95}s) errors={dict(Counter(r['status'] for r in rows if not r['ok']))}")
    verdict = success >= args.slo_success and (p95 or 1e9) <= args.slo_ttft_p95
    print("VERDICT:", "within SLO" if verdict else "SLO BREACH: grab request ids from monitor.csv and trace them")


if __name__ == "__main__":
    main()
