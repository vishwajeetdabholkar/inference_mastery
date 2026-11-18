#!/usr/bin/env python3
"""Shared helpers for every lab: config, one streaming chat call, percentiles.

stdlib only (urllib + json), so the labs run anywhere Python 3.9+ runs.
Config comes from env vars:
  TOGETHER_API_KEY   required for live calls
  MODEL              default model for all labs (default: zai-org/GLM-5.3-Flash)
  TOGETHER_BASE_URL  default https://api.together.ai/v1

Self-check (no network): python3 labs/common.py --selftest
"""
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request

BASE_URL = os.environ.get("TOGETHER_BASE_URL", "https://api.together.ai/v1").rstrip("/")
MODEL = os.environ.get("MODEL", "zai-org/GLM-5.3-Flash")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def api_key():
    key = os.environ.get("TOGETHER_API_KEY")
    if not key:
        sys.exit("Set TOGETHER_API_KEY first (see 00_setup.md).")
    return key


def cached_tokens(usage):
    """Models report cached tokens in one of two places; check both."""
    if not usage:
        return 0
    details = usage.get("prompt_tokens_details") or {}
    return details.get("cached_tokens") or usage.get("cached_tokens") or 0


def _consume(lines, t0, clock=time.time):
    """Parse an SSE stream of chat-completion chunks into one result dict.

    Counts reasoning_content as output: reasoning models stream thinking first,
    and a TTFT that ignores it reports 0 or infinity.
    """
    out = {"ttft": None, "chunks": 0, "text": "", "reasoning": "", "usage": None,
           "finish_reason": None, "tool_calls": {}}
    for raw in lines:
        line = raw.decode() if isinstance(raw, bytes) else raw
        if not line.startswith("data: "):
            continue
        payload = line[6:].strip()
        if payload == "[DONE]":
            break
        d = json.loads(payload)
        if d.get("usage"):
            out["usage"] = d["usage"]
        for c in d.get("choices", []):
            delta = c.get("delta") or {}
            got = False
            if delta.get("content"):
                out["text"] += delta["content"]
                got = True
            if delta.get("reasoning_content") or delta.get("reasoning"):
                out["reasoning"] += delta.get("reasoning_content") or delta.get("reasoning")
                got = True
            for tc in delta.get("tool_calls") or []:
                slot = out["tool_calls"].setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                fn = tc.get("function") or {}
                slot["name"] += fn.get("name") or ""
                slot["arguments"] += fn.get("arguments") or ""
                got = True
            if got:
                out["chunks"] += 1
                if out["ttft"] is None:
                    out["ttft"] = clock() - t0
            if c.get("finish_reason"):
                out["finish_reason"] = c["finish_reason"]
    out["tool_calls"] = [out["tool_calls"][k] for k in sorted(out["tool_calls"])]
    return out


def chat(messages, model=None, max_tokens=256, stream=True, timeout=300, **extra):
    """One chat completion. Never raises: errors come back as ok=False + status.

    Returns ttft/e2e (s), completion & cached tokens, decode tok/s,
    tokens-per-chunk (>1.5 hints at speculative decoding), tool calls, text,
    and any x-ratelimit-* / retry-after headers.
    """
    body = {"model": model or MODEL, "messages": messages, "max_tokens": max_tokens,
            "stream": stream, **extra}
    if stream:
        body.setdefault("stream_options", {"include_usage": True})
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions", data=json.dumps(body).encode(),
        # explicit User-Agent: the edge rejects urllib's default one with 403 "error code: 1010"
        headers={"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json",
                 "User-Agent": "inference-mastery-labs/1.0"})
    t0 = time.time()
    res = {"ok": False, "status": None, "headers": {}}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            res["status"] = r.status
            res["headers"] = {k.lower(): v for k, v in r.headers.items()
                              if k.lower().startswith(("x-ratelimit", "retry-after", "x-request-id"))}
            if stream:
                res.update(_consume(r, t0))
            else:
                d = json.loads(r.read())
                msg = d["choices"][0]["message"]
                res.update(ttft=None, chunks=1, text=msg.get("content") or "",
                           reasoning=msg.get("reasoning_content") or msg.get("reasoning") or "",
                           usage=d.get("usage"), finish_reason=d["choices"][0].get("finish_reason"),
                           tool_calls=[{"name": t["function"]["name"], "arguments": t["function"]["arguments"]}
                                       for t in msg.get("tool_calls") or []])
        res["ok"] = True
    except urllib.error.HTTPError as e:
        res["status"] = e.code
        res["headers"] = {k.lower(): v for k, v in e.headers.items()
                          if k.lower().startswith(("x-ratelimit", "retry-after", "x-request-id"))}
        res["error"] = e.read().decode(errors="replace")[:300]
    except Exception as e:  # timeouts, resets: the transport-level failures
        res["error"] = f"{type(e).__name__}: {e}"[:300]
    res["e2e"] = time.time() - t0
    usage = res.get("usage") or {}
    res["prompt_tokens"] = usage.get("prompt_tokens")
    res["completion_tokens"] = usage.get("completion_tokens")
    res["cached_tokens"] = cached_tokens(usage)
    ct, ttft = res["completion_tokens"], res.get("ttft")
    res["tps"] = (ct - 1) / (res["e2e"] - ttft) if ct and ct > 1 and ttft and res["e2e"] > ttft else None
    res["tokens_per_chunk"] = ct / res["chunks"] if ct and res.get("chunks") else None
    return res


def pct(values, p):
    """Nearest-rank percentile; fine for the small samples the labs use."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    return vals[min(len(vals) - 1, max(0, math.ceil(p / 100 * len(vals)) - 1))]


def fmt(v, unit="", nd=2):
    return "-" if v is None else f"{v:.{nd}f}{unit}"


def save_csv(name, rows):
    """Append result rows to labs/results/<name>.csv so runs can be compared over time."""
    if not rows:
        return None
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{name}.csv")
    keys = list(rows[0].keys())
    new = not os.path.exists(path)
    with open(path, "a") as f:
        if new:
            f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    return path


def filler(n_tokens, seed="x"):
    """~n_tokens of deterministic text (about 1.3 tokens per word for these words)."""
    import random
    words = ("system network cache tensor kernel packet vector matrix buffer socket thread queue "
             "shard replica router proxy engine token prompt decode prefill batch memory latency "
             "cluster node region bandwidth scheduler gateway").split()
    rng = random.Random(str(seed))
    return " ".join(rng.choice(words) for _ in range(int(n_tokens / 1.3)))


def _selftest():
    clock = iter([1.5, 1.6, 1.7, 1.8]).__next__
    lines = [
        b'data: {"choices":[{"delta":{"reasoning_content":"hmm"}}]}',
        b'data: {"choices":[{"delta":{"content":"Hel"}}]}',
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"get_weather","arguments":"{\\"ci"}}]}}]}',
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ty\\": \\"Pune\\"}"}}]},"finish_reason":"tool_calls"}]}',
        b'data: {"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":9,"prompt_tokens_details":{"cached_tokens":64}}}',
        b"data: [DONE]",
    ]
    out = _consume(lines, t0=1.0, clock=clock)
    assert abs(out["ttft"] - 0.5) < 1e-9, out["ttft"]
    assert out["chunks"] == 4 and out["text"] == "Hel" and out["reasoning"] == "hmm"
    assert json.loads(out["tool_calls"][0]["arguments"]) == {"city": "Pune"}
    assert out["finish_reason"] == "tool_calls"
    assert cached_tokens(out["usage"]) == 64
    assert cached_tokens({"cached_tokens": 7}) == 7 and cached_tokens(None) == 0
    assert pct([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50) == 5 and pct([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10
    assert pct([], 50) is None
    print("common.py selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
