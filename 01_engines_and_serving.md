# 01 — Inference Engines & Serving

**Goal:** understand what happens between "request in" and "token out", measure it honestly,
and know which engine and which knobs to reach for given a workload.

**Deliverables for this module**
1. A `labs/results/bench.csv` with a closed-loop sweep and an open-loop sweep on two models.
2. A one-paragraph written answer: "Where is this endpoint's concurrency knee, and why there?"
3. A context-ladder table showing TTFT vs prompt length, with prefill tok/s.
4. A one-page engine-selection cheat sheet in your own words (§4).

---

## 1. The request lifecycle

```
client
  → edge proxy           (TLS, auth at the edge, rate limits, retries, region routing)
  → router               (pick a replica: cache-aware / session-aware / least-loaded; admission)
  → gateway             (chat template, tokenization, tool/reasoning parsers, streaming)
  → engine scheduler     (waiting queue → running batch; preemption when KV runs out)
      → PREFILL          (process the whole prompt in parallel, write its KV cache)
      → DECODE loop      (one token per step per sequence, reading all weights + its KV)
  → gateway              (detokenize, parse reasoning / tool calls, SSE chunks)
  → client
```

Each hop can fail or add latency on its own, which is why a client-side TTFT alert can fire
while the engine is perfectly healthy (see case study 2 and the TTFT ladder in module 04).
A typical production stack: an edge proxy (regions, TLS, rate limits) → an in-datacenter
router (replica selection + admission) → a gateway in front of each engine (template,
tokenize, parse) → the engine (vLLM / SGLang / TRT-LLM). Older stacks add sidecar proxies
in the pod, and those can mint their own custom error codes.

## 2. Prefill vs decode — the two workloads inside every request

| | Prefill | Decode |
|---|---|---|
| What it does | Processes all prompt tokens at once | Generates one token per step |
| Bottleneck | **Compute** (big matmuls, high arithmetic intensity) | **Memory bandwidth** (re-read all weights + KV each step) |
| Shows up as | TTFT | TPOT / ITL, decode tok/s |
| Scales with | Prompt length (≈ linear; attention part quadratic) | Batch size helps throughput, hurts per-user speed slightly |

**The arithmetic you must be able to do on a whiteboard:**

- An 8B model in BF16 is 16 GB. An H100 reads HBM at ~3.35 TB/s. Each decode step reads
  every weight once → **~200 steps/s ceiling for a single user** (16 GB ÷ 3.35 TB/s ≈ 4.8 ms),
  and the tensor cores sit mostly idle.
- **Batching** is the fix: one weight read now produces a token for *every* sequence in the
  batch. That's why throughput rises almost linearly with batch size at first.
- The catch is the **KV cache**: every running sequence keeps keys/values for all its tokens.
  Llama-3.1-8B: 2 × 32 layers × 8 KV heads × 128 dim × 2 bytes = **128 KiB per token**.
  One user at 128k context = **16 GiB**, as big as the model. Memory, not compute, caps batch size.
- **MoE models**: compute per token follows *active* params (e.g. A10B of 122B), but **all**
  experts must sit in GPU memory, and at large batch different tokens hit different experts,
  so decode reads far more than the active slice.

Run `python3 labs/kv_calc.py --preset llama-3.1-8b --gpus 1 --gpu-mem 80 --context 32000`
and explain every line of the output.

## 3. What a modern engine does with that

- **Continuous batching:** new requests join the running batch at every step instead of
  waiting for the slowest sequence. Without it, one long generation stalls everyone.
- **Paged attention:** KV lives in fixed-size blocks (pages), allocated on demand, so
  fragmentation drops from ~60–80% waste to <4% (the vLLM paper's headline).
- **Chunked prefill:** long prompts are split into chunks interleaved with decode steps, so
  a 100k-token prompt doesn't freeze every other user's token stream. The chunk budget
  (`--max-num-batched-tokens` in vLLM, chunked prefill size in SGLang) trades TTFT of the big
  request against ITL of everyone else.
- **CUDA graphs:** decode steps are tiny; launch overhead matters. Engines capture graphs
  for a set of batch sizes. A decode falling back to eager mode can be 4–6× slower (a per-pod
  ITL above ~100 ms is a useful CUDA-graph fallback alarm on large MoE fleets).
- **Scheduler + preemption:** when the KV pool fills, the scheduler preempts (evicts or swaps)
  running sequences. Preemption shows up as latency spikes and lost cache, not errors.
- **Prefix caching (radix / hash-based):** reuse the KV of a shared prompt prefix. Covered in
  depth in module 02 because it's the single biggest lever in most real workloads.

**Read the source (60 min, highly recommended):** follow one request through vLLM v1:
`vllm/v1/core/sched/scheduler.py` (`schedule()`: waiting → running, token budget, preemption)
→ `vllm/v1/core/kv_cache_manager.py` (`get_computed_blocks` = prefix hits, `allocate_slots`)
→ `vllm/v1/core/block_pool.py` (free list, eviction, block hashes). Source:
https://github.com/vllm-project/vllm. SGLang equivalents: `python/sglang/srt/managers/scheduler.py`
and `python/sglang/srt/mem_cache/radix_cache.py`.

## 4. Engine selection

| Engine | Strengths | Watch-outs | Seen in production for |
|---|---|---|---|
| **vLLM** | Broadest model support, fastest to support new archs, huge community, V1 scheduler, KV connectors (NIXL, Mooncake, CPU offload), spec decoding (EAGLE/MTP) | Many knobs; defaults rarely optimal for a specific workload | Disaggregated MoE serving with Dynamo + NIXL; a distributed KV cache (Mooncake) behind a gateway |
| **SGLang** | RadixAttention prefix cache, very strong for agentic/multi-turn, overlap scheduler, fast structured output, good DeepSeek/Qwen support | Hybrid-model knobs (Mamba state) can silently disable caching | Hybrid Qwen3.5 serving behind a gateway (case study 1) |
| **TensorRT-LLM** | Peak performance on NVIDIA, best low-precision kernels (FP8/NVFP4), mature EAGLE3 + disaggregated serving | Slower to support new architectures; build/config complexity; custom forks | Disaggregated MoE serving with EAGLE3 speculative decoding |
| **TGI** | Simple, stable, good HF integration | Smaller performance/feature velocity than vLLM/SGLang; check current maintenance status before recommending | Legacy deployments |
| **NVIDIA Dynamo** (orchestration, not an engine) | Disaggregated prefill/decode, KV-aware routing across engine pools, NIXL KV transfer | Operational complexity (pair rules, partitions, many components) | Large disaggregated MoE fleets |
| **SMG** (gateway, not an engine) | Rust gateway: templates, tokenization, tool/reasoning parsers, cache-aware routing, gRPC to engines | Parameters can be accepted and dropped between layers | In front of SGLang/vLLM engines |

**How to choose, in order:** (1) Does the engine support the model *well* today (parsers,
quantization, spec decoding)? (2) What's the workload shape: prefix-heavy agentic → SGLang/vLLM
with strong prefix cache; latency-critical on NVIDIA with a supported model → TRT-LLM;
huge prefill-heavy fleet → Dynamo PD on vLLM/TRT-LLM. (3) Can your team operate it (images,
upgrades, debugging)? "Fastest in a benchmark" loses to "we can debug it at 3 a.m."

## 5. Benchmarking fundamentals

**Metrics**

| Metric | Definition | Who cares |
|---|---|---|
| TTFT | Request sent → first token (including reasoning tokens) | Chat UX, voice, agents waiting on tools |
| TPOT / ITL | Time per output token / inter-token latency (p50, p99) | Streaming smoothness |
| Decode TPS (per request) | ≈ 1 / TPOT | "How fast does it type" |
| E2E latency | Sent → last token | Batch jobs, non-streaming clients |
| Throughput | Total output (and input) tokens/s or requests/min across the system | Capacity, cost per token |
| **Goodput** | Throughput of requests that **met the SLO** (e.g., TTFT p95 < 1 s) | The number that actually matters for sizing |
| Cache hit rate | cached tokens ÷ prompt tokens (token-weighted) | Cost and TTFT on prefix-heavy workloads |

**Rules for honest numbers**

1. **Percentiles, not averages.** Report p50 and p95/p99. A mean TTFT hides the queueing tail.
2. **Know where you measure.** Client-side TTFT includes the network and every proxy hop;
   engine-side TTFT doesn't. A TTFT alert measured at the edge proxy sees all of it, so a network
   problem looks like a slow model unless you split the two.
3. **Closed loop vs open loop.** Closed loop (N users in a loop) self-throttles: when the
   server slows, arrivals slow. Open loop (Poisson arrivals) keeps sending, so queueing and
   429/503s appear exactly like in production. Use closed loop to find the knee, open loop
   to validate an SLO.
4. **Control the cache.** Salt prompts at the *start* to defeat prefix caching, or share a
   prefix on purpose. Never compare a cached run against an uncached one.
5. **Match the workload shape.** Input/output length distributions, shared-prefix share,
   multi-turn session churn. We sized one customer on a synthetic shape (~7.3k in, 32 out,
   ~4k shared system prompt) *and* their real 7-day profile; they gave different answers.
6. **Compare like with like.** Same timeout (a 60 s vs 240 s client timeout changes success
   rates), same precision (FP8 *KV cache* is not FP8 *weights*: we caught a "BF16-optimized"
   track that only quantized the KV cache), same context length, same concurrency cap.
7. **Count tokens from usage, not chunks.** With speculative decoding one SSE chunk can carry
   several tokens; counting chunks undercounts TPS.
8. **Don't let the client be the bottleneck.** A laptop that sleeps or drops Wi-Fi voids a
   run (use `caffeinate -i` on macOS). Client inflight caps (e.g. 192) silently throttle load.

**Tools you should know:** `vllm bench serve`, SGLang `bench_serving`, NVIDIA `genai-perf` /
AIPerf, and for gateway-level load a traffic simulator with Poisson arrivals, multi-turn sessions,
and ramp + sustain stages. The labs here are deliberately tiny
versions of the same ideas.

## 6. Throughput vs latency — it depends on the workload

| Workload | Shape | Optimize for | Typical levers |
|---|---|---|---|
| Chat assistant | Short-medium in, medium out | TTFT p95 + steady TPS | Moderate batch, prefix cache on system prompt |
| Coding agent | Very long, mostly-shared prefix, many turns | Cache hit rate, TTFT | Cache-aware + session-aware routing, big KV budget, prompt ordering |
| RAG / document QA | Long unique input, short output | Prefill throughput | Chunked prefill tuning, more prefill capacity, PD disaggregation at scale |
| Long-context reasoning | Long in, long hidden reasoning | Decode TPS at batch | Spec decoding, FP8/FP4 weights, KV FP8 |
| Voice (STT/TTS/realtime) | Short, latency-critical | TTFT and *recovery time* | Warm spare capacity, pre-pulled images, low batch |
| Offline batch | Anything, deadline in hours | Throughput / cost per token | Max batch, aggressive quantization, cheapest GPU |

Real example: one long-context customer runs ~100k-token prompts with ~250-token outputs at
~95% cache hit. It's prefill-dominated, so the fleet is disaggregated with **2 prefill engines
per decode engine**, plus a distributed KV store so prefill engines can reuse each other's cache.

## 7. Engine configuration — the knobs that matter

| Knob (vLLM / SGLang) | Controls | Raise it when | Risk if wrong |
|---|---|---|---|
| `--max-num-seqs` / `--max-running-requests` | Max sequences in the running batch | GPU underutilized, KV usage low, queue building | Higher ITL; KV exhaustion → preemption |
| `--gpu-memory-utilization` / `--mem-fraction-static` | Share of HBM pre-allocated for weights + KV | KV-bound and activations fit | OOM at warmup or under long prompts |
| `--max-num-batched-tokens` / chunked prefill size | Tokens processed per step (prefill chunk budget) | TTFT high for long prompts, ITL has headroom | Decode stalls (ITL spikes) during big prefills |
| `--max-model-len` / `--context-length` | Max sequence length | Customers need longer context | Smaller effective KV pool per sequence; startup OOM |
| `--kv-cache-dtype fp8` | KV precision | KV-bound, need more concurrency or context | Small quality risk; validate |
| `--block-size` / `--page-size` | KV block granularity | Aligning with router / KV-store hash blocks (64 is common with distributed KV stores) | Lower cache hit granularity |
| `--enable-prefix-caching` (vLLM) / radix cache (SGLang default) | Reuse shared-prefix KV | Always for chat/agentic | Hybrid models need extra knobs (module 02) |
| CUDA-graph batch sizes | Which batch sizes get graphs | Decode at batch sizes outside the captured set | Memory for graphs; eager fallback is slow |
| `--speculative-config` | Draft method, tokens per step | Decode-bound at low/medium batch | Slower at high batch if acceptance is low |
| `-tp` / `--enable-expert-parallel` | Parallelism layout | Model doesn't fit / need lower latency | Communication overhead (module 02 §4) |
| Hybrid-model knobs: `--mamba-scheduler-strategy`, `--max-mamba-cache-size`, SSM state dtype | Recurrent-state memory and prefix-cache support | Qwen3.5-style hybrid models | Silent caps on concurrency; dead prefix cache |

**The lesson from production:** engines often cap you *silently*. On Qwen3.5 in SGLang, effective
max running requests = `min(max_running_requests, mamba_slots ÷ ratio)`; we configured 128 and
got 15–22 until we pinned `--max-mamba-cache-size` and halved SSM state with bf16. Always
confirm effective limits from **engine startup logs** (`max_total_num_tokens`,
`max_running_requests`, "Mamba Cache is allocated"), never from the config you submitted.

---

## Labs

### Lab 1.1 — Find the concurrency knee (closed loop)

```bash
python3 labs/bench.py --conc 1,2,4,8,16 --n 12 --prompt-tokens 2000 --max-tokens 256
MODEL=Qwen/Qwen3.5-9B python3 labs/bench.py --conc 1,2,4,8,16 --n 12
```

1. Plot (or eyeball) TTFT p50/p95 and aggregate tok/s against concurrency.
2. The **knee** is where aggregate throughput stops growing while TTFT keeps rising.
3. Write down: per-request TPS at conc 1 vs 16. Why does it drop? (shared bandwidth, bigger batch per step)
4. On shared serverless your "server" is a fleet shared with others: note how much noise
   there is between two identical runs. That variance is part of the answer.

### Lab 1.2 — Queueing under open-loop load

```bash
python3 labs/bench.py --rps 0.5,1,2,4 --duration 45 --prompt-tokens 3000
```

1. Compare TTFT p95 at each rate against Lab 1.1 at similar concurrency. Open loop exposes queueing.
2. If you see 429/503, read the printed status breakdown. 429 = admission/rate limit (you), 503 =
   capacity/availability (them). Never lump them together.
3. Re-run with `--cache shared`. TTFT should drop: you just measured the value of prefix caching.

### Lab 1.3 — Prompt length vs TTFT

```bash
python3 labs/context_ladder.py --sizes 1000,8000,32000,64000
```

1. Is TTFT roughly linear in prompt tokens? Compute prefill tok/s at each size.
2. Does decode TPS stay flat? At what length does it start to drop, and why (attention reads a bigger KV)?
3. Run the same ladder on a small model and a frontier MoE. Which one has the better prefill
   rate per dollar?

### Lab 1.4 — Read the scheduler (no API)

Using §3's file list, answer in writing: (a) where does vLLM decide which waiting request runs
next? (b) where are prefix-cache hits counted? (c) what happens when `allocate_slots` fails?

---

## Self-check

1. A customer says "the model got slower" but GPU utilization is flat. Name three causes and how
   you'd tell them apart. *(load/queueing, longer prompts, lost cache, network hop, preemption)*
2. Why can doubling `max_running_requests` *reduce* throughput? *(KV exhaustion → preemption,
   recompute; or the engine silently caps it anyway)*
3. TTFT p50 is fine but p99 is 10×. What's your first hypothesis? *(queueing or long-prompt prefill
   interleaving; check prompt-length distribution and waiting queue)*
4. Why is an open-loop test the right way to validate an SLO?
5. When would you pick TRT-LLM over vLLM, and what would make you regret it?

**Explain it out loud (2 min, no notes):** "Walk me through what happens to a 50k-token request
from the moment it arrives until the first token streams back, and where the time goes."
