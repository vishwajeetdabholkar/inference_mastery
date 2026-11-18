# 02 — Inference Optimization

**Goal:** know the five big levers (KV cache, prefix caching, speculative decoding,
parallelism, quantization) plus disaggregation, well enough to say *which* one fixes a given
customer problem, what it costs, and how you'd prove it worked.

**Deliverables for this module**
1. `prefix_cache.csv` showing a warm hit, a key on/off comparison, and two deliberately broken caches.
2. `spec_probe.csv` on two models, with one sentence on whether each speculates.
3. The sizing worksheet (§8) solved on paper, then checked with `kv_calc.py`.
4. A P:D ratio recommendation for the worked example in §6, in your own words.

---

## 1. KV cache design

**Size it.** KV bytes/token = 2 × layers × KV heads × head_dim × bytes. Attention variants
exist mostly to shrink this number:

| Attention | KV per token | Example |
|---|---|---|
| MHA (every head has K/V) | largest | older models |
| GQA (groups of query heads share K/V) | ÷ (heads / kv_heads) | Llama-3 8B/70B: 8 KV heads → 128 / 320 KiB per token |
| MLA (compressed latent + small RoPE key) | ~1 latent per layer | DeepSeek-V3: 61 × 576 × 2 B ≈ 69 KiB per token |
| Sliding window | capped at the window | some layers of several hybrid models, Gemma |
| Linear attention / SSM (Mamba, GDN, KDA) | **fixed state per sequence**, not per token | Qwen3.5 hybrid, Kimi K3 (KDA) |

**Budget it.** KV pool = GPUs × HBM × mem_fraction − weights − activations/graphs. Divide by
bytes/token = total tokens you can hold; divide by context = concurrent sequences. When the
pool is full the scheduler **preempts**: latency spikes, cache lost, no error returned.

**Shrink it.** FP8 KV cache halves bytes/token (usually small quality impact; validate it).
Production data point: KV-FP8 on a 1×B200 Qwen3.5 deployment added ~0 throughput (the
deployment was prefill-compute-bound) but **cut cache eviction from ~20% to ~9%**. Faster? No.
Better cache retention? Yes. Know which problem you're solving.

**Tier it.** GPU HBM → CPU RAM offload → distributed KV store over RDMA (Mooncake, LMCache).
One production setup pools a large slice of every GPU's memory into a shared KV store, so a
local miss can be served from a peer's memory instead of recomputing the prefill. Over two
days, prompt tokens came ~81% from the local cache, ~15% from the remote store, ~3% computed.

**Failure mode you must know: tiers degrade silently.** In that setup a failed remote lookup
returns a 0-token hit and the request just recomputes. **Remote KV problems show up as a lower
cache hit rate and a higher TTFT, never as errors.** Watch hit rate per source, not error rate.

## 2. Prefix caching — the biggest lever in real workloads

**How it works.** The engine hashes the prompt in blocks (vLLM: hash per KV block; SGLang:
radix tree over tokens). A new request reuses KV for its **longest matching prefix** and
prefills only the rest. The first difference invalidates everything after it.

**Three layers must cooperate:**
1. **The prompt** must be byte-stable at the front: static system prompt and *sorted* tool
   definitions first, dynamic content (timestamps, user turn) last. Lab 2.1 breaks this on purpose.
2. **The router** must send the same prefix to the replica that holds it. Production routers use
   cache-aware-by-hash + session affinity. Under load they deliberately move traffic off
   "hot" workers (routing reason `cache_miss_all_hotspots`): load balance vs hit rate is a
   tunable trade-off, not a bug.
3. **The engine** must actually retain and reuse the KV. Not guaranteed (see below).

**Hybrid models break the naive picture.** For Mamba/linear-attention layers there is no
per-token KV to reuse. The engine must keep snapshots of the recurrent state at prefix
boundaries. On Qwen3.5 in SGLang, the default `--mamba-scheduler-strategy no_buffer` keeps no
such snapshots, so **prefix caching hit 0% on every stock config, including the certified one**.
`--mamba-scheduler-strategy extra_buffer` allocates buffers for the state snapshots (and enables
the overlap scheduler and radix cache with speculative paths). Result: 0% → ~98% hit on repeats,
TTFT on a 100k-token prompt 16.5 s → 4.1 s, decode 161 → 196 tok/s.
vLLM's equivalent idea: `--prefix-cache-retention-interval` keeps resumable checkpoints for
sliding-window/Mamba groups only every N tokens, so **a "hit" can resume up to N tokens short
of the true shared prefix**.

**Measure it right.** Token-weighted: Σ cached_tokens ÷ Σ prompt_tokens, over successful
requests only. Per-request averages overweight tiny requests. Enable cache reporting in the
engine (`--enable-cache-report` in SGLang) or you'll see `cached_tokens=0` and wrongly conclude
caching is broken.

**Cache economics change comparisons.** A customer's old model showed 127k input tok/s on
4×H100. That was ~90% cache-hit economics on standard attention, where a cached prefix is nearly
free. Cache-normalized, the new hybrid model did ~27k *uncached* tok/s on 2 GPUs vs ~13k on 4.
Always compare uncached throughput, or the cache rate hides the real result.

## 3. Speculative decoding

**Idea:** decode is bandwidth-bound, so the GPU has spare compute. A cheap proposer drafts k
tokens; the target model verifies all k in **one** forward pass and keeps the longest prefix it
agrees with (plus one corrected token). With correct rejection sampling, the output distribution
equals the target model's: it's lossless.

| Proposer | How | Notes |
|---|---|---|
| Draft model | Small model of the same family | Simple; needs a matching tokenizer |
| EAGLE / EAGLE3 | Light head on the target's hidden states | High acceptance; common in TRT-LLM and vLLM |
| MTP heads | Multi-token-prediction heads trained with the model (DeepSeek-V3 and others) | No separate model to serve |
| n-gram / prompt lookup | Copy spans from the prompt | Great for edits, RAG, code; free |
| Model-specific drafters (e.g. DFlash) | Custom draft models/kernels per model family | Check acceptance per workload |

**What decides the win: acceptance rate × cost per step.** One large MoE in production: 5 draft tokens,
~3.1 accepted per step, so each streamed chunk carries ~3 tokens. Gains shrink at high batch
(compute is no longer free) and with unpredictable output. That's why configs scale draft length
with batch size. One production config used 2 spec tokens at batch 1–4 and 3 at batch 5–32.

**Spec decoding interacts with everything.** A real crash: drafts were zero-padded to a fixed
length for CUDA-graph capture, and the **guided-decoding (structured output) bitmask path tripped
over the padded positions inside the EAGLE3 loop**, crash-looping prefill pods only. Upstream had
already fixed it, and the fix was a backport (case study 4). When spec decoding is on, test it
with tool calling / JSON mode / long context, not just plain chat.

## 4. Parallelism

| Strategy | Splits | Communication | Use when | Production example |
|---|---|---|---|---|
| **TP** (tensor) | Each weight matrix across GPUs | All-reduce every layer (needs NVLink) | Model doesn't fit one GPU, or you need lower latency | TP8 inside one 8-GPU node |
| **PP** (pipeline) | Layers across GPUs/nodes | Activations between stages; pipeline bubbles | Across nodes without fast interconnect | Rare for latency-sensitive serving |
| **EP** (expert) | MoE experts across GPUs | All-to-all dispatch/combine per MoE layer | Large MoE; pairs with TP or DP-attention | TP8 + EP on a large MoE |
| **DP** (data) | Whole replicas | None between replicas | Scale throughput; the default unit of scaling | Every fleet |
| **DP-attention** | Attention by DP, MoE by EP | All-to-all | MLA models (latent KV doesn't shard across TP heads) | DeepSeek-style serving |
| **Context parallel** | Sequence/KV across GPUs | Ring/all-to-all on attention | Very long context decode | Long-context MoE decode |
| **Multi-node TP** | TP across nodes over NVLink fabric | MNNVL all-gather / reduce-scatter | Rack-scale systems (GB300 NVL72) | TP8 across two 4-GPU rack-scale nodes |

**Same GPUs, different trade-off (real data):** on 2×B200, two TP1 replicas served ~980 req/min
but at **TTFT p50 8.7 s**, while one TP2 replica served 814 req/min at **0.9 s**. TP1×N wins raw
requests per GPU; TP2 wins latency. The customer's SLO decides, not the benchmark.

**Pair rule (operational consequence of multi-node TP):** if one engine spans a leader pod and
a follower pod, losing *either* takes the whole engine down, and rolling one without the other
leaves a stale half. Treat the pair as the unit of scheduling, rollout and alerting.

## 5. Quantization

| Format | What's quantized | Wins | Notes |
|---|---|---|---|
| FP8 (E4M3) weights (+activations, W8A8) | Weights, often activations | ~2× less memory and bandwidth vs BF16; FP8 tensor cores on Hopper+ | Default choice for serving large models today |
| NVFP4 / FP4 | Weights (+activations) | ~4× vs BF16; Blackwell FP4 tensor cores | Common for frontier MoEs on Blackwell |
| MXFP4 | Weights | Similar to FP4 | gpt-oss ships this way |
| INT4 AWQ / GPTQ | Weights only | Big memory win for bandwidth-bound decode | Compute still in 16-bit; weaker at high batch |
| KV-cache FP8 | KV cache | 2× KV capacity → concurrency/context, better retention | Validate long-context quality |

**What quantization buys you:** less memory (fit on fewer GPUs, bigger KV pool), less bandwidth
(faster decode), and on Blackwell more FLOPs (faster prefill). It doesn't fix a scheduler or
cache problem. The Qwen3.5 FP8 model fits a single B200 (~123 GB weights) with ~28 GB of KV
headroom, enough for a working deployment but not for high concurrency at long context.

**Validate quality every time:** task evals for the use case, tool-calling conformance (we run a
57-test function-calling suite on every config variant), long-context retrieval, and a few
golden prompts compared side by side. "Same speed, quietly worse answers" is the failure mode.

## 6. Prefill/decode (P:D) disaggregation

**Why:** prefill (compute-heavy, bursty) and decode (bandwidth-heavy, steady) interfere on the
same GPU. A big prefill stalls everyone's decode (ITL spikes). Splitting them into separate
engine pools lets you size and tune each for its own bottleneck.

**How (NVIDIA Dynamo pattern):** frontend with a KV-aware router → pick the prefill engine
with the most cached prefix blocks → prefill → transfer KV to a decode engine over RDMA (NIXL;
measured p50/p95 ~17–35 / 40–64 ms) → decode streams tokens. Optionally, a shared KV store
across prefill engines (Mooncake) raises the cache hit rate further (~95% vs ~78% without it,
in one pair of fleets).

**When it pays off:** long prompts and high scale (prefill-dominated workloads), strict ITL SLOs,
very large models. When it doesn't: short prompts, small fleets, teams without the operational
maturity. It adds components (frontends, KV transfer, partitions, pair rules), and each one is
a new failure mode: requests stuck in a frontend queue, NIXL transfer failure loops, decode
rollouts that drop sessions.

**Setting the P:D ratio (worked example):** prompts p50 ~100k tokens, output p50 ~250–280
tokens, ~95% cache hit, ~35 RPS.
- Uncached prefill per request ≈ 5% × 100k = 5k tokens of real compute, and **every** request
  still pays KV transfer and lookups.
- Decode per request is short (~280 tokens) but must stream smoothly at batch.
- Production settled at roughly **2 prefill : 1 decode** on one hardware generation. On a
  different generation with no shared KV store (more recompute), the ratio was rebalanced from
  1:1 to roughly 1:2 within days as real traffic arrived.

The method: measure prefill queue time vs decode running batch per engine, then move capacity to
whichever side is queueing. The ratio is a measured outcome, not a formula.

## 7. MoE kernel tuning (the free win people miss)

Fused MoE kernels (Triton) load a tuned config keyed by expert shape, GPU, dtype, block shape
and Triton version. If the file is missing, the engine logs `Using default MoE kernel config.
Performance might be sub-optimal!` and leaves real throughput on the table. Fix: run the
engine's tuning script (`benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` in SGLang,
`benchmarks/kernels/benchmark_moe.py` in vLLM) on the **same GPU, TP size and dtype**, ship the
JSON with the deployment, confirm `Using configuration from … for MoE layer` at startup, and
A/B throughput. Re-tune on any GPU / dtype / Triton / model-revision change.

## 8. Sizing worksheet (solve on paper, then check with `kv_calc.py`)

Assume mem_fraction 0.9 and ignore activations. Answers are at the end.

1. Llama-3.3-70B on 2×H100 (80 GB), 8k context. BF16 weights? FP8 weights? FP8 weights + FP8 KV?
2. Qwen3-32B on 1×H100, 32k context. BF16? FP8?
3. DeepSeek-V3 (671B, MLA) FP8 on 8×H200 (141 GB) at 128k context. Then on 8×H100.
4. Llama-3.1-8B on 1×H100 at 128k context: how many users?
5. A hybrid model holds ~144 MB of recurrent state per running sequence (fp32). With 44 GB left
   after weights, how many sequences fit by state alone? What happens with bf16 state?

<details><summary>Answers</summary>

1. BF16: weights 141 GB, KV budget ~2.8 GB → **1 sequence**: it fits on paper but is useless.
   FP8 weights: 73 GB budget → ~224k tokens → **~27 sequences**. + FP8 KV → **~54**.
2. BF16: 65.6 GB weights, 6.4 GB budget, can't hold even one 32k sequence (8.6 GB) → **0**.
   FP8: 39 GB budget → **4** sequences at 32k. Long context is a KV problem, not a weights problem.
3. 8×H200: 671 GB weights, ~344 GB budget, ~69 KiB/token → ~4.9M tokens → **~37 sequences**
   at 128k on paper. Caveat: under pure TP, MLA's latent KV is *replicated* on every rank (it
   can't be sharded by heads), so real capacity is far lower. That's why DP-attention exists.
   8×H100: weights exceed memory → **doesn't fit**; use more nodes or FP4.
4. 128 KiB × 131,072 = 17.2 GB per user; 56 GB budget → **3 users**. This is why long-context
   products are expensive.
5. 44 GB ÷ 0.144 GB ≈ 305 slots; but engines with snapshot buffers need several slots per
   running request (ratio 4–5 on Qwen3.5) → ~60–75 running. bf16 state (~73 MB) roughly
   doubles it. This exact math explained why a config asking for 128 running requests got 15–22.
</details>

---

## Labs

### Lab 2.1 — Prefix caching, working and broken

```bash
MODEL=deepseek-ai/DeepSeek-V4.1-Flash python3 labs/prefix_cache.py --reps 4
MODEL=zai-org/GLM-5.3-Flash          python3 labs/prefix_cache.py --reps 4
MODEL=Qwen/Qwen3.5-9B                python3 labs/prefix_cache.py --only warm   # control: no cached pricing
```

1. `warm`: request 1 should be cold, requests 2+ should hit. What hit % do you get? Why not 100%?
   (the tokens after the shared prefix, plus template tokens)
2. `key=off` vs `key=on`: does `prompt_cache_key` raise the hit rate on a shared fleet? Explain
   it in routing terms (affinity → same replica → warm cache).
3. `timestamp@top` vs `@bottom`: explain the result with "the earliest change invalidates
   everything after it".
4. `tools:shuffled` vs `sorted`: why does tool order matter? (tools are rendered into the prompt
   by the chat template)
5. Write the 3 rules you'd give a customer whose agent gets a 20% cache hit rate.

### Lab 2.2 — Observe speculative decoding

```bash
python3 labs/spec_probe.py --reps 3
MODEL=moonshotai/Kimi-K3.1-Pro python3 labs/spec_probe.py --reps 3
```

Compare decode TPS and tokens/chunk for copy vs creative vs random-words. Explain the gap via
acceptance rate. If there's no gap, give two possible reasons.

### Lab 2.3 — Memory math

Solve §8 on paper first, then check each answer with `labs/kv_calc.py`. Then size the model
your favorite customer runs, from its `config.json` on Hugging Face.

### Lab 2.4 — Read the source

- vLLM spec decode: `vllm/v1/spec_decode/` (find where draft tokens are verified and accepted).
- TRT-LLM EAGLE3: `tensorrt_llm/_torch/speculative/eagle3.py` and `drafting_loops.py` in
  https://github.com/NVIDIA/TensorRT-LLM (find where draft tokens are padded for CUDA graphs).
- vLLM retention interval: `vllm/config/cache.py` (`prefix_cache_retention_interval`).

---

## Self-check

1. A customer's cache hit rate dropped from 95% to 60% with no deploy. List four causes in order
   of likelihood and the signal for each. *(replica count change / not-ready pods, router hotspot
   shedding, KV-store master restart, workload shape change, prompt-ordering change on their side)*
2. Why can FP8 KV improve cache hit rate without improving throughput?
3. When does speculative decoding make things *slower*?
4. 2×TP1 or 1×TP2 for a customer with a 400 ms TTFT p50 SLO? Why?
5. Explain P:D disaggregation to an engineering manager in three sentences, including the cost.

**Explain it out loud (2 min):** "A customer wants to cut cost by 40% without hurting latency.
Walk me through the levers you'd evaluate, in order, and how you'd prove each one worked."
