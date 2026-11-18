# 06 — Interview Drills

For Solutions Architect / Forward Deployed Engineer / inference-support interviews. The bar is
not "define X" but **"when and where to apply X: read a customer's workload, recommend the right
trade-off, spot what will break in production."**

## Practice protocol (no AI tools, the way the real screen works)

1. Set a timer. Answer **out loud** (record yourself, or practice with a colleague).
2. Structure every technical answer: **clarify → mental model → trade-off → how I'd measure → real example**.
3. After each answer, check it against the "strong answer covers" bullets. Score 0–2 per bullet.
4. Re-do any question scoring below 70% two days later. Spaced repetition beats cramming.
5. Twice a week, do one **customer scenario** (§6) end to end in 25 minutes on a whiteboard.

---

## 1. Engines & serving

**Q1. Walk me through what happens to a request from API call to first token.**
Strong answer covers: edge → router → gateway (template, tokenize) → scheduler queue → prefill
→ first token; where latency hides at each hop; client-side vs engine-side TTFT.

**Q2. Why is decode memory-bound and prefill compute-bound? Show me the math.**
Covers: weights re-read per step; ~16 GB ÷ 3.35 TB/s ≈ 4.8 ms → ~200 tok/s single-user
ceiling for an 8B BF16 model on H100; batching amortizes reads; prefill has high arithmetic intensity.

**Q3. vLLM vs SGLang vs TRT-LLM: how do you choose for a customer?**
Covers: model support first, workload shape (prefix-heavy → strong prefix cache; latency-critical
on NVIDIA → TRT-LLM), operability, a real example of each, and what would make you switch.

**Q4. How would you benchmark an endpoint for a customer?**
Covers: their request shape; open vs closed loop; percentiles; cache control; goodput at SLO;
measurement point; like-for-like comparisons (timeouts, precision); tools (`vllm bench serve`,
`genai-perf`, a gateway-level traffic simulator).

**Q5. TTFT p99 is 10× p50. What's going on?**
Covers: queueing past the knee, long-prompt prefill interleaving (chunked prefill budget),
preemption, cold cache after a restart; how you'd tell them apart (queue time vs prefill time,
prompt-length distribution, restarts).

**Q6. Name the engine knobs you'd tune first and why.**
Covers: max running requests, memory fraction, chunked prefill / batched tokens, context length,
KV dtype, prefix caching, spec decoding, parallelism. Confirm effective limits from startup logs.

## 2. Inference optimization

**Q7. Compute the KV cache for Llama-3.3-70B at 32k context. How many users fit on 2×H100?**
Covers: 2 × 80 × 8 × 128 × 2 B = 320 KiB/token; ~10.5 GB per user at 32k; BF16 weights barely
fit; FP8 weights → ~73 GB KV budget → ~7 users at 32k (~14 with FP8 KV).

**Q8. Prefix caching: how does it work, and how can it silently fail?**
Covers: block hashing / radix tree; longest-prefix match; router affinity; prompt ordering;
hybrid-model state snapshots (`extra_buffer`); retention intervals; tiered caches that fail to
recompute; token-weighted measurement.

**Q9. Explain speculative decoding. When does it hurt?**
Covers: draft + verify in one pass, lossless; acceptance rate × step cost; EAGLE/MTP/n-gram;
hurts at high batch and on unpredictable output; interactions with structured output and CUDA graphs.

**Q10. TP vs PP vs EP vs DP: pick for a 600B MoE on 8×B200 serving long context.**
Covers: TP within node (NVLink), EP for experts (all-to-all), DP-attention for MLA, context
parallel for long context; the latency vs throughput-per-GPU trade-off (2×TP1 vs 1×TP2 data).

**Q11. A customer asks for FP4. What do you tell them?**
Covers: memory/bandwidth/compute wins on Blackwell; quality risk; how you validate (task evals,
tool-calling suite, long context, golden diffs); KV quantization as a separate decision.

**Q12. When is P:D disaggregation worth it, and how do you set the ratio?**
Covers: prefill/decode interference; long prompts + scale; KV transfer (NIXL/RDMA) cost; extra
failure modes; ratio from measured prefill queue vs decode batch; the ~2:1 and 1:1 → ~1:2 rebalancing examples.

## 3. Models, gateways & harnesses

**Q13. Which open model would you recommend for a coding agent, and how would you defend it?**
Covers: requirements first; shortlist of 3; measured eval on their tasks; tool-call conformance;
cacheability; cost model with cached pricing; deployment option.

**Q14. What does a model gateway do, and how have you seen one fail?**
Covers: template/tokenize/route/parse/admission; a parameter accepted but dropped; 413 masked as
500; non-ejection of unreachable workers; parser mismatches.

**Q15. Dense vs MoE vs hybrid linear attention: what changes for serving?**
Covers: memory ∝ total params vs compute ∝ active; EP and MoE kernel tuning; fixed recurrent
state per sequence; prefix-cache and concurrency implications.

**Q16. A customer's agent gets a 20% cache hit rate. What do you change?**
Covers: static content first, sorted tools, timestamps last, `prompt_cache_key` / session
affinity, stable templates; measure with token-weighted hit rate before and after.

## 4. Production & customer lifecycle

**Q17. How do you size a dedicated endpoint for a customer?**
Covers: sustainable throughput per replica at SLO (not at saturation); peak in tokens/min;
headroom; failure domains; re-check on real traffic; the 728-GPU rebalance example.

**Q18. Design the alerts for a high-priority endpoint.**
Covers: success rate (excluding 4xx), traffic cutoff vs baseline, capacity vs committed, 5xx burst,
stalled workers; pitfalls (auto-resolve, absolute thresholds at scale, probes as traffic,
single-engine loss below % thresholds).

**Q19. How do you roll out a new engine image to a large fleet?**
Covers: canary first; unit of failure (pairs); batch size vs capacity floor; verify serving, not
just Ready; abort criteria; don't mix with scale changes; rollback path; shadow traffic option.

**Q20. An alert auto-resolved but the customer says it's still failing. Go.**
Covers: trust the customer; split by code and layer; ready vs desired; pod → node; mitigate
(cordon + move); holding update with next-update time; confirm recovery in metrics, not alerts.

**Q21. Tell me how you'd communicate a degradation to a customer whose contract forbids
sharing infrastructure details.**
Covers: impact in their terms (latency, errors, cache rate), actions and next update, no GPU or
cluster specifics, no blame, a post-incident summary.

## 5. Behavioural (map each to a case study)

| Prompt | Use |
|---|---|
| "Tell me about going beyond the ask." | Case 1: fewer GPUs → found dead prefix cache, 0 → 98% hit |
| "A hard production incident you led." | Case 2: 502s, control group, protocol-level error |
| "A time you were wrong." | Case 3: dropped the first hypothesis publicly |
| "Debugging something deep." | Case 4: engine source → upstream backport |
| "Turning a ticket into a product fix." | Case 5 (parameter dropped) or the batch-stall bonus |
| "Working across teams under pressure." | Case 8: voice drops → image pre-pull, 20 → ~2 min |
| "Capacity / business impact." | Case 9: 728 GPUs moved from measured tokens-per-GPU |

Shape each as: situation (with numbers) → what I did (ruled out X with Y data) → result (numbers)
→ what changed afterwards. Include one honest correction; it makes you credible.

## 6. Customer scenarios (25 min each, whiteboard)

Each tests reading the workload, recommending a trade-off, and spotting what breaks.

**S1. Long-context legal search.** Prompts 80–150k tokens, answers ~300 tokens, 20 RPS peak,
TTFT p95 < 5 s, documents reused across users ~70% of the time.
*Expect:* prefill-dominated; prefix cache + router affinity; consider P:D and a shared KV tier;
KV budget math; risk: cache misses under load-balancing, preemption at long context.

**S2. Voice assistant.** Short turns, TTFT p95 < 400 ms, spiky traffic, zero tolerance for
dropped sessions.
*Expect:* low batch, warm spares, pre-pulled images, fast ejection of bad workers, regional
placement; risk: cold starts, single hung engine without a spare.

**S3. Migration from another provider in 2 weeks.** Same model family, 50 RPS, wants zero downtime.
*Expect:* shadow traffic or weighted split, conformance tests (tool calls, templates), load test at
1.5× peak, abort criteria, rollback, a communication plan.

**S4. Cost-down request.** Customer spends heavily on a frontier model; wants −40% cost, no
visible quality loss.
*Expect:* measure the cache rate first, then prompt-ordering fixes, FP8/FP4, smaller model for
sub-tasks (routing), dedicated vs serverless break-even, quality gates for each step.

**S5. "Your endpoint is slow."** No other detail.
*Expect:* clarify (TTFT or TPS? since when? which requests?), request ids, the TTFT ladder, a
status table, a holding update.

## 7. Rapid-fire (30 seconds each)

- What's goodput? · Why percentiles? · What's a 499? · 429 vs 503? · What is NIXL? · What is
  Mooncake? · Why does MLA need DP-attention? · What's an acceptance rate? · What does chunked
  prefill trade? · Why do CUDA graphs matter for decode? · What's a canary? · What's shadow
  traffic? · Why is TPM a better sizing unit than RPS? · What does `extra_buffer` fix? · What does
  "Using default MoE kernel config" mean?
