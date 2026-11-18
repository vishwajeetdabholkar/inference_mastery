# 03 — Models, Gateways & Harnesses

**Goal:** reason about *which* open model fits a use case, read an architecture from its
config, and understand the gateway layer between the API and the engine, because that layer
is where many "the model is broken" reports actually come from.

**Deliverables for this module**
1. `tool_calls.csv` across 3 models, with every FAIL explained by a hypothesis about which layer dropped it.
2. A model-selection scorecard for one realistic use case (§3), backed by your own measurements.
3. Architecture notes for 3 models read from their `config.json` (§2).

---

## 1. The open-model landscape (snapshot: Nov 2026, re-check monthly)

| Family | Current examples | Known for | Architecture notes |
|---|---|---|---|
| **Moonshot Kimi** | K2.6, K2.7 Code, K3, K3.1 Pro | Frontier agentic/coding, 1M context, Together's recommended chat & reasoning pick (K3.1 Pro) | Large MoE; K3 is hybrid (KDA linear attention + full attention); served NVFP4 |
| **Z.ai GLM** | GLM-5.2, 5.3, 5.3 Flash | Coding agents (5.3 recommended), function calling at volume (5.3 Flash) | MoE; thinking on by default; FP4 on serverless |
| **Qwen** | Qwen3.5 (9B → 397B-A17B), 3.6, 3.7, 3.8 | Widest size range, strong multilingual, the default fine-tuning base | Qwen3.5 is hybrid (Gated DeltaNet + attention) → needs hybrid-aware engine settings |
| **DeepSeek** | V4 Pro, V4.1 Flash | Strong reasoning per dollar, extremely cheap cached input | MoE; the V3 line introduced MLA (tiny KV per token) and MTP heads. Check each new release's config |
| **MiniMax** | M3.1, M3.1 Flash | 1M context at low price | Large MoE |
| **OpenAI gpt-oss** | 120B (20B for fine-tuning/dedicated) | Open-weight reasoning with an effort knob | MoE, ships MXFP4 |
| **Meta** | Llama 4 Scout/Maverick, Muse Spark Open 816B-A42B, Muse Glimmer 30B | Ecosystem, multimodal, long context | MoE (Llama 4, Muse Spark) |
| **Google Gemma 4** | 31B, 26B-A4B | Efficient small/mid models, VLM variants | Dense and MoE variants (the Gemma line uses interleaved sliding-window attention; verify in config) |
| **NVIDIA Nemotron 3.5** | Super VL, Lightning | Vision-language, efficient | Hybrid / VL variants |

Always cross-check against the live catalog (https://docs.together.ai/docs/serverless/models)
and Together's recommended-models page (https://docs.together.ai/docs/inference/recommended-models).
Strength claims age fast; measure on the customer's task before recommending anything.

## 2. Architecture patterns, and why each matters for serving

| Pattern | What to look for in `config.json` | Serving consequence |
|---|---|---|
| Dense | no expert fields | Compute ∝ all params; simplest to serve |
| MoE | `num_experts` / `n_routed_experts`, `num_experts_per_tok` | Memory ∝ total params, compute ∝ active; needs EP at scale; MoE kernel tuning; load imbalance across experts |
| GQA | `num_key_value_heads` < `num_attention_heads` | Smaller KV → more concurrency |
| MLA | `kv_lora_rank`, `qk_rope_head_dim` | Tiny KV; latent replicated under TP → DP-attention |
| Sliding window | `sliding_window`, layer-type lists | KV capped per layer; prefix reuse needs retention checkpoints |
| Hybrid linear attention (Mamba / GDN / KDA) | `layer_types` / `linear_attn` configs, `mamba_*` | Fixed recurrent state per sequence; prefix caching needs state snapshots; silent concurrency caps |
| MTP heads | `num_nextn_predict_layers` | Built-in speculative decoding |
| Long-context RoPE scaling | `rope_scaling` / `rope_parameters` (YaRN) | Engines must read it correctly (a real TRT-LLM fix: reading transformers-5 `rope_parameters` for YaRN) |
| Multimodal | `vision_config`, image token limits | Image preprocessing cost, per-request image caps, payload size limits |
| Reasoning | chat template thinking markers | Reasoning tokens bill as output; TTFT must count them; reasoning parser required |

**Exercise:** open the `config.json` of three models on Hugging Face (one dense, one MoE + MLA,
one hybrid) and fill this table for each: total/active params, layers, KV heads × head_dim (or
MLA dims), attention type per layer, context length, and the one serving risk it implies.

## 3. Model selection framework (what an FDE/SA actually does)

1. **Requirements first:** task type, quality bar (how measured), latency SLO (TTFT/TPS), context
   length, tool calling / JSON needs, languages, modality, data/licensing constraints, volume, budget.
2. **Shortlist 3:** a frontier pick, a cost-efficient pick, a small/fast pick.
3. **Measure, don't assume:** a 20–50 example eval on *their* data, plus `bench.py` (latency at
   their prompt shape), `tool_calls.py` (conformance), and `prefix_cache.py` (cacheability).
4. **Cost model:** (input tokens × (1 − cache rate) × input price) + (input × cache rate × cached
   price) + (output tokens × output price), per request, × volume. Then compare with a dedicated
   endpoint: GPU-hours × price for the replicas needed to hold the SLO at peak.
5. **Deployment option:** serverless (best-effort, pay per token) → dedicated endpoint (reserved
   GPUs, predictable latency, custom weights, opt-in optimizations) → provisioned throughput
   (committed throughput). Volume and SLO strictness move you right.
6. **Write it down** as a one-page recommendation: winner, runner-up, numbers, risks, and what
   you'd re-test after launch.

**Scorecard template**

| Model | Task score (n=…) | TTFT p95 @ their shape | Decode TPS | Tool-call conformance | Cache hit | $ / 1k requests | Risks |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

## 4. Model gateways (SMG and friends)

A gateway sits between clients and engines. In SMG (Shepherd Model Gateway, Rust, open source):

```
request → auth / admission → chat template → tokenize → ROUTE (pick worker)
        → engine generate (gRPC or HTTP) → detokenize → reasoning parser → tool-call parser → SSE
```

**What it owns, and how each piece fails in production:**

| Responsibility | Production failure we've seen |
|---|---|
| Protocol adaptation (OpenAI / Anthropic / Responses → engine) | A request parameter (`parallel_tool_calls`) was **accepted (200) but on the excluded-fields list for every backend**, so it never reached the engine. LangChain sends it by default, so agents fired 2–6 tool calls with real side effects. |
| Request validation | Truncated tool-call arguments in conversation *history* triggered **7,489 consecutive 400s** for one agent client. Fix: repair or default malformed history args instead of rejecting the whole request. |
| Body limits | An oversized body got a fast 413 from the gateway, which closed the connection while the router was still uploading, so **the client saw a 500** (`BodyWrite … Broken pipe`). Always check for a 413 on the same request id. |
| Routing policy | Cache-aware routing kept sending traffic (and new sessions) to workers on a node with broken pod networking, even after ~7,850 health-check failures per worker per day: no ejection. |
| Health / circuit breaking | With the circuit breaker disabled, a bad worker is removed only after the health-check threshold (e.g. 10 × 20 s ≈ 3+ min). Every request in that window fails. |
| Parsers (reasoning, tool calls) | Each model family needs its own parser; a wrong or outdated one corrupts tool calls without erroring. |
| Admission (per-org fair queues) | 429s from the gateway mean admission rejection, not engine overload. Read the error code before blaming capacity. |

**Routing policies to know (SMG `model_gateway/src/policies/`):** `round_robin`, `random`,
`power_of_two` (pick the less loaded of two), `cache_aware` (prefix-tree affinity with load
thresholds), `prefix_hash` / `consistent_hashing` (deterministic affinity), `dp_min_token`
(DP-attention aware), `bucket`, `manual`. Production forks often add cost-aware and session-aware
variants with hotspot detection. The core trade-off: **cache affinity vs load balance**.

**Other gateways you'll meet:** LiteLLM (multi-provider proxy), Envoy AI Gateway, provider SDK
routers, and Dynamo's own frontend (KV-aware routing inside a disaggregated fleet).

## 5. Chat templates are code

The chat template turns `messages + tools` into the exact token sequence the model was trained
on. It decides where tool definitions go, how reasoning is marked, and which flags are honored.

- A tool flag (`defer_loading: true`) was meant to keep deferred tools out of the prompt. Our
  servers ignored it, so the prefix changed every request and **cached tokens dropped to 0**.
  Pinning it down took comparing **8 revisions** of the official template against what we served.
- Template-added tokens explain why a "fully shared" prompt never reaches 100% cache hit.
- Some gateways can render tools exactly like `vllm serve` does (a match-vLLM mode) to avoid
  subtle prompt differences between serving paths.

**Rule:** version-pin the template with the model revision, diff it on every upgrade, and test
tool calling + caching after any template change.

## 6. Harnesses (coding agents and agent frameworks)

Agent harnesses (Claude Code, Codex, OpenCode, Pi, LangGraph, custom loops) are the heaviest
users of an inference API. What they stress:

- **Very long, mostly-static prefixes** → cache hit rate is everything; the ordering rules from
  module 02 (static first, sorted tools, timestamps last) decide the bill and the TTFT.
- **Tool calling fidelity**: parallel calls, `tool_choice`, streamed argument fragments, orphan
  tool-call ids in history. Servers differ in strictness. A 57-test conformance suite showed the same 3
  strictness failures across every config of one model, so that's a model/gateway trait, not a config bug.
- **Protocol surface**: OpenAI Chat Completions, Responses (`previous_response_id`, server-side
  history), Anthropic Messages (`cache_control` is a no-op on Together because caching is automatic).
- **Long sessions** → session affinity in the router, and graceful handling of client disconnects (499s).

---

## Labs

### Lab 3.1 — Conformance across models

```bash
MODEL=zai-org/GLM-5.3-Flash        python3 labs/tool_calls.py
MODEL=moonshotai/Kimi-K3.1-Pro     python3 labs/tool_calls.py
MODEL=deepseek-ai/DeepSeek-V4.1-Flash python3 labs/tool_calls.py
```

For every FAIL with `http=200`, write a hypothesis about which layer is responsible: model
behaviour, chat template, gateway protocol adaptation, or parser. How would you prove it?
(Hint: compare streaming vs non-streaming, and compare the same model through two serving paths.)

### Lab 3.2 — Selection scorecard for a real use case

Pick a use case, e.g. *"customer-support agent: ~3k-token system prompt + 4 tools, replies under
200 tokens, TTFT p95 < 1.5 s, 50 RPS peak"*. Write 20 test prompts with expected behaviour. For
3 models, run `bench.py --prompt-tokens 3000 --max-tokens 200 --cache shared`, `tool_calls.py`,
and your 20 prompts. Fill the §3 scorecard, compute $/1k requests from catalog prices, and
write the one-page recommendation.

### Lab 3.3 — Run a gateway yourself (optional, no GPU)

Run SMG (https://github.com/lightseekorg/smg) in front of Ollama on a laptop, or in front of
vLLM on one GPU. Change the routing policy between `round_robin` and `cache_aware`, rerun the same
multi-turn traffic, and compare cache hit rate and latency.

### Lab 3.4 — Read the gateway source

In https://github.com/lightseekorg/smg:
- `model_gateway/src/policies/cache_aware.rs`: when does it break affinity to balance load?
- `model_gateway/src/routers/grpc/pipeline.rs`: the per-request pipeline stages.
- `crates/tool_parser/src/parsers/` (e.g. `glm4_moe.rs`, `kimik2.rs`, `qwen.rs`): how a raw
  model output becomes a `tool_calls` array.
- `model_gateway/src/routers/http/router.rs`: retries and worker selection on the HTTP path.

---

## Self-check

1. A customer says "Kimi ignores `parallel_tool_calls=false`". List the layers a request crosses
   and how you'd find the one that drops the flag.
2. Why does an MoE model with 10B active parameters still need ~120 GB of GPU memory in FP8?
3. What's the first thing you check when a hybrid model shows 0% cache hit?
4. A customer wants "the best model for coding agents" at the lowest cost. Walk through your method.
5. Why can a gateway 429 and an engine 503 call for completely different fixes?

**Explain it out loud (2 min):** "What does a model gateway do, and tell me about a time the
gateway, not the model, was the root cause."
