# Inference Mastery — a self-sufficient runbook for production LLM inference

A hands-on path from "I can call an LLM API" to "I can read a customer's workload, pick the
right serving trade-off, and spot what will break in production." Built from real production
work on large GPU fleets (1,000+ GPU deployments, frontier MoE models, disaggregated serving),
with exercises you can run today using nothing but Python and a Together AI serverless key.

> Scope: inference engines & serving, inference optimization, the open-model & gateway
> landscape, and the production/customer lifecycle. Post-training (SFT, LoRA, DPO, RL) is
> intentionally out of scope for this runbook.

---

## How it's organized

| Module | Covers | Labs |
|---|---|---|
| [00 Setup](00_setup.md) | Key, models, costs, conventions, troubleshooting | `common.py --selftest` |
| [01 Engines & Serving](01_engines_and_serving.md) | Request lifecycle, prefill vs decode, vLLM / SGLang / TRT-LLM / TGI, benchmarking (TTFT, TPS, ITL, goodput), throughput vs latency, engine configuration | `bench.py`, `context_ladder.py` |
| [02 Inference Optimization](02_inference_optimization.md) | KV cache design, prefix caching (incl. hybrid models), speculative decoding, parallelism, quantization, prefill/decode disaggregation, MoE kernel tuning | `prefix_cache.py`, `spec_probe.py`, `kv_calc.py` |
| [03 Models, Gateways & Harnesses](03_models_gateways_harnesses.md) | Open-model families, architecture patterns, model selection, model gateways (SMG), parsers & chat templates, agent harnesses | `tool_calls.py` |
| [04 Production & Customer Lifecycle](04_production_lifecycle.md) | Production readiness, sizing, rollouts & canaries, monitoring & alerting, incident method, customer communication | `monitor.py` |
| [05 Case Studies](05_case_studies.md) | Ten real incidents/optimizations, each with symptom → evidence → root cause → lesson | read + reproduce |
| [06 Interview Drills](06_interview_drills.md) | Question bank, whiteboard prompts, customer scenarios, no-AI practice protocol | mock sessions |

Every concept module follows the same shape: **mental model → what matters in production →
labs → self-check questions → "explain it out loud" prompt.**

---

## The 4-week path (~60–90 min/day)

| Week | Days 1–2 | Days 3–4 | Day 5 | Weekend |
|---|---|---|---|---|
| **1 — Serving** | 00 Setup, 01 §1–3 | Lab 1.1 + 1.2 (`bench.py`) | Lab 1.3 (`context_ladder.py`) + 01 §4–6 | Case studies 1–3 · self-check 01 |
| **2 — Optimization** | 02 §1–2 (KV, prefix cache) + Lab 2.1 | 02 §3 (spec decode) + Lab 2.2 · §4 parallelism | 02 §5–7 + Lab 2.3 (`kv_calc.py`) | Case studies 4–6 · self-check 02 |
| **3 — Models & gateways** | 03 §1–3 (families, architectures, selection) | 03 §4–6 (gateways, templates, harnesses) | Lab 3.1 (`tool_calls.py`) across 3 models | Selection scorecard exercise · self-check 03 |
| **4 — Production** | 04 §1–3 (readiness, sizing, rollouts) | 04 §4–6 (monitoring, incidents, comms) + Lab 4.1 | Case studies 7–10 | 06 drills: 2 mock sessions, **no AI tools** |

Use the deliverables column in each module as your "definition of done". If a week ends
without the deliverables, the week isn't done: extend it rather than skipping ahead.

---

## Readiness checklist

You're at a good level when you can do all of these **from memory, out loud, without notes**:

**Serving**
- [ ] Explain why decode is memory-bandwidth-bound and prefill is compute-bound, with the arithmetic.
- [ ] Define TTFT, TPOT/ITL, TPS, throughput, goodput, and say which one each stakeholder cares about.
- [ ] Find the concurrency knee of an endpoint and explain what happens to TTFT past it.
- [ ] Pick between vLLM, SGLang, TRT-LLM and TGI for a given workload and justify it.
- [ ] Name 6 engine knobs and the trade-off each one controls.

**Optimization**
- [ ] Compute KV bytes/token for a model from its config, and max concurrency for a GPU shape.
- [ ] Explain why prefix caching can silently fail on hybrid (Mamba/linear-attention) models.
- [ ] Explain speculative decoding, acceptance rate, and why it can hurt at high batch.
- [ ] Choose TP / PP / EP / DP for a model + hardware and explain the communication cost.
- [ ] Compare FP8, FP4/NVFP4, AWQ/GPTQ and KV-cache quantization, and how you'd validate quality.
- [ ] Explain prefill/decode disaggregation, when it pays off, and how you'd set the P:D ratio.

**Models & gateways**
- [ ] Summarize the major open model families and what each is best at.
- [ ] Recognize dense vs MoE, MLA, GQA, hybrid linear attention, sliding window, MTP from a config.
- [ ] Walk through a model-selection decision for a customer use case.
- [ ] Explain what a model gateway does (templating, parsing, routing, admission) and how it fails.

**Production**
- [ ] Size a deployment from a customer's traffic profile (RPS, tokens in/out, cache rate, SLO).
- [ ] Design a rollout (canary, blue/green, shadow traffic) and its abort criteria.
- [ ] Design the 4–5 alerts that matter for an endpoint, with thresholds and their failure modes.
- [ ] Run a structured incident investigation that locates the failing layer with data.
- [ ] Write a customer update during an incident: what's affected, what you're doing, next update time.

---

## Sources this runbook is built from

Hands-on production experience (incidents, optimizations and capacity work on large GPU fleets),
plus these public references (start here when a module says "go deeper"):

- vLLM docs & blog — https://docs.vllm.ai · https://blog.vllm.ai
- SGLang docs — https://docs.sglang.ai
- TensorRT-LLM docs — https://nvidia.github.io/TensorRT-LLM/
- NVIDIA Dynamo — https://github.com/ai-dynamo/dynamo
- SMG (Shepherd Model Gateway) — https://github.com/lightseekorg/smg
- Together AI docs & blog — https://docs.together.ai · https://www.together.ai/blog
- Lilian Weng — https://lilianweng.github.io · Chip Huyen — https://huyenchip.com/blog/
- Hugging Face blog — https://huggingface.co/blog

> Case studies are anonymized: no customer, company-internal system or infrastructure identifiers.
> Numbers are kept where they carry the lesson.
