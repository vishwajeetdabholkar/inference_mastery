# 05 — Case Studies from Production

Ten real investigations, anonymized. Each one teaches a concept from modules 01–04, and
each is written so you can **replay it**: read only the *Symptom*, write your plan, then read on.

Format: **Symptom → Investigation → Root cause → Fix & result → Lesson → Concepts**.

---

## 1. "Can we run it on fewer GPUs?" — the prefix cache that never worked

**Symptom.** A customer wanted to replace a 235B model on 4×H100 with a newer 122B hybrid
model. The catalog configuration reached 271 req/min against a 1,432 req/min target.

**Investigation.** Deployed and benchmarked a chain of configs, changing one thing at a time:
cache reporting on, exact clone of the engine team's config, pinned platform versions, raised
concurrency. `cached_tokens` stayed at 0 on identical repeated prompts in every variant. That
elimination chain ruled out reporting, versions and slot scarcity.

**Root cause.** The model is hybrid (linear attention + attention). With SGLang's default
`--mamba-scheduler-strategy no_buffer`, no recurrent-state snapshots are kept at prefix
boundaries, so the radix cache can never hit and the overlap scheduler is off. Every stock
config, including the certified one, shipped with dead prefix caching.

**Fix & result.** `--mamba-scheduler-strategy extra_buffer`: cache hit 0% → ~98% on repeats,
100k-token TTFT 16.5 s → 4.1 s, decode 161 → 196 tok/s. Then FP8 weights on 2×B200 TP2:
**814 req/min per replica at TTFT p50 0.9 s**, so 2 replicas clear the target. Also caught that
a "BF16-optimized" comparison only had FP8 *KV cache*, and re-ran a true FP8-weights track
before recommending anything. A further finding: the engine silently capped running requests at
15–22 (not the configured 128) because of recurrent-state slot math; pinning
`--max-mamba-cache-size` and bf16 state raised it.

**Lesson.** Check the cache hit rate before buying GPUs; verify effective limits in startup logs;
make comparisons apples to apples.
**Concepts.** 02 §2 (hybrid prefix caching), 01 §7 (silent caps), 02 §8 Q5.

---

## 2. The 502s everyone blamed on the network

**Symptom.** A ~1,300-GPU deployment paged 21 times in ~32 hours, each a burst of 502s failing
20–170 live requests at once. First guess: a flaky cluster network.

**Investigation.** Mapped every burst to the edge proxy pod, destination cluster and exact
error. Each burst came from **one proxy pod at a time, a different pod each time**, and hit two
different clusters. Then ruled things out: ingress showed no restarts and no errors; edge health
checks had 0 failures; failing pods were on different nodes; another edge sent 100+ RPS to the
same clusters with zero 502s. **Control group:** old proxy pods served 1.5M requests with 0 drops,
new pods served 1M with 441 drops, and every failing pod broke 10 s–3 min after starting.

**Root cause.** `http2: client connection lost` is what Go's HTTP/2 client raises when its own
PING health check times out. Freshly started proxy pods were killing their own connections,
~150 streams at a time. Request traces showed the backend **completing with 200 while the
client received 502**: GPU work done, request failed anyway.

**Fix & result.** One evidence-backed thread routed the problem to the proxy owners
(fresh-pod connection handling) instead of a vague network ticket. The next day it recurred
without a rollout, from pod churn within the same release, which confirmed "fresh pod" over
"bad version". An earlier claim in the thread was corrected publicly when new data contradicted it.

**Lesson.** Find a control group; read the protocol-level error; trace one request end to end.
**Concepts.** 04 §5 (incident method, 502 attribution), 01 §1 (hops).

---

## 3. The 503 cascade with flat traffic

**Symptom.** Success rate dropped hard on one cluster, then another, then recurred the same day.
Request volume looked flat, so "too much traffic" didn't fit.

**Investigation.** Walked the path: request mix, engine stats, cache hit rate, pod health.
The first hypothesis ("traffic shape changed") was wrong, and was dropped publicly.

**Root cause.** Two triggers together: a top-of-the-hour burst plus one tenant sending long,
never-cached prompts. The engines preempted other work to fit them, time per output token
doubled, pods filled, and sessions spilled onto pods that didn't hold their context, so the
cache hit rate collapsed, which made everything slower. A cascade.

**Fix & result.** The customer found and blocked the tenant on their side, matching the analysis.
Router behaviour was tuned so overload returns a 429 ("try again") instead of a 503, and sessions
wait for their own pod rather than moving to a cold one.

**Lesson.** Follow second-order effects (memory pressure → preemption → cache loss).
**Concepts.** 02 §1–2 (preemption, affinity), 04 §5 (TTFT ladder P4/P5).

---

## 4. The crash that restarts couldn't fix

**Symptom.** A flagship MoE deployment (disaggregated, TRT-LLM, EAGLE3 speculative decoding)
crash-looped in waves, always on **prefill** pods, never decode. Weeks of restarts didn't help.

**Investigation.** Deterministic, role-specific crashes point to a code path, not hardware.
Read the engine source along the prefill path with speculative decoding and structured output enabled.

**Root cause.** Draft tokens are zero-padded to a fixed length for CUDA-graph compatibility. The
guided-decoding bitmask path tripped over the padded positions inside the EAGLE3 drafting loop,
which only runs on prefill-only pods in this disaggregated layout.

**Fix & result.** Found the upstream TensorRT-LLM fix, backported it to the fork's release
branch, and opened the PR for the engine team: a merge-ready patch instead of an escalation.

**Lesson.** When a failure is deterministic, restarting won't fix it: the answer is in the code.
Test speculative decoding together with structured output.
**Concepts.** 02 §3 (spec decoding interactions), 01 §3 (CUDA graphs).

---

## 5. HTTP 200, but the parameter did nothing

**Symptom.** A production agent customer set `parallel_tool_calls=false` and still got 2–6 tool
calls per message, firing duplicated side effects (links sent, follow-ups scheduled). Their
framework (LangChain) sends the flag by default, and their executor runs every tool call.

**Investigation.** Reproduced on production for two model families. Traced the parameter
through every layer: the API schema accepted it (any value → 200); its only consumer upstream
was guided decoding, which isn't used for natively parsed tool calls; and the gateway listed it
among **excluded fields for every backend**, so it was stripped before reaching the engine.

**Root cause.** Accepted, then silently dropped in protocol adaptation, and nothing enforced the
contract `parallel_tool_calls=false ⇒ ≤ 1 tool call`.

**Fix & result.** Wrote a self-contained engineering handoff: the contract (streaming and
non-streaming; don't 400 the flag or every LangChain client breaks), the two-layer root cause,
and a local repro guide, so the fix could be built without touching a live endpoint.

**Lesson.** Verify behaviour, not status codes. Ask which layer dropped it.
**Concepts.** 03 §4 (gateway failure modes); Lab 3.1 automates this check.

---

## 6. The chat template that zeroed the cache

**Symptom.** A customer ready to launch production traffic on a new fast model saw **0 cached
tokens** whenever their tools carried `defer_loading: true`, which blocked the launch.

**Investigation.** Reproduced with exact token counts: deferred tools were still rendered into
the prompt, changing the prefix every request. Compared **8 revisions** of the official chat
template to confirm the intended behaviour and identified exactly which one was being served.

**Root cause.** The served template/gateway combination ignored the flag. Narrowed to two
specific candidates for engineering.

**Lesson.** Chat templates are code: pin, diff and test them like code.
**Concepts.** 03 §5, 02 §2 (prefix stability).

---

## 7. The unreachable node the router never ejected

**Symptom.** A 1M-context endpoint paged every 30–90 minutes with 500/502/504 bursts concentrated
on one cluster, for 3.5 days. The node and pods all looked healthy in Kubernetes.

**Investigation.** Router logs showed **4,800 of 4,805** upstream failures on that cluster went to
4 engine pods, all on one node. Errors were network-level: `No route to host` (EHOSTUNREACH)
and TCP connect timeouts, which explained 504s at exactly ~20 s (two 10 s connect timeouts on a
retry to the same dead worker). Kubernetes said `Ready=true`, NIC counters were clean and pods
passed readiness, because local probes run on the node itself. The node still served ~169k
requests/day vs ~290k on peers: intermittently reachable, not down.

**Root cause.** Intermittent pod-network (CNI-path) reachability from the router's node to that
node's pod IPs. The router's health checks failed ~7,850 times per worker per day, yet it kept
routing to those workers, including brand-new sessions.

**Fix & result.** Immediate relief: cordon and drain the node so its pods move elsewhere.
Root cause routed to cluster networking, plus a router ask: eject workers on health failure.

**Lesson.** "Healthy in Kubernetes" ≠ reachable from the caller. Concentrate errors by
pod → node to find a single bad host. Ejection policy matters as much as detection.
**Concepts.** 04 §4–5.

---

## 8. Dropped live calls on a voice endpoint

**Symptom.** A real-time speech-to-text customer reported dropped calls on two consecutive days,
with ~100 failed session starts in one window.

**Investigation.** Day 1: a single engine hung (requests stuck ~235 s) and there was no spare
replica to absorb its traffic. Day 2: a node failed, and its replacement took ~20 minutes to
serve, because of slow container-image extraction and CPU contention from co-located workloads.
A separate incident on the same endpoints traced drops to fiber damage at an upstream internet
provider, not our servers.

**Fix & result.** Engineering pre-loaded the engine image on all nodes: replacement startup went
from ~20 minutes to ~2–2.5. The customer got precise timestamps and a calm, non-technical
explanation each time.

**Lesson.** For real-time AI, recovery time is part of the latency budget. Keep warm spare
capacity and pre-pulled images.
**Concepts.** 04 §1 (startup time, failure behaviour), 04 §6 (communication).

---

## 9. Moving 700+ GPUs without hurting live traffic

**Symptom.** An older endpoint was sized for ~3.9B tokens/min but used ~1B; a newer endpoint
needed to reach ~3B.

**Investigation.** Measured real tokens-per-minute per GPU on each endpoint and sized the old one
for its actual 1B TPM.

**Fix & result.** Freed **728 GPUs (182 pods)** and added the capacity to the new endpoint on one
cluster deliberately. Before merging, flagged that removing one cluster's config could leave a
dangling route. It did cause a short burst of ~330 × 500s, caught and explained quickly. A standing
1,300-GPU target for the new endpoint became the team's overshoot check.

**Lesson.** Size from measured throughput. Own predicted risks: stage route removals separately.
**Concepts.** 04 §2 (sizing), 04 §3 (change risk).

---

## 10. Distributed KV cache: the hit rate that fell as the fleet grew

**Symptom.** A customer running ~150k-token prompts on a fleet with a shared, RDMA-backed KV
store asked why the cache hit rate dropped in their latest load tests.

**Investigation.** Token-weighted hit rate by run: ~98.6% at 14 workers falling to 90–94% at 40–45
workers, with uncached tokens per request rising from ~2–3k to ~8–11k. The router was moving
load off "hot" workers (`cache_miss_all_hotspots`), and failed remote lookups degrade silently
to recompute. Separately, ~800k of the org's "errors" were **mirrored shadow-traffic** 429s,
not customer requests.

**Root cause.** More workers + stricter hotspot shedding = less affinity, plus a slow remote-lookup
path stalling the scheduler.

**Fix & result.** A fresh KV-store master, asynchronous lookups (no scheduler stall) and RPC
timeouts: production settled at **~96–97%** token-weighted hit. Shadow traffic was excluded from
all customer-facing metrics.

**Lesson.** Cache affinity and load balance trade off; tiered caches fail silently; always
exclude shadow traffic.
**Concepts.** 02 §1–2, 04 §3 rule 7.

---

## Bonus: two short ones about reading the error pattern

- **"Payment required" with money in the account.** Intermittent 402s on ~1 in 4 requests while
  billing showed ample credit. *Always failing* points at the request or account; *sometimes
  failing* points at infrastructure. Mapping errors to clusters isolated a stale balance cache in
  one part of the routing layer. Customer got a retry workaround the same day.
- **Batch jobs that expired without running.** One customer's ticket turned out to be a
  platform-wide stall: job pickups stopped for ~2.5 days. Before solving it for one account,
  check whether it's happening to everyone; then confirm the customer wasn't billed and catch
  stuck resubmissions.
