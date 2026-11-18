# 04 — Production & Customer Lifecycle

**Goal:** take a model from "it works in a test" to "it survives real traffic, we know when
it's sick, we can change it safely, and the customer trusts what we tell them."

**Deliverables for this module**
1. A `monitor.py` run (≥ 10 min) with an SLO verdict and per-minute buckets.
2. A production readiness review (PRR) for the §7 scenario, using the template.
3. An alert set for the §7 scenario with thresholds and the reasoning behind each number.
4. Three incident drills completed out loud (Lab 4.2) with your written timeline.

---

## 1. Production readiness: the checklist before real traffic

| Area | Question | Evidence that answers it |
|---|---|---|
| Correctness | Do tool calls, JSON mode, reasoning, long context and multimodal work on *this* config? | Conformance suite pass list, golden-prompt diffs, long-context ladder |
| Performance | Does it hold the SLO at peak, with the customer's real request shape? | Open-loop load test at 1.0× and 1.5× peak, with cache behaviour like production |
| Capacity | How many replicas for peak + failure headroom, per failure domain? | Sizing sheet (§2) |
| Effective limits | What did the engine *actually* allocate? | Startup logs: max running requests, KV tokens, state slots (configs lie) |
| Startup | How long from pod scheduled → serving? | Measured: image pull + weight load + warmup (big MoE: 10–15 min) |
| Failure behaviour | What happens when a replica, node or cluster dies mid-traffic? | Kill test or a documented failover path; router ejection time |
| Observability | Can we see RPS, TTFT, decode TPS, errors by code, cache hit, queue and ready-vs-desired? | Dashboard + alert links in the runbook |
| Rollback | How do we undo a bad change in minutes? | Previous config/deployment still available; documented steps |
| Ownership | Who's paged, and who owns each layer? | Alert labels + routing, escalation map |

**Two production gotchas that belong on every checklist:**
- **Configs are often immutable once deployed** (a new config = a new deployment, i.e. blue/green).
  Plan rollouts around that instead of expecting an in-place patch.
- **Idle test deployments get swept** (an inactivity sweeper stopped ours after ~1 h without
  traffic). A load test against a stopped deployment measures nothing.

## 2. Capacity sizing from measured numbers

1. Measure **sustainable throughput per replica at the SLO** (not at saturation): requests/min,
   input tok/min, output tok/min at the TTFT/TPS target, with realistic cache behaviour.
2. Get the customer's **peak** demand in the same units (tokens per minute is the most honest
   unit; RPS hides prompt size).
3. Replicas = peak ÷ per-replica sustainable × headroom (1.2–1.5), then round up **per failure
   domain** (cluster, region), so losing one domain doesn't breach the SLO.
4. Re-check after launch: measured TPM per GPU on real traffic beats any synthetic estimate.

**Real example:** an older endpoint was sized for ~3.9B tokens/min but actually needed ~1B; a
newer one needed to grow to ~3B. Sizing from measured tokens-per-GPU freed **728 GPUs (182
pods)** and moved them to the endpoint that needed them, placed deliberately on one cluster
rather than spread blindly. A standing GPU target for the growing endpoint (1,300 GPUs) became
the team's overshoot check.

**Customer-facing sizing example:** target 1,432 req/min at TTFT p50/p95 400/883 ms. One
optimized replica sustained 814 req/min at TTFT p50 0.9 s → 2 replicas clear the throughput
target, and 3 replicas at lower per-replica load are needed for the strict latency band. Give
the customer both numbers and let them choose the trade-off.

## 3. Rollouts: change things without breaking them

| Technique | What it is | Use for |
|---|---|---|
| **Canary** | Roll one unit, verify, then fan out | Every image/config change |
| **Blue/green** | New deployment alongside old; shift traffic; keep old for rollback | Immutable configs; engine/model swaps |
| **Traffic split / weighted routing** | Send x% to the new version | Gradual migrations, A/B on real traffic |
| **Shadow traffic** | Mirror real requests to a test deployment, drop its responses | Testing optimizations on real shapes with zero customer risk |
| **Rolling restart in batches** | Replace k units at a time, keep capacity ≥ floor | Template changes on large fleets |

**Rules learned the hard way (from large disaggregated fleets):**
1. **Canary first, even when you fan out per group.** Rolling one unit in *each* rack at once
   means a bad image degrades every rack simultaneously, which is strictly worse than one rack down.
2. **Respect the unit of failure.** If one engine is a leader + follower pod pair, roll both
   together; deleting one leaves a stale half.
3. **Never combine a roll with a replica-count change** or a second template change on the same
   group; the scheduler's ordering gets confounded.
4. **Verify each batch end to end before the next:** pods Ready *and* registered with service
   discovery *and* one real request served through the new unit. Kubernetes Ready is not "serving".
5. **Batch size = capacity floor you can afford.** E.g. 2 of 6 frontends at a time (≥ 66%
   capacity), 1 per rack where a rack has a single frontend.
6. **Don't restart stuck workers on live traffic blindly.** Replaying stuck requests once caused
   621 client cancellations (499s). Drain or shift traffic first.
7. **Exclude shadow traffic from customer metrics.** Mirrored requests appear in request logs
   under the customer's org; one endpoint's "errors" were inflated by ~800k shadow 429s.
   Filter by deployment id or a shadow request-id marker.

**Abort criteria (decide before you start):** 5xx above baseline for N minutes, TTFT p95 > X,
any pod crash-looping, cache hit rate drop > Y points, or ready < committed floor. Write them in
the change ticket.

## 4. Monitoring & alerting

**The LLM golden signals:** RPS by status code · TTFT p50/p95 · decode TPS p50 · 5xx and 429
rates (separately) · cache hit rate · engine queue (waiting requests) and KV usage · preemptions ·
ready vs committed replicas · client cancellations (499).

**A minimal alert set that worked for high-priority dedicated endpoints:**

| Alert | Condition | Why this shape |
|---|---|---|
| Success rate < 95% | 2xx / (non-4xx) over 10 m, per deployment, skip < 0.2 RPS | Excludes client errors; ignores idle noise |
| Sudden traffic cutoff | 5 m RPS < 20% of trailing 30 m baseline, baseline > 1 RPS | Catches "customer failed over / can't reach us", which error-rate alerts miss entirely |
| Capacity below 85% of committed | ready ÷ committed replicas per role | Catches node loss before latency moves |
| 5xx burst | 5xx > 1.5 RPS over 2 m | Fast signal for hard failures |
| Worker stalled | per pod: waiting requests, 0 prompt + 0 generated tokens over 10 m | Catches hung engines that health checks still pass |

**Alerting pitfalls we hit (each one is a good interview story):**
- **Auto-resolved ≠ fixed.** Error rate can dip under threshold during load-balancer convergence
  while the capacity problem persists. If the customer says it's still failing, believe the customer.
- **Absolute thresholds don't scale.** 1.5 RPS of 5xx is ~0.1% at 1,300 RPS but 100% at 1.5 RPS.
  Re-derive thresholds when traffic changes by an order of magnitude.
- **Percentage capacity alerts miss single-engine loss.** At 34 committed, "< 85%" needs 6 engines
  down. One crash-looping engine on a bad node can sit unnoticed for days.
- **Probes and polling count as traffic.** Health checks and `/workers` polling (~1 RPS) make an
  idle deployment look busy. Exclude them in every query.
- **Routing policy can override severity.** A policy that pages on any `team=…` label turns
  warnings into pages. Set the receiver per rule.
- **A dead metadata service may not trip any alert.** E.g. a KV-store master outage shows up only
  as cache-hit loss and TTFT growth.
- **Console status ≠ customer impact.** "DEGRADED" with 492/512 ready is normal churn at scale;
  check RPS and success rate before paging humans.

## 5. Incident method: locate the layer with data

1. **Confirm impact from the customer's side.** Edge success rate, the customer's own description,
   client cancellations. Not the auto-resolved alert.
2. **Split the signal.** By status code (429 ≠ 503 ≠ 504), by layer (edge vs engine), by cluster,
   by deployment/revision, by pod, by node. A "9% error rate" can be 28% on a 15% canary and 0%
   everywhere else.
3. **Find a control group.** Old pods vs new pods, cluster A vs B, another edge sending to the same
   clusters, another customer on the same fleet. A comparison ends arguments.
4. **Check what changed.** Deploys, config revisions, scale events, node events, maintenance,
   the customer's traffic shape.
5. **Hypothesis + the signal that would confirm it.** Say both out loud. Kill hypotheses fast.
6. **Mitigate before root-causing.** Roll back the canary, cordon the node, shift traffic. Don't
   wait for a code fix to stop the bleeding.
7. **Trace individual requests end to end.** Request id → router decision → engine pod → node →
   the *actual* engine error (not just "HTTP 500").
8. **Correct yourself publicly** when new data contradicts an earlier claim.

**TTFT ladder (stop at the first probe that lights up):**
P1 edge vs engine TTFT (where is the latency?) → P2 engine crashes / restarts (hot-pod crash →
cold re-warm spike that self-heals in 15–30 min) → P3 capacity (ready replicas stepped down) →
P4 load (RPS or input-token surge; queue time ≫ prefill time) → P5 cache (hit-rate cliff without
restarts = routing/affinity or workload change) → P6 network/edge (health-check flaps, proxy-minted
errors, packet loss: ping and curl it from your laptop).

**Status-code attribution cheat sheet**

| Code | Usually means | First check |
|---|---|---|
| 429 | Admission / rate limit (per org or per key) | Which limiter, which org, request rate vs limit; quota formula |
| 499 | Client gave up (timeout, cancel) | Did latency rise first? Baseline rate? |
| 500 | Engine or gateway error, or a masked 413 | Engine log line for the request id; look for a 413 on the same id |
| 502 | Upstream connection failed or dropped | Which hop; `http2: client connection lost` = proxy-side connection drop |
| 503 | No healthy capacity / overload shed | Ready vs desired, warm-up lag, queue depth, circuit breaker |
| 504 | Timeout between hops | Which timeout fired; unreachable node/pod (`No route to host`), slow dependency |

**For every 5xx report:** request id(s), source engine pod, its node, and the actual engine error.
Collapse duplicates into one row per pod + error.

## 6. Customer communication

- **Lead with a status table** (RPS, decode p50, TTFT p50/p95, 5xx, 4xx/429, capacity, a 2–3 line
  note). Numbers first, narrative second.
- **Holding update while still investigating:** what's affected, what isn't, what we're doing,
  and when the next update comes. No blame, no guesses stated as facts, no "it's fixed" until
  metrics confirm it.
- **Translate, don't dump.** Customers want impact and action, not CUDA errors. Some contracts
  forbid sharing infra details (GPU type, clusters, replica counts); keep to load, latency,
  errors and cache rate.
- **Close the loop:** post-incident summary with timeline (UTC), impact (requests, minutes),
  root cause, mitigation, prevention, and what the customer can do (e.g. retry with backoff).

**Holding update template**
> We're seeing elevated latency and some errors on your endpoint since 14:12 UTC. Roughly half of
> requests are affected; the rest complete normally. We've identified the affected capacity and
> are moving traffic off it now. Next update by 14:45 UTC, or sooner if it clears.

## 7. Scenario for the PRR and alert deliverables

> A customer wants a dedicated endpoint for a support-agent product: ~4k-token shared system
> prompt + ~1.5k tokens of conversation per turn, ~150-token replies, tool calling (6 tools),
> peak 40 RPS during US business hours, TTFT p95 < 1.5 s, 99.9% monthly success. They'll migrate
> from another provider in two weeks and want zero-downtime cutover.

Write: (a) the PRR using §1's table, (b) a sizing estimate with explicit assumptions (§2),
(c) the rollout/cutover plan with abort criteria (§3), (d) the alert set with thresholds (§4),
and (e) the first customer message you'd send if the cutover canary shows 3% errors.

---

## Labs

### Lab 4.1 — Synthetic monitor with an SLO verdict

```bash
python3 labs/monitor.py --duration 600 --interval 10 --slo-success 0.995 --slo-ttft-p95 2.0
```

1. Read the per-minute buckets. Is there a minute that's worse than the rest? Correlate it with
   the time of day (shared serverless load).
2. Take one failed probe's `request_id` from `labs/results/monitor.csv`. That id is what you'd
   hand to support or trace through the platform.
3. Run two monitors at once against two models. Which one is more stable, and how would you prove
   it isn't noise? (longer window, percentiles, repeat at another hour)

### Lab 4.2 — Incident drills (out loud, no AI)

Time-box each drill to 15 minutes. Say which signal you'd pull, in order, and what each one would
confirm or rule out. Ideally a partner plays the system and reveals data only when you ask for it.
Then open the outline and compare.

**Drill A: slower, but nothing is failing.** TTFT p95 doubled since 09:00. No errors, request
rate flat, prompt sizes unchanged.

<details><summary>Outline</summary>

Split edge vs engine TTFT first (it's engine-side). Restarts? None. Ready replicas? Up, not
down: someone added replicas at 09:00. Cache hit rate fell from ~90% to ~55%: with more workers,
prefix affinity spreads thinner and the router sheds "hot" workers more often, so prompts land on
replicas without their cache. Mitigate: revert or retune affinity/hotspot settings, and give new
replicas time to warm. Lesson: a capacity *increase* can make latency worse.
</details>

**Drill B: one customer, 30% 400s.** Since yesterday one agent customer gets HTTP 400 on ~30% of
requests; every other customer is fine.

<details><summary>Outline</summary>

Group the 400s by error text and request shape, not by time. They all carry a conversation
history with a truncated (invalid JSON) tool-call argument, which the gateway now rejects
strictly, and every follow-up in that session fails the same way. Workaround: the customer repairs
or drops the bad turn. Fix: the gateway repairs or defaults malformed *history* arguments
instead of rejecting the whole request (case study 5's sibling).
</details>

**Drill C: 504s at exactly 20 seconds.** Bursts of 504s, all from one cluster, almost all at
~20 s latency.

<details><summary>Outline</summary>

A fixed latency is a timeout signature: two 10 s connect timeouts, from a retry hitting the same
dead worker. Group failures by upstream worker: nearly all go to a few pods on one node. Errors
are `No route to host` / connect timeouts, yet Kubernetes says the node is Ready (local probes
pass). Mitigate: cordon and drain the node. Follow up: pod-network root cause, and a router policy
that ejects workers after repeated health-check failures (case study 7).
</details>

### Lab 4.3 — Case-study replays

For case studies 2, 3, 7 and 8 in `05_case_studies.md`: read only the **Symptom**, write your
investigation plan (signals, in order, with the hypothesis each tests), then read the rest and score yourself.

---

## Self-check

1. An alert auto-resolved 5 minutes after firing; the customer says it's still broken. What do you do first?
2. Design the abort criteria for rolling a new engine image across 3 clusters.
3. Why is "tokens per minute" a better sizing unit than RPS?
4. Name two failure modes that produce no 5xx at all, and the metric that catches each.
5. Write a 3-line customer update for an incident where you don't yet know the root cause.

**Explain it out loud (2 min):** "Tell me about a production incident you led: how you found the
failing layer, how you mitigated, and what you told the customer while you were still working."
