# 00 — Setup

Everything here runs on a laptop with Python 3.9+. No GPU needed: the labs measure a real
production inference service (Together AI serverless) from the outside, and one lab is
pure offline math.

## 1. Get a key and set it

1. Create a key at https://api.together.ai/settings/api-keys (free credits are enough).
2. Export it in the shell you'll run labs from:

```bash
export TOGETHER_API_KEY=...            # never commit it, never paste it in chat
export MODEL=zai-org/GLM-5.3-Flash     # optional: default model for every lab
cd inference-mastery                   # the repo root
python3 labs/common.py --selftest      # offline sanity check of the parsing code
python3 labs/kv_calc.py --selftest     # offline sanity check of the memory math
```

## 2. Pick models deliberately

Labs default to `zai-org/GLM-5.3-Flash`. Swap with `MODEL=...` per run. Choose by what
the lab needs, and always check the live list first (models change monthly):

```bash
curl -s https://api.together.ai/v1/models -H "Authorization: Bearer $TOGETHER_API_KEY" \
  -H "User-Agent: labs" | python3 -c "import json,sys; [print(m['id']) for m in json.load(sys.stdin) if m.get('type')=='chat']"
```

| Need | Good choices (Nov 2026 catalog) | Why |
|---|---|---|
| Cheap default, tool calling, cached pricing | `zai-org/GLM-5.3-Flash` | $0.15 in / $0.50 out / $0.03 cached per 1M |
| Prefix-cache lab (cache visible + cheap) | `deepseek-ai/DeepSeek-V4.1-Flash`, `MiniMaxAI/MiniMax-M3.1-Flash` | very low cached-input price |
| Small & fast baseline | `Qwen/Qwen3.5-9B` | small model, compare against MoE giants |
| Frontier MoE (compare behaviour) | `moonshotai/Kimi-K3.1-Pro`, `zai-org/GLM-5.3`, `meta-models/Muse-Spark-Open-816B-A42B` | large MoE, long context, reasoning |
| Reasoning with effort control | `openai/gpt-oss-120b` | MXFP4, reasoning effort knob |

Models without a cached-input price (e.g. `Qwen/Qwen3.5-9B`, `openai/gpt-oss-120b`) won't show
cache savings; use them as a control group in the prefix-cache lab.

## 3. Cost guardrails

Running **every** lab once on the Flash models costs well under $1. The expensive knobs are
`context_ladder.py --sizes` with 128k+ prompts on a frontier model, and long `monitor.py` runs
on a frontier model. Check prices on https://docs.together.ai/docs/serverless/models before
pointing a lab at a $3/M-input model.

## 4. Conventions in the labs

- **stdlib only** (`urllib`, `json`, `threading`): nothing to install, nothing hidden.
- `labs/common.py::chat()` is the single place a request is made. It never raises; errors
  come back as `ok=False` with `status`, so labs can count 429/503/timeouts.
- **TTFT counts reasoning tokens.** Reasoning models stream `reasoning_content` before
  `content`; a TTFT that ignores it measures 0 or infinity. (A real bug we hit in production
  load tests on Qwen3.5.)
- **Decode TPS** = `(completion_tokens − 1) / (e2e − TTFT)`, from server-reported usage.
- **tokens/chunk** = completion tokens ÷ streamed chunks; > ~1.5 hints at speculative decoding.
- Results append to `labs/results/<lab>.csv`, so you can compare runs across days and models.

## 5. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `403 error code: 1010` | Edge (Cloudflare) blocks the default `Python-urllib` User-Agent | `common.py` already sends a custom User-Agent; keep one in your own scripts too |
| `401 Invalid API key` | Wrong/expired key, or a key scoped to something other than serverless | New key from the settings page |
| `429` bursts | Your account's serverless rate limit | Lower `--conc`/`--rps`; read `x-ratelimit-*` headers the labs print |
| `503` under load | Shared serverless capacity is busy (best-effort tier) | Re-run at a quiet hour; that variance *is* the lesson in Lab 1.2 |
| `cached_tokens` always 0 | Model has no cached pricing, or your prefix isn't byte-stable | See Lab 2.1; try `prompt_cache_key` |
| TTFT is `-` but request succeeded | Non-streaming call, or a model that sent nothing before finishing | Labs stream by default; check `max_tokens` isn't tiny on a reasoning model |

## 6. Optional: go below the API

The labs observe a production system from outside. To see the inside, run an engine yourself:

- **vLLM on one GPU:** `pip install vllm`, serve a small Qwen model, then rerun Labs 1.1–2.1 against
  `http://localhost:8000/v1` (set `TOGETHER_BASE_URL`) while changing one engine flag at a time.
- **SMG on a laptop:** run [SMG](https://github.com/lightseekorg/smg) in front of Ollama with a
  ~0.5 GB model; no GPU needed. Change routing policies and watch the behaviour change.
- **Source to read while studying:** [vLLM](https://github.com/vllm-project/vllm),
  [SGLang](https://github.com/sgl-project/sglang), [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM),
  [SMG](https://github.com/lightseekorg/smg), [Dynamo](https://github.com/ai-dynamo/dynamo).
