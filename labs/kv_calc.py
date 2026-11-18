#!/usr/bin/env python3
"""Lab 2.3 — GPU memory math: weights, KV cache per token, and how many users fit.

The single most useful back-of-envelope in inference. Offline, no API calls.

  KV bytes/token (standard / GQA attention) = 2 (K and V) x layers x kv_heads x head_dim x bytes
  KV bytes/token (MLA, DeepSeek-style)      = layers x (kv_lora_rank + rope_dim) x bytes
  weights (GB)                              = params (B) x bytes per param
  KV budget (GB)                            = gpus x gpu_mem x mem_fraction - weights - per-request state
  max concurrent sequences                  = KV budget / (KV bytes per token x context per sequence)

Hybrid models (Mamba / GDN / KDA linear attention) also hold a fixed-size recurrent state
per running request; pass --state-mb-per-seq to subtract it (Qwen3.5: ~144 MB fp32, ~73 MB bf16).

Examples:
  python3 labs/kv_calc.py --preset llama-3.1-8b --gpus 1 --gpu-mem 80 --context 32000
  python3 labs/kv_calc.py --preset deepseek-v3 --kv-bytes 1 --weight-bytes 1 --gpus 8 --gpu-mem 180
  python3 labs/kv_calc.py --layers 64 --kv-heads 8 --head-dim 128 --params-b 32 --gpus 1 --gpu-mem 80
  python3 labs/kv_calc.py --selftest
"""
import argparse
import sys

GiB = 1024 ** 3

# Public architecture numbers (from each model's config.json on Hugging Face).
PRESETS = {
    "llama-3.1-8b":  dict(layers=32, kv_heads=8, head_dim=128, params_b=8.0),
    "llama-3.3-70b": dict(layers=80, kv_heads=8, head_dim=128, params_b=70.6),
    "qwen3-32b":     dict(layers=64, kv_heads=8, head_dim=128, params_b=32.8),
    # MLA: KV is a compressed latent (kv_lora_rank 512) + decoupled RoPE key (64) per layer
    "deepseek-v3":   dict(layers=61, mla_dim=512 + 64, params_b=671.0),
}


def kv_bytes_per_token(layers, kv_heads=None, head_dim=None, kv_bytes=2, mla_dim=None):
    if mla_dim:
        return layers * mla_dim * kv_bytes
    return 2 * layers * kv_heads * head_dim * kv_bytes


def plan(layers, params_b, gpus, gpu_mem_gb, mem_fraction=0.9, weight_bytes=2, kv_bytes=2,
         kv_heads=None, head_dim=None, mla_dim=None, context=8192, state_mb_per_seq=0.0):
    per_tok = kv_bytes_per_token(layers, kv_heads, head_dim, kv_bytes, mla_dim)
    weights_gb = params_b * weight_bytes  # decimal GB, matches how model sizes are quoted
    budget_gb = gpus * gpu_mem_gb * mem_fraction - weights_gb
    per_seq_gb = per_tok * context / 1e9 + state_mb_per_seq / 1e3
    return {
        "kv_bytes_per_token": per_tok,
        "kv_kib_per_token": per_tok / 1024,
        "weights_gb": weights_gb,
        "kv_budget_gb": budget_gb,
        "kv_tokens_total": int(budget_gb * 1e9 / per_tok) if budget_gb > 0 else 0,
        "gb_per_seq_at_context": per_seq_gb,
        "max_seqs_at_context": int(budget_gb / per_seq_gb) if budget_gb > 0 else 0,
        "fits": budget_gb > 0,
    }


def _selftest():
    # Llama-3.1-8B BF16: 2*32*8*128*2 = 131,072 B = 128 KiB/token (an "eighth of a megabyte")
    assert kv_bytes_per_token(32, 8, 128, 2) == 131_072
    # 128k context for one user = 16 GiB, about the size of the 8B model's weights
    assert kv_bytes_per_token(32, 8, 128, 2) * 131_072 == 16 * GiB
    # Llama-3.3-70B BF16: 320 KiB/token; FP8 KV halves it
    assert kv_bytes_per_token(80, 8, 128, 2) == 327_680 and kv_bytes_per_token(80, 8, 128, 1) == 163_840
    # DeepSeek-V3 MLA BF16: 61 * 576 * 2 = 70,272 B (~69 KiB), vs several MiB/token as plain MHA
    assert kv_bytes_per_token(61, mla_dim=576) == 70_272
    # Production check (Qwen3.5-122B-A10B FP8 on 1xB200, 0.85 mem fraction, ~125 GB weights):
    # KV headroom = 0.85*180 - 125 = 28 GB, the number our runbook measured.
    # (layer/head numbers here are placeholders; only the budget is asserted)
    p = plan(layers=48, params_b=125, gpus=1, gpu_mem_gb=180, mem_fraction=0.85, weight_bytes=1,
             kv_heads=2, head_dim=256)
    assert abs(p["kv_budget_gb"] - 28.0) < 1e-6, p["kv_budget_gb"]
    # A model that doesn't fit must say so
    assert not plan(layers=80, params_b=70.6, gpus=1, gpu_mem_gb=80, kv_heads=8, head_dim=128)["fits"]
    print("kv_calc.py selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--layers", type=int)
    ap.add_argument("--kv-heads", type=int)
    ap.add_argument("--head-dim", type=int)
    ap.add_argument("--mla-dim", type=int, help="kv_lora_rank + rope dim for MLA models")
    ap.add_argument("--params-b", type=float, help="total parameters in billions (MoE: ALL experts)")
    ap.add_argument("--weight-bytes", type=float, default=2, help="2=BF16, 1=FP8, 0.5=FP4")
    ap.add_argument("--kv-bytes", type=float, default=2, help="2=BF16 KV, 1=FP8 KV")
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--gpu-mem", type=float, default=80, help="GB per GPU: H100 80, H200 141, B200 180")
    ap.add_argument("--mem-fraction", type=float, default=0.9)
    ap.add_argument("--context", type=int, default=8192, help="tokens held per sequence")
    ap.add_argument("--state-mb-per-seq", type=float, default=0.0)
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    cfg = dict(PRESETS.get(a.preset, {}))
    for k in ("layers", "kv_heads", "head_dim", "mla_dim", "params_b"):
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)
    if "layers" not in cfg or "params_b" not in cfg:
        sys.exit("Give --preset or at least --layers, --params-b and (--kv-heads + --head-dim) or --mla-dim")
    p = plan(gpus=a.gpus, gpu_mem_gb=a.gpu_mem, mem_fraction=a.mem_fraction, weight_bytes=a.weight_bytes,
             kv_bytes=a.kv_bytes, context=a.context, state_mb_per_seq=a.state_mb_per_seq, **cfg)
    print(f"KV per token        : {p['kv_kib_per_token']:.1f} KiB ({int(p['kv_bytes_per_token']):,} B)")
    print(f"weights             : {p['weights_gb']:.1f} GB")
    print(f"KV budget           : {p['kv_budget_gb']:.1f} GB  ({a.gpus} x {a.gpu_mem} GB x {a.mem_fraction})")
    if not p["fits"]:
        print("DOES NOT FIT: weights exceed the memory budget. Add GPUs (TP), quantize, or both.")
        return
    print(f"KV tokens (total)   : {p['kv_tokens_total']:,}")
    print(f"per sequence @ {a.context:,} : {p['gb_per_seq_at_context']:.2f} GB")
    print(f"max concurrent seqs : {p['max_seqs_at_context']}  (upper bound; engines reserve extra for activations)")


if __name__ == "__main__":
    main()
