"""
离线 profiling: 测 Qwen3.5-35B-A3B DFlash 的 draft model (DM) 在真实 prompt 长度下的
prefill / decode 时间, 配合 benchmark 端测的 target model (TM) prefill/decode, 用来判断
端到端推理时间到底卡在哪一段。

DM (DFlashDraftModel) 结构 (model/Qwen3.5-35B-A3B-DFlash):
  - 8 层 Qwen3DFlash decoder, hidden=2048, 非因果 cross-attention;
  - 每层 K/V 来自 target 的 5 层 hidden ([1,10,19,28,37]) 拼接经 fc(10240->2048);
  - parallel drafting: 一次 forward 用 (1 bonus + k mask) 个 query token 并行出 k 个 draft。

测量口径 (用 cuda.Event, 每个阶段 warmup 后取多次中位):
  - DM-prefill: 对整段 prompt 的 target_hidden 做 fc + hidden_norm + K/V 投影
               (对齐 vLLM 的 precompute_and_store_context_kv, context KV 只算一次);
  - DM-decode : 一个 draft step = (1+k) 个 query token 对 ctx_len 个 KV 做 cross-attn 前向。
               这是投机解码里每验证一轮 DM 的真实开销。

target_hidden 用合成张量 (shape 决定算力, 数值不影响 kernel 计时), prompt_len 用真实样本的
近似值 (见 --prompt-len, 由 benchmark 端实测 prompt_tokens 填入)。
"""

import argparse
import time

import torch
from transformers import AutoConfig, AutoModel


def _sync_time(fn, iters: int, warmup: int = 3) -> float:
    """返回单次调用的中位耗时 (ms), 用 cuda.Event 计时。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft-path", default="./model/Qwen3.5-35B-A3B-DFlash")
    ap.add_argument("--prompt-len", type=int, required=True,
                    help="该样本的 prompt token 数 (含 vision token), 由 benchmark 端实测填入")
    ap.add_argument("--num-spec", type=int, default=5, help="num_speculative_tokens (k)")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    device = "cuda"
    dtype = torch.bfloat16

    cfg = AutoConfig.from_pretrained(args.draft_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.draft_path, trust_remote_code=True, dtype=dtype,
    ).to(device).eval()

    H = cfg.hidden_size
    n_tgt = len(cfg.dflash_config["target_layer_ids"])
    ctx = args.prompt_len
    k = args.num_spec
    q_len = 1 + k  # bonus + k mask tokens

    # 合成 target_hidden: [B, ctx, n_tgt*H] (5 层拼接), 数值随机不影响计时
    target_hidden = torch.randn(1, ctx, n_tgt * H, device=device, dtype=dtype)
    noise_embed = torch.randn(1, q_len, H, device=device, dtype=dtype)

    # ---- DM-prefill: fc + hidden_norm + 每层 K/V 投影 (context 只算一次) ----
    def prefill():
        th = model.hidden_norm(model.fc(target_hidden))
        for layer in model.layers:
            attn = layer.self_attn
            _ = attn.k_proj(th)
            _ = attn.v_proj(th)
        return th

    prefill_ms = _sync_time(prefill, args.iters)

    # ---- DM-decode: 一个完整 draft step 前向 (q_len query 对 ctx KV cross-attn) ----
    # position_ids 须覆盖 context+query 全长: dflash 内部 k=cat([k_ctx,k_noise]) 长度为
    # ctx+q_len, apply_rotary_pos_emb 对整个 k 套 cos/sin, 对 q 只取最后 q_len 个位置。
    pos_ids = torch.arange(ctx + q_len, device=device).unsqueeze(0)

    @torch.inference_mode()
    def decode():
        return model(
            position_ids=pos_ids,
            noise_embedding=noise_embed,
            target_hidden=target_hidden,
            attention_mask=None,
            use_cache=False,
        )

    decode_ms = _sync_time(decode, args.iters)

    print(f"[DM {args.label}] prompt_len(ctx)={ctx} k={k} hidden={H} layers={cfg.num_hidden_layers}")
    print(f"  DM-prefill (context K/V 投影, 算一次):  {prefill_ms:.3f} ms")
    print(f"  DM-decode  (一个 draft step, {q_len} query): {decode_ms:.3f} ms")


if __name__ == "__main__":
    main()
