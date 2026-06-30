#!/usr/bin/env python
"""单条视频上 TM/DM × prefill/decode 的分段计时 (adapter on vs off)。

为什么要这个脚本:
  results/eval_dflash_spec_videomme_*.json 只有端到端 tokens/s 与压缩比, 看不出 video
  adapter 把时间花在/省在哪一段。本脚本在 DFlashDraftModel.spec_generate 的四个边界用
  cuda.Event 插桩 (不重写那段微妙的循环), 对单条 VideoMME 样本分别跑 adapter on/off,
  报告:
    - TM-prefill : target 首次前向 (vision encode + LLM prefill);
    - TM-decode  : 每轮对投机 block 的 target verify 前向 (合并均值);
    - DM-decode  : 每个 draft step 的 draft 前向 (合并均值; on 模式 KV 被压短 → 更快);
    - DM-prefill : video_adapter.compress 的一次性开销 (仅 on 模式; off 标 N/A)。

  插桩方式: monkeypatch target.forward / draft.forward / video_adapter.compress,
  每次调用记一对 cuda.Event(enable_timing=True), 全程不 sync; 跑完一次 spec_generate
  后统一 torch.cuda.synchronize() 再 elapsed_time, 避免 per-call sync 扭曲计时。

  off 模式下 compress 不执行 (draft.forward 调用未传 input_ids), 故 DM-prefill = N/A;
  off 模式首个 draft step 内含全长 context K/V 投影, 该开销自然计入 DM-decode 均值。

用法 (7B + VideoMME):
  python scripts/profile_dflash_video_tmdm.py \
      --target-model-path /prj/.../ViSpec/model/Qwen2.5-VL-7B-Instruct \
      --draft-path ziantan/outputs/qwen2.5-vl-7b-dflash-stage3-adapter-frozen/epoch_9_step_93000 \
      --data-path datasets/eval/videomme_dflash_video.jsonl \
      --is-vlm --min-pixels 50176 --max-pixels 200704 --max-length 4096 \
      --max-new-tokens 128 --index 0 --modes on,off
"""

import argparse
import inspect
import json
import os
import sys
import time

import torch
from transformers import AutoProcessor

# Make `specforge` and the sibling eval script importable when run as a script.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _SCRIPT_DIR)

from specforge.modeling.draft.dflash import DFlashDraftModel  # noqa: E402
from specforge.data.online_mm_dataset import OnlineMMDataset  # noqa: E402
# Reuse the exact model-loading / vision-input helpers from the eval script so the
# profiled path is identical to the evaluation path.
from eval_dflash_spec_video import (  # noqa: E402
    resolve_submodule,
    build_vision_inputs,
)


class EventTimer:
    """累积 cuda.Event 对; 全程不 sync, 收尾统一 elapsed_time 求各段时间。"""

    def __init__(self):
        # bucket -> list[(start_event, end_event)]
        self.buckets = {}
        self.enabled = False  # warmup 期间关掉, 不污染正式计时
        self._states = []  # per-wrapper call counters, zeroed on reset()

    def reset(self):
        self.buckets = {}
        # 关键: 每次正式测量前把各 wrapper 的调用序号清零, 否则上一 mode 跑完后
        # target.forward 的 n 已 >0, 下一 mode 的首次调用不会再被判成 'TM-prefill'。
        for st in self._states:
            st["n"] = 0

    def wrap(self, obj, attr, classify):
        """monkeypatch obj.attr; classify(call_index, *args) -> bucket 名。"""
        orig = getattr(obj, attr)
        state = {"n": 0}
        self._states.append(state)

        def wrapped(*args, **kwargs):
            if not self.enabled:
                return orig(*args, **kwargs)
            bucket = classify(state["n"], args, kwargs)
            state["n"] += 1
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            out = orig(*args, **kwargs)
            e.record()
            self.buckets.setdefault(bucket, []).append((s, e))
            return out

        setattr(obj, attr, wrapped)
        return orig  # caller can restore if needed

    def summarize(self):
        """sync 后返回 {bucket: {count, total_ms, mean_ms}}。"""
        torch.cuda.synchronize()
        out = {}
        for bucket, pairs in self.buckets.items():
            times = [s.elapsed_time(e) for s, e in pairs]
            total = sum(times)
            out[bucket] = {
                "count": len(times),
                "total_ms": total,
                "mean_ms": total / len(times) if times else 0.0,
            }
        return out


def parse_args():
    p = argparse.ArgumentParser(description="TM/DM prefill/decode profiling on one video")
    p.add_argument("--target-model-path", required=True)
    p.add_argument("--draft-path", required=True)
    p.add_argument("--data-path", required=True, help="Pre-tokenized JSONL")
    p.add_argument("--is-vlm", action="store_true")
    p.add_argument("--min-pixels", type=int, default=None)
    p.add_argument("--max-pixels", type=int, default=None)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--index", type=int, default=0, help="Which sample in the JSONL")
    p.add_argument(
        "--truncate-to-prompt",
        action="store_true",
        help="Cut input_ids to the loss_mask==0 prefix (drop the baked-in answer) "
        "so the model generates fresh tokens. Use for training-set samples.",
    )
    p.add_argument("--modes", default="on,off", help="Comma list: on,off")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda"
    dtype = torch.bfloat16
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    # ---- processor / tokenizer ----
    assert args.is_vlm, "this profiler targets VLM video samples"
    proc_kwargs = {}
    if args.min_pixels is not None:
        proc_kwargs["min_pixels"] = args.min_pixels
    if args.max_pixels is not None:
        proc_kwargs["max_pixels"] = args.max_pixels
    processor = AutoProcessor.from_pretrained(
        args.target_model_path, trust_remote_code=args.trust_remote_code, **proc_kwargs
    )
    tokenizer = processor.tokenizer

    # ---- target model ----
    print(f"[load] target: {args.target_model_path}")
    from transformers import AutoModelForImageTextToText

    target = (
        AutoModelForImageTextToText.from_pretrained(
            args.target_model_path,
            torch_dtype=dtype,
            trust_remote_code=args.trust_remote_code,
        )
        .eval()
        .to(device)
    )
    image_token_id = getattr(target.config, "image_token_id", None)
    embed_tokens = resolve_submodule(
        target,
        [
            "model.embed_tokens",
            "model.language_model.embed_tokens",
            "model.model.embed_tokens",
            "language_model.model.embed_tokens",
        ],
        "embed_tokens",
    )
    lm_head = resolve_submodule(
        target, ["lm_head", "model.lm_head", "language_model.lm_head"], "lm_head"
    )

    # ---- draft model (+ video_adapter) ----
    print(f"[load] draft: {args.draft_path}")
    draft = (
        DFlashDraftModel.from_pretrained(args.draft_path, torch_dtype=dtype)
        .to(device)
        .eval()
    )
    has_adapter = draft.video_adapter is not None
    print(f"[info] draft has video_adapter={has_adapter}")
    if not has_adapter and "on" in modes:
        print("[warn] no video_adapter; 'on' == 'off'.")

    stop_token_ids = []
    if tokenizer.eos_token_id is not None:
        stop_token_ids = [tokenizer.eos_token_id]

    # ---- data: one sample ----
    ds = OnlineMMDataset(args.data_path, max_len=args.max_length)
    assert 0 <= args.index < len(ds), f"index {args.index} out of range (n={len(ds)})"
    rec = ds[args.index]
    input_ids = rec["input_ids"].unsqueeze(0).to(device)
    image_files = rec["image_files"]
    # Training-set samples bake the assistant answer + EOS into input_ids, so spec
    # decoding from the tail hits EOS immediately (gen_tokens~3). Truncate to the
    # prompt portion (loss_mask==0 prefix) so the model actually generates; all
    # visual tokens live in the prompt, so vision-token counts are unaffected.
    if args.truncate_to_prompt:
        loss_mask = rec["loss_mask"]
        ans_start = int((loss_mask == 1).nonzero()[0].item()) if (loss_mask == 1).any() else loss_mask.numel()
        input_ids = input_ids[:, :ans_start]
    print(f"[data] sample #{args.index}: prompt_len={input_ids.shape[1]} "
          f"frames={len(image_files) if image_files else 0}")

    vis = {}
    if image_files:
        vis = build_vision_inputs(processor, input_ids, image_files, image_token_id)
        fwd_params = inspect.signature(target.forward).parameters
        if "mm_token_type_ids" in fwd_params and image_token_id is not None:
            mm = torch.zeros_like(input_ids, dtype=torch.int32)
            mm[input_ids == image_token_id] = 1
            vis["mm_token_type_ids"] = mm

    # ---- install timers (monkeypatch) ----
    timer = EventTimer()
    # target.forward: call 0 = TM-prefill, rest = TM-decode (verify)
    timer.wrap(
        target, "forward",
        lambda n, a, k: "TM-prefill" if n == 0 else "TM-decode",
    )
    # draft.forward (the DFlashDraftModel itself is called via __call__ -> forward):
    # every call = one DM-decode step.
    timer.wrap(draft, "forward", lambda n, a, k: "DM-decode")
    # adapter.compress: one-shot per prefill = DM-prefill (only fires in 'on' mode).
    if has_adapter:
        timer.wrap(draft.video_adapter, "compress", lambda n, a, k: "DM-prefill")

    def run_once(disable):
        return draft.spec_generate(
            target=target,
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            stop_token_ids=stop_token_ids,
            temperature=args.temperature,
            pixel_values=vis.get("pixel_values"),
            image_grid_thw=vis.get("image_grid_thw"),
            mm_token_type_ids=vis.get("mm_token_type_ids"),
            disable_adapter=disable,
            embed_tokens=embed_tokens,
            lm_head=lm_head,
        )

    results = {}
    for mode in modes:
        disable = mode == "off"
        # warmup (timers off so events aren't recorded)
        timer.enabled = False
        with torch.no_grad():
            run_once(disable)
        # measured run
        timer.reset()
        timer.enabled = True
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out_ids = run_once(disable)
        seg = timer.summarize()  # this syncs
        wall = (time.perf_counter() - t0) * 1000
        timer.enabled = False

        accept = draft._spec_generate_acceptance_lengths
        gen = int(out_ids.shape[1] - input_ids.shape[1])
        results[mode] = {
            "segments_ms": seg,
            "wall_ms": wall,
            "gen_tokens": gen,
            "avg_accept_len": (sum(accept) / len(accept)) if accept else 0.0,
            "kv_len_before": draft._spec_generate_kv_len_before,
            "kv_len_after": draft._spec_generate_kv_len_after,
            "compress_ratio": (
                draft._spec_generate_kv_len_before
                / draft._spec_generate_kv_len_after
                if draft._spec_generate_kv_len_after > 0
                else 1.0
            ),
        }

    # ---- report ----
    print("\n===== TM/DM prefill/decode profiling (single video) =====")
    print(f"sample #{args.index}  prompt_len={input_ids.shape[1]}  "
          f"frames={len(image_files) if image_files else 0}  "
          f"max_new_tokens={args.max_new_tokens}\n")
    seg_order = ["TM-prefill", "TM-decode", "DM-prefill", "DM-decode"]
    hdr = f"{'segment':>12} | " + " | ".join(f"{m+' total/mean(ms)':>22}" for m in modes)
    print(hdr)
    print("-" * len(hdr))
    for s in seg_order:
        row = f"{s:>12} | "
        cells = []
        for m in modes:
            d = results[m]["segments_ms"].get(s)
            if d is None:
                cells.append(f"{'N/A':>22}")
            else:
                cells.append(f"{d['total_ms']:>9.1f} / {d['mean_ms']:>7.3f} (x{d['count']})".rjust(22))
        print(row + " | ".join(cells))
    print("-" * len(hdr))
    for m in modes:
        r = results[m]
        print(f"[{m}] wall={r['wall_ms']:.1f}ms  gen_tokens={r['gen_tokens']}  "
              f"accept_len={r['avg_accept_len']:.3f}  "
              f"kv {r['kv_len_before']:.0f}->{r['kv_len_after']:.0f} "
              f"(x{r['compress_ratio']:.2f})")

    summary = {
        "target": args.target_model_path,
        "draft": args.draft_path,
        "data": args.data_path,
        "index": args.index,
        "prompt_len": int(input_ids.shape[1]),
        "num_frames": len(image_files) if image_files else 0,
        "max_new_tokens": args.max_new_tokens,
        "modes": results,
    }
    out_dir = args.output_dir or os.path.join(os.path.dirname(_SCRIPT_DIR), "results")
    os.makedirs(out_dir, exist_ok=True)
    tag = os.path.basename(args.data_path).replace(".jsonl", "")
    out_path = os.path.join(out_dir, f"profile_tmdm_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
