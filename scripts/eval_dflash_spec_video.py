#!/usr/bin/env python
"""DFlash video-adapter 投机解码评测 (纯 PyTorch, 不依赖 SGLang/vLLM)。

为什么要这个脚本:
  benchmark_sd_vlm.py 走外部 vLLM/SGLang 的原生 DFLASH 后端, 那里不加载也不执行
  video_adapter 的 compress, 所以训出的 adapter 在那条路上等于没生效。仓库里唯一会
  调 compress 的推理代码是 DFlashDraftModel.spec_generate, 但它原本是死代码 (无调用方)。
  本脚本把 spec_generate 接成可跑的评测入口, 对每条样本分别跑:
    - 开 adapter (disable_adapter=False): 视觉 token 被压成 budget 个, draft KV 变短;
    - 关 adapter (disable_adapter=True):  draft KV 保留完整视觉上下文 (= 普通 dflash)。
  对比两者的 acceptance length / tokens-per-second / draft KV 长度 (压缩比), 验证
  compress 是否带来加速、是否掉接受率。

数据: 复用预 tokenize 的 JSONL (input_ids/loss_mask/image_files 或 image_file)。
  视频样本 = image_files 多帧; 图文样本 = image_file 单图。两者走同一处理路径
  (抽帧成多图, 非 native video), 与训练端 generate_dflash_data 完全一致。

用法 (7B):
  python scripts/eval_dflash_spec_video.py \
      --target-model-path .../Qwen2.5-VL-7B-Instruct \
      --draft-path        .../qwen2.5-vl-7b-dflash-stage3-adapter-frozen/epoch_X_step_Y \
      --data-path         datasets/train/qwen2.5-vl-7b-dflash_video.jsonl \
      --is-vlm --min-pixels 50176 --max-pixels 200704 \
      --num-samples 20 --max-new-tokens 128
"""

import argparse
import inspect
import json
import os
import sys
import time

import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor

# Make `specforge` importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from specforge.modeling.draft.dflash import DFlashDraftModel  # noqa: E402
from specforge.data.online_mm_dataset import OnlineMMDataset  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="DFlash video-adapter spec-decode eval")
    p.add_argument("--target-model-path", required=True)
    p.add_argument(
        "--draft-path",
        required=True,
        help="Trained draft checkpoint dir (contains config.json + weights, "
        "including video_adapter params).",
    )
    p.add_argument("--data-path", required=True, help="Pre-tokenized JSONL")
    p.add_argument("--is-vlm", action="store_true")
    p.add_argument("--min-pixels", type=int, default=None)
    p.add_argument("--max-pixels", type=int, default=None)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--num-samples", type=int, default=20)
    p.add_argument(
        "--modes",
        default="on,off",
        help="Comma list of adapter modes to run: 'on' (compress), 'off' (control), 'ar' (pure target autoregressive baseline).",
    )
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument(
        "--output-dir",
        default=None,
        help="Where to dump the JSON result (default: results/).",
    )
    return p.parse_args()


def resolve_submodule(root, candidates, what):
    """Return the first attribute path in `candidates` that exists on `root`.

    VLM wrappers nest embed_tokens / lm_head differently across model families
    (Qwen2.5-VL vs Qwen3.5-VL-MoE), so probe a few known layouts instead of
    hardcoding one.
    """
    for path in candidates:
        obj = root
        ok = True
        for attr in path.split("."):
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False
                break
        if ok:
            return obj
    raise AttributeError(
        f"Could not locate {what} on target model; tried: {candidates}. "
        f"Inspect the model structure and add the right path."
    )


def build_vision_inputs(processor, input_ids, image_files, image_token_id):
    """Replicate generate_dflash_data's processor path for one sample.

    Uses the full processor(images=..., text=...) call so that multi-frame
    samples get the same VIDEO_TOTAL_PIXELS budget splitting as at generation
    time, producing matching vision-token counts.
    """
    if not image_files:
        return {}
    assert processor is not None, "VLM eval requires a processor"
    images = [Image.open(p).convert("RGB") for p in image_files]
    content = [{"type": "image"} for _ in image_files]
    content.append({"type": "text", "text": "placeholder"})
    dummy_text = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(images=images, text=dummy_text, return_tensors="pt").to(
        input_ids.device
    )
    out = {}
    if "pixel_values" in inputs:
        out["pixel_values"] = inputs["pixel_values"]
    if "image_grid_thw" in inputs:
        out["image_grid_thw"] = inputs["image_grid_thw"]
        if image_token_id is not None:
            merge_size = getattr(processor.image_processor, "merge_size", 2)
            grid = inputs["image_grid_thw"]
            n_vision = int((grid.prod(dim=-1) // (merge_size**2)).sum().item())
            n_placeholder = int((input_ids == image_token_id).sum().item())
            assert n_vision == n_placeholder, (
                f"vision token mismatch: processor emits {n_vision} for "
                f"{len(images)} frame(s) but input_ids has {n_placeholder} "
                f"placeholders. Check min/max_pixels match generation time."
            )
    return out


def main():
    args = parse_args()
    device = "cuda"
    dtype = torch.bfloat16
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    # ---- processor / tokenizer ----
    processor = None
    tokenizer = None
    if args.is_vlm:
        proc_kwargs = {}
        if args.min_pixels is not None:
            proc_kwargs["min_pixels"] = args.min_pixels
        if args.max_pixels is not None:
            proc_kwargs["max_pixels"] = args.max_pixels
        processor = AutoProcessor.from_pretrained(
            args.target_model_path,
            trust_remote_code=args.trust_remote_code,
            **proc_kwargs,
        )
        tokenizer = processor.tokenizer
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)

    # ---- target model (bare HF, vision tower included for VLM) ----
    print(f"[load] target: {args.target_model_path}")
    if args.is_vlm:
        from transformers import AutoModelForImageTextToText

        device_map_env = os.environ.get("SPECFORGE_TARGET_DEVICE_MAP")
        extra = {}
        sharded = False
        if device_map_env:
            extra["device_map"] = device_map_env
            sharded = True
        target = AutoModelForImageTextToText.from_pretrained(
            args.target_model_path,
            torch_dtype=dtype,
            trust_remote_code=args.trust_remote_code,
            **extra,
        ).eval()
        if not sharded:
            target = target.to(device)
    else:
        from transformers import AutoModelForCausalLM

        target = AutoModelForCausalLM.from_pretrained(
            args.target_model_path,
            torch_dtype=dtype,
            output_hidden_states=True,
            trust_remote_code=args.trust_remote_code,
        ).eval().to(device)

    image_token_id = getattr(target.config, "image_token_id", None)

    # Resolve embed_tokens / lm_head for the (possibly nested) VLM structure.
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

    # ---- draft model (+ video_adapter) from trained checkpoint ----
    print(f"[load] draft: {args.draft_path}")
    draft = (
        DFlashDraftModel.from_pretrained(args.draft_path, torch_dtype=dtype)
        .to(device)
        .eval()
    )
    has_adapter = draft.video_adapter is not None
    print(
        f"[info] draft has video_adapter={has_adapter}, "
        f"block_size={draft.block_size}, target_layer_ids={draft.target_layer_ids}"
    )
    if not has_adapter and "on" in modes:
        print("[warn] draft has no video_adapter; 'on' mode == 'off' mode.")

    stop_token_ids = []
    if tokenizer is not None and tokenizer.eos_token_id is not None:
        stop_token_ids = [tokenizer.eos_token_id]

    # ---- data ----
    ds = OnlineMMDataset(args.data_path, max_len=args.max_length)
    n = min(args.num_samples, len(ds))
    print(f"[data] {args.data_path}: using {n}/{len(ds)} samples")

    # ---- eval loop ----
    # per-mode accumulators
    agg = {m: {"accept": [], "gen_tokens": 0, "time": 0.0, "kv_before": [], "kv_after": []} for m in modes}

    for i in range(n):
        rec = ds[i]
        input_ids = rec["input_ids"].unsqueeze(0).to(device)
        image_files = rec["image_files"]

        vis = {}
        if args.is_vlm and image_files:
            vis = build_vision_inputs(
                processor, input_ids, image_files, image_token_id
            )
            # mm_token_type_ids for Qwen3.5-VL M-RoPE (gate by forward signature)
            fwd_params = inspect.signature(target.forward).parameters
            if "mm_token_type_ids" in fwd_params and image_token_id is not None:
                mm = torch.zeros_like(input_ids, dtype=torch.int32)
                mm[input_ids == image_token_id] = 1
                vis["mm_token_type_ids"] = mm

        for mode in modes:
            if mode == "ar":
                # Pure autoregressive baseline (target only, no draft)
                torch.cuda.synchronize()
                t0 = time.time()
                with torch.no_grad():
                    gen_kwargs = dict(
                        input_ids=input_ids,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                    )
                    if vis.get("pixel_values") is not None:
                        gen_kwargs["pixel_values"] = vis["pixel_values"]
                    if vis.get("image_grid_thw") is not None:
                        gen_kwargs["image_grid_thw"] = vis["image_grid_thw"]
                    if vis.get("mm_token_type_ids") is not None:
                        gen_kwargs["mm_token_type_ids"] = vis["mm_token_type_ids"]
                    out_ids = target.generate(**gen_kwargs)
                torch.cuda.synchronize()
                dt = time.time() - t0
                gen = int(out_ids.shape[1] - input_ids.shape[1])
                a = agg[mode]
                a["accept"].append(1.0)
                a["gen_tokens"] += gen
                a["time"] += dt
                a["kv_before"].append(0)
                a["kv_after"].append(0)
            else:
                disable = mode == "off"
                torch.cuda.synchronize()
                t0 = time.time()
                out_ids = draft.spec_generate(
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
                torch.cuda.synchronize()
                dt = time.time() - t0

                accept = draft._spec_generate_acceptance_lengths
                gen = int(out_ids.shape[1] - input_ids.shape[1])
                a = agg[mode]
                if accept:
                    a["accept"].append(sum(accept) / len(accept))
                a["gen_tokens"] += gen
                a["time"] += dt
                a["kv_before"].append(draft._spec_generate_kv_len_before)
                a["kv_after"].append(draft._spec_generate_kv_len_after)

        if (i + 1) % 5 == 0 or i == n - 1:
            msg = f"[{i+1}/{n}]"
            for mode in modes:
                a = agg[mode]
                al = sum(a["accept"]) / len(a["accept"]) if a["accept"] else 0.0
                msg += f"  {mode}: accept={al:.2f}"
            print(msg, flush=True)

    # ---- summary ----
    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    summary = {
        "target": args.target_model_path,
        "draft": args.draft_path,
        "data": args.data_path,
        "num_samples": n,
        "max_new_tokens": args.max_new_tokens,
        "modes": {},
    }
    print("\n===== DFlash spec-decode eval =====")
    header = f"{'mode':>6} | {'avg_accept_len':>14} | {'tokens/s':>9} | {'speedup':>7} | {'kv_before':>10} | {'kv_after':>9} | {'compress_x':>10}"
    print(header)
    print("-" * len(header))
    ar_tps = None
    if "ar" in modes:
        a = agg["ar"]
        ar_tps = a["gen_tokens"] / a["time"] if a["time"] > 0 else None
    for mode in modes:
        a = agg[mode]
        al = mean(a["accept"])
        tps = a["gen_tokens"] / a["time"] if a["time"] > 0 else 0.0
        kvb = mean(a["kv_before"])
        kva = mean(a["kv_after"])
        cx = kvb / kva if kva > 0 else 1.0
        speedup = tps / ar_tps if ar_tps and ar_tps > 0 else 0.0
        summary["modes"][mode] = {
            "avg_accept_len": al,
            "tokens_per_s": tps,
            "speedup_vs_ar": speedup,
            "kv_len_before": kvb,
            "kv_len_after": kva,
            "compress_ratio": cx,
        }
        speedup_str = f"{speedup:.2f}x" if speedup > 0 else "  base"
        print(
            f"{mode:>6} | {al:>14.3f} | {tps:>9.2f} | {speedup_str:>7} | {kvb:>10.1f} | {kva:>9.1f} | {cx:>9.2f}x"
        )

    out_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
    )
    os.makedirs(out_dir, exist_ok=True)
    tag = os.path.basename(args.data_path).replace(".jsonl", "")
    out_path = os.path.join(out_dir, f"eval_dflash_spec_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
