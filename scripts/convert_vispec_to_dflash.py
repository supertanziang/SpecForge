#!/usr/bin/env python3
# coding=utf-8
"""把 ViSpec 预生成的多模态 .pt 数据转成 DFlash 训练用的 JSONL 格式。

ViSpec .pt 每条:
    {
      "full_ids":   Tensor[S] long,   # 完整 token 序列(prompt + target 生成的长回复),
                                       # 已含图像占位 token(Qwen2.5-VL image_token_id=151655)
      "prompt_len": int,              # prompt 长度; 生成段 = full_ids[prompt_len:]
      "image_file": str,              # 相对 ViSpec 根目录的图片路径
    }

DFlash 训练格式(每行一个 JSON):
    {
      "input_ids":  list[int],        # = full_ids
      "loss_mask":  list[int],        # = [0]*prompt_len + [1]*(S-prompt_len), 长度 S
      "image_file": str,              # 绝对路径(已拼 ViSpec 根), VLM 算 target hidden 时 Image.open 用
    }

attention_mask 不落盘(全 1, 训练时按 input_ids 长度现生成, 省体积)。

用法:
    python scripts/convert_vispec_to_dflash.py \
        --src datasets/train/gen_mm_combined \
        --out datasets/train/qwen2.5-vl-7b-dflash.jsonl \
        --image-root /prj/corp/crd/morpheus/lasvegas/china-scratch/ziantan/ViSpec \
        --max-len 4096 \
        --min-gen-tokens 32
"""

import argparse
import json
import os

import torch
from tqdm import tqdm


def list_pt_files(src: str):
    """递归收集 *.pt(followlinks=True: gen_mm_combined 下是软链接拼装的多份数据)。"""
    files = []
    for root, _dirs, fnames in os.walk(src, followlinks=True):
        for fn in fnames:
            if fn.endswith(".pt"):
                files.append(os.path.join(root, fn))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(
        description="Convert ViSpec multimodal .pt files to DFlash JSONL"
    )
    parser.add_argument(
        "--src",
        type=str,
        default="datasets/train/gen_mm_combined",
        help="ViSpec .pt 目录(递归扫描, 含软链接)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="datasets/train/qwen2.5-vl-7b-dflash.jsonl",
        help="输出 JSONL 路径",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default="/prj/corp/crd/morpheus/lasvegas/china-scratch/ziantan/ViSpec",
        help="image_file 是相对此根目录的路径, 转换时拼成绝对路径",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=4096,
        help="full_ids 截断长度(与训练 --max-length 对齐)",
    )
    parser.add_argument(
        "--min-gen-tokens",
        type=int,
        default=32,
        help="生成段(loss_mask=1)少于此值的样本丢弃(对齐 dflash 的 2*block_size 过滤)",
    )
    parser.add_argument(
        "--check-image",
        action="store_true",
        help="转换时校验图片文件存在(慢但安全); 默认不校验",
    )
    args = parser.parse_args()

    files = list_pt_files(args.src)
    if not files:
        raise FileNotFoundError(f"未在 {args.src} 找到任何 .pt 文件")
    print(f"找到 {len(files)} 个 .pt 文件, 开始转换 -> {args.out}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    n_written = 0
    n_skip_short = 0
    n_skip_noimg = 0
    with open(args.out, "w") as fout:
        for fp in tqdm(files, desc="converting"):
            rec = torch.load(fp, map_location="cpu", weights_only=False)
            full_ids = rec["full_ids"]
            if full_ids.dim() == 2:
                full_ids = full_ids[0]
            full_ids = full_ids[: args.max_len]
            seq = full_ids.shape[0]
            prompt_len = min(int(rec["prompt_len"]), seq)

            gen_len = seq - prompt_len
            if gen_len < args.min_gen_tokens:
                n_skip_short += 1
                continue

            image_file = rec["image_file"]
            abs_image = (
                image_file
                if os.path.isabs(image_file)
                else os.path.join(args.image_root, image_file)
            )
            if args.check_image and not os.path.exists(abs_image):
                n_skip_noimg += 1
                continue

            loss_mask = [0] * prompt_len + [1] * gen_len  # 长度 == seq

            fout.write(
                json.dumps(
                    {
                        "input_ids": full_ids.tolist(),
                        "loss_mask": loss_mask,
                        "image_file": abs_image,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n_written += 1

    print(
        f"完成: 写出 {n_written} 条, "
        f"跳过(生成段<{args.min_gen_tokens}) {n_skip_short} 条"
        + (f", 跳过(图片缺失) {n_skip_noimg} 条" if args.check_image else "")
    )
    print(f"输出: {args.out}")


if __name__ == "__main__":
    main()
