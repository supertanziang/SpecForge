"""将 gen_mm_combined (.pt 目录格式) 转为 OnlineMMDataset 可读的 JSONL。

用法:
    conda activate specforge
    python scripts/convert_gen_mm_to_jsonl.py \
        --input-dir datasets/train/gen_mm_combined \
        --output datasets/train/gen_mm_combined.jsonl \
        --image-base /prj/corp/crd/morpheus/lasvegas/china-scratch/ziantan/ViSpec
"""

import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--image-base", type=str,
        default="/prj/corp/crd/morpheus/lasvegas/china-scratch/ziantan/ViSpec",
    )
    parser.add_argument("--subdirs", type=str, default="online_0,online_1,online_2")
    args = parser.parse_args()

    subdirs = [s.strip() for s in args.subdirs.split(",") if s.strip()]
    pt_files = []
    for sub in subdirs:
        d = os.path.join(args.input_dir, sub)
        if not os.path.isdir(d):
            print(f"WARN: {d} 不存在, 跳过")
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".pt"):
                pt_files.append(os.path.join(d, f))

    print(f"共 {len(pt_files)} 个 .pt 文件, 从 {len(subdirs)} 个子目录")

    written = 0
    skipped = 0
    with open(args.output, "w") as out:
        for pt_path in tqdm(pt_files, desc="converting"):
            try:
                d = torch.load(pt_path, map_location="cpu", weights_only=False)
            except Exception as e:
                print(f"  SKIP {pt_path}: {e}")
                skipped += 1
                continue

            full_ids = d["full_ids"].tolist()
            prompt_len = int(d["prompt_len"])
            image_file = d["image_file"]

            # image_file 是相对路径,拼成绝对路径
            abs_image = os.path.join(args.image_base, image_file)
            if not os.path.exists(abs_image):
                print(f"  SKIP {pt_path}: image not found {abs_image}")
                skipped += 1
                continue

            loss_mask = [0] * prompt_len + [1] * (len(full_ids) - prompt_len)

            rec = {
                "input_ids": full_ids,
                "loss_mask": loss_mask,
                "image_file": abs_image,
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    print(f"完成: {written} 条写入 {args.output}, {skipped} 条跳过")


if __name__ == "__main__":
    main()
