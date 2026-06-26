#!/usr/bin/env python3
# coding=utf-8
"""从 dflash 同款 tokenized jsonl ({input_ids, loss_mask, image_file}) 离线算 eagle3
vocab_mapping (d2t / t2d 张量), 写到 train_eagle3.py 的 --vocab-mapping-path 指向的位置。

为啥不复用 specforge.data.preprocessing.generate_vocab_mapping_file:
    它要求 HF Dataset 形如 {input_ids, loss_mask}, 与我们 vlm-online-mm 路径用的
    OnlineMMDataset 不兼容; 而且 import specforge 又会拽到 yunchang。这里直接复刻
    100 行算法, 不依赖任何 specforge 模块。

用法:
    python scripts/build_vocab_mapping_from_jsonl.py \
        --jsonl datasets/train/qwen2.5-vl-7b-dflash.jsonl \
        --out cache/vocab_mapping/qwen2.5-vl-7b-eagle3.pt \
        --target-vocab-size 152064 \
        --draft-vocab-size 32000
"""

import argparse
import json
import os
from collections import Counter

import torch
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--target-vocab-size", type=int, default=152064)
    parser.add_argument("--draft-vocab-size", type=int, default=32000)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    print(f"[vocab_mapping] scan {args.jsonl}")
    token_dict = Counter()
    n = 0
    with open(args.jsonl) as f:
        for line in tqdm(f, desc="counting"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ids = rec["input_ids"]
            mask = rec["loss_mask"]
            for tid, m in zip(ids, mask):
                if m == 1:
                    token_dict[tid] += 1
            n += 1
    print(f"[vocab_mapping] samples={n}, unique tokens={len(token_dict)}")

    # 复刻 specforge.data.preprocessing.process_token_dict_to_mappings
    if len(token_dict) < args.draft_vocab_size:
        existing = set(token_dict.keys())
        for tid in range(args.target_vocab_size):
            if tid not in existing:
                token_dict[tid] = 0
            if len(token_dict) >= args.draft_vocab_size:
                break

    sorted_items = sorted(token_dict.items(), key=lambda x: -x[1])[: args.draft_vocab_size]
    target_ids = [tid for tid, _ in sorted_items]

    d2t = torch.tensor(target_ids, dtype=torch.long) - torch.arange(
        args.draft_vocab_size, dtype=torch.long
    )
    t2d = torch.zeros(args.target_vocab_size, dtype=torch.bool)
    t2d[target_ids] = True

    torch.save({"d2t": d2t, "t2d": t2d}, args.out)
    print(f"[vocab_mapping] saved: {args.out}")


if __name__ == "__main__":
    main()
