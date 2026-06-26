#!/usr/bin/env python3
# coding=utf-8
"""把 ViSpec 预生成的多模态 .pt 反 tokenize 成 eagle3 训练用的 ShareGPT-style JSONL。

ViSpec .pt 每条:
    {
      "full_ids":   Tensor[S] long,   # 完整 token, 含 system/user/assistant header + 图像 patch tokens
                                       # (prompt_len 之后是 assistant 生成的回复)
      "prompt_len": int,
      "image_file": str,              # 相对 ViSpec 根目录
    }

prompt 段 decode 出来形如:
    <|im_start|>system
    {system}<|im_end|>
    <|im_start|>user
    {user_text}<|vision_start|><|image_pad|>...<|image_pad|><|vision_end|>{tail_text}<|im_end|>
    <|im_start|>assistant
也可能 image 在 user_text 之前 (`<|vision_start|>...<|vision_end|>{user_text}<|im_end|>`)。

eagle3 训练管线 (specforge.data.preprocessing.preprocess_vlm_conversations) 会自己:
  1. 读 image, 用 Qwen2.5-VL processor 重算 patch token 数;
  2. 套 specforge 自己的 qwen2-vl chat_template (system_prompt = "You are a helpful assistant.");
所以我们要做的就是: 把 user 纯文本 query 和 assistant 的纯文本回复抠出来, image 路径透传。

输出 JSONL 格式 (每行):
    {
      "conversations": [
        {"role": "user",      "content": "<user 纯文本, 不含 image 占位>"},
        {"role": "assistant", "content": "<assistant 纯文本, 末尾去掉 <|im_end|>>"}
      ],
      "image": "<图片绝对路径>"
    }

用法:
    python scripts/convert_vispec_to_eagle3.py \
        --src datasets/train/gen_mm_combined \
        --out datasets/train/qwen2.5-vl-7b-eagle3.jsonl \
        --tokenizer model/Qwen2.5-VL-7B-Instruct \
        --image-root /prj/corp/crd/morpheus/lasvegas/china-scratch/ziantan/ViSpec \
        --num-proc 32 \
        --max-len 4096
"""

import argparse
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

# Qwen2.5-VL 视觉与对话特殊 token id (训练数据中固定不变)
IM_START_ID = 151644
IM_END_ID = 151645
VISION_START_ID = 151652
VISION_END_ID = 151653
IMAGE_PAD_ID = 151655


def list_pt_files(src: str):
    files = []
    for root, _dirs, fnames in os.walk(src, followlinks=True):
        for fn in fnames:
            if fn.endswith(".pt"):
                files.append(os.path.join(root, fn))
    return sorted(files)


def _strip_image_block(ids: list) -> list:
    """删除 [VISION_START ... VISION_END] 区段 (含两端) 与中间所有 image_pad."""
    out = []
    in_vision = False
    for t in ids:
        if t == VISION_START_ID:
            in_vision = True
            continue
        if t == VISION_END_ID:
            in_vision = False
            continue
        if in_vision:
            continue
        if t == IMAGE_PAD_ID:
            # 防御: 万一没有 vision_start 包裹的孤儿 image_pad
            continue
        out.append(t)
    return out


def _extract_last_user_segment(prompt_ids: list) -> Optional[list]:
    """从 prompt token 序列里, 抠出最后一段 user turn 的 token (位于
    `<|im_start|>user\\n ... <|im_end|>`); 返回去掉头尾标记后的 token 列表 (不含 image)。"""
    # 反向找最后一个 IM_START 后面跟 user header
    n = len(prompt_ids)
    last_user_start = -1
    i = 0
    while i < n - 2:
        if prompt_ids[i] == IM_START_ID:
            # IM_START 后的 token 应是 'user' 或 'system' / 'assistant' 这种 role token
            # 用 decode 匹配更稳, 这里直接看常量
            # 找下一个 newline (id 198) 之间是 'user' (id=872 in qwen tokenizer)
            # 但 role token 可能多 token, 所以保守一点: 记下所有 IM_START, 后处理时挑最后一个 user
            pass
        i += 1
    # 更稳的做法: 切片所有 [IM_START ... IM_END] 段, 找 role 为 user 的最后一段
    return _slice_last_role_segment(prompt_ids, role_name=b"user")


def _slice_last_role_segment(prompt_ids: list, role_name: bytes, tokenizer=None) -> Optional[list]:
    """切出最后一个 role==role_name 的 turn 内容 (不含 IM_START/IM_END/role 标头)。"""
    # 先收集所有 IM_START 位置
    starts = [i for i, t in enumerate(prompt_ids) if t == IM_START_ID]
    if not starts:
        return None

    target_role = role_name.decode() if isinstance(role_name, bytes) else role_name

    # 反向遍历, 找第一个 role 为 target 的 turn
    for s in reversed(starts):
        # turn 形如: IM_START <role tokens> \n <content> IM_END (或到序列末尾)
        # role 由 IM_START 之后到第一个换行 (198) 之间的 token decode 得到
        nl = None
        end = None
        for j in range(s + 1, len(prompt_ids)):
            if prompt_ids[j] == 198 and nl is None:  # newline
                nl = j
            if prompt_ids[j] == IM_END_ID:
                end = j
                break
        if nl is None:
            continue
        role_tokens = prompt_ids[s + 1 : nl]
        # 用 tokenizer decode role 段做匹配
        if tokenizer is None:
            # 静态匹配: qwen 里 'user' = [872], 'system' = [8948], 'assistant' = [77091]
            STATIC = {"user": [872], "system": [8948], "assistant": [77091]}
            if role_tokens != STATIC.get(target_role, None):
                continue
        else:
            decoded = tokenizer.decode(role_tokens, skip_special_tokens=False).strip()
            if decoded != target_role:
                continue

        content_start = nl + 1
        content_end = end if end is not None else len(prompt_ids)
        return prompt_ids[content_start:content_end]
    return None


_IMG_PLACEHOLDER_RE = re.compile(r"<\|vision_start\|>(?:<\|image_pad\|>)*<\|vision_end\|>")


# ----------------------- worker (per-process tokenizer) -----------------------
_WORKER_TOK = None  # 每个 worker 进程内的 tokenizer 单例


def _init_worker(tokenizer_path: str):
    global _WORKER_TOK
    _WORKER_TOK = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def _process_one(args):
    fp, image_root, max_len, min_gen_tokens = args
    try:
        rec = torch.load(fp, map_location="cpu", weights_only=False)
    except Exception as exc:
        return ("err_load", str(exc), None)
    full_ids = rec["full_ids"]
    if full_ids.dim() == 2:
        full_ids = full_ids[0]
    full_ids = full_ids[:max_len]
    seq = full_ids.shape[0]
    prompt_len = min(int(rec["prompt_len"]), seq)
    gen_len = seq - prompt_len
    if gen_len < min_gen_tokens:
        return ("skip_short", None, None)

    ids_list = full_ids.tolist()
    prompt_ids = ids_list[:prompt_len]
    gen_ids = ids_list[prompt_len:]

    # 抠最后一个 user turn 的 content
    user_content_ids = _slice_last_role_segment(
        prompt_ids, role_name="user", tokenizer=_WORKER_TOK
    )
    if user_content_ids is None:
        return ("skip_no_user", None, None)

    # 去掉 image 块, 只剩纯文本 token
    user_text_ids = _strip_image_block(user_content_ids)
    user_text = _WORKER_TOK.decode(user_text_ids, skip_special_tokens=False).strip()
    # 兜底再 regex 清一遍 (防 tokenizer 解码出 placeholder 字面量)
    user_text = _IMG_PLACEHOLDER_RE.sub("", user_text).strip()
    if not user_text:
        # 有的 ViSpec 样本 prompt 是纯图片无文字, eagle3 chat_template 仍能跑, 但为稳起见跳过
        return ("skip_empty_user", None, None)

    # assistant 回复: 去掉末尾 IM_END (生成段最后一个 token 通常是 IM_END=151645)
    if gen_ids and gen_ids[-1] == IM_END_ID:
        gen_ids = gen_ids[:-1]
    if gen_ids and gen_ids[-1] == 198:  # 末尾 newline
        gen_ids = gen_ids[:-1]
    asst_text = _WORKER_TOK.decode(gen_ids, skip_special_tokens=False)
    asst_text = _IMG_PLACEHOLDER_RE.sub("", asst_text).strip()
    if not asst_text:
        return ("skip_empty_asst", None, None)

    image_file = rec["image_file"]
    abs_image = (
        image_file
        if os.path.isabs(image_file)
        else os.path.join(image_root, image_file)
    )

    obj = {
        "conversations": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": asst_text},
        ],
        "image": abs_image,
    }
    return ("ok", obj, fp)


def main():
    parser = argparse.ArgumentParser(
        description="Convert ViSpec multimodal .pt to eagle3 ShareGPT JSONL"
    )
    parser.add_argument(
        "--src", default="datasets/train/gen_mm_combined", help="ViSpec .pt 目录"
    )
    parser.add_argument(
        "--out",
        default="datasets/train/qwen2.5-vl-7b-eagle3.jsonl",
        help="输出 JSONL",
    )
    parser.add_argument(
        "--tokenizer",
        default="model/Qwen2.5-VL-7B-Instruct",
        help="用来 decode prompt/gen 的 tokenizer 路径 (与 target 模型一致)",
    )
    parser.add_argument(
        "--image-root",
        default="/prj/corp/crd/morpheus/lasvegas/china-scratch/ziantan/ViSpec",
    )
    parser.add_argument("--max-len", type=int, default=4096)
    parser.add_argument(
        "--min-gen-tokens",
        type=int,
        default=32,
        help="生成段少于此值的丢弃 (与 dflash 转换保持一致)",
    )
    parser.add_argument("--num-proc", type=int, default=16)
    parser.add_argument(
        "--limit", type=int, default=0, help="只处理前 N 个 .pt (调试用), 0=全部"
    )
    args = parser.parse_args()

    files = list_pt_files(args.src)
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError(f"未在 {args.src} 找到 .pt 文件")
    print(f"[convert_vispec_to_eagle3] 共 {len(files)} 个 .pt -> {args.out}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    counts = {"ok": 0, "skip_short": 0, "skip_no_user": 0,
              "skip_empty_user": 0, "skip_empty_asst": 0, "err_load": 0}

    work = [(fp, args.image_root, args.max_len, args.min_gen_tokens) for fp in files]

    with open(args.out, "w") as fout, ProcessPoolExecutor(
        max_workers=args.num_proc,
        initializer=_init_worker,
        initargs=(args.tokenizer,),
    ) as ex:
        for status, payload, _src in tqdm(
            ex.map(_process_one, work, chunksize=64),
            total=len(work),
            desc="converting",
        ):
            counts[status] = counts.get(status, 0) + 1
            if status == "ok":
                fout.write(json.dumps(payload, ensure_ascii=False) + "\n")

    print(f"[convert_vispec_to_eagle3] 完成: {counts}")
    print(f"[convert_vispec_to_eagle3] 输出: {args.out}")


if __name__ == "__main__":
    main()
