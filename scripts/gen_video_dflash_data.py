#!/usr/bin/env python3
# coding=utf-8
"""为 dflash VLM 投机解码生成「视频训练数据」(直接产 dflash JSONL)。

== 为什么是「视频抽帧成多帧图」而不是原生 video ==

评测端 benchmark_sd_vlm.py 把视频当成「一串独立图片」喂模型: opencv 均匀抽 N 帧,
每帧编码成一个 image_url, 用 image_token(Qwen2.5-VL=151655), image_grid_thw 形状
[N,3], 完全不走 native video(无 video_token / pixel_values_videos / video_grid_thw)。

投机解码 draft 的命门是 target hidden 分布对齐: 评测是多帧图, 训练就必须多帧图。
所以本脚本同样抽帧成多图, 抽帧公式 + 自适应帧数都复用评测端 export_video_eval_datasets
/ benchmark_sd_vlm, 保证训练分布 == 评测分布。

== 产出格式(dflash JSONL, 每行一条, 与 convert_vispec_to_dflash.py 对齐) ==
    {
      "input_ids":   list[int],   # 完整序列(prompt + target 生成), 已含 N*帧 的 image_token
      "loss_mask":   list[int],   # [0]*prompt_len + [1]*gen_len
      "image_files": list[str],   # N 帧 jpg 绝对路径(顺序 == input_ids 里 image_token 段顺序)
    }

== 数据源 ==
lmms-lab/NExTQA 的 MC(multiple-choice) split: 8564 题 / 1000 视频 / 5 候选(a0~a4) +
answer 索引。视频在同仓库 videos.zip(6.2GB)。与评测集(Video-MME/LongVideoBench)
完全不同 -> 无泄漏。每个视频默认只取 1 题(--per-video 1), 量小质优。

== 用法(在 dflash 环境) ==
    # 1) 先下载并解压视频(videos.zip -> --video-dir)
    python -c "from huggingface_hub import hf_hub_download; \
        print(hf_hub_download('lmms-lab/NExTQA','videos.zip',repo_type='dataset'))"
    unzip <上面打印的路径> -d datasets/train/nextqa/

    # 2) 7B 生成(单卡, 多帧 token 重, max_pixels 砍半控预算)
    python scripts/gen_video_dflash_data.py \
        --model /path/to/Qwen2.5-VL-7B-Instruct \
        --video-dir datasets/train/nextqa/NExTVideo \
        --out datasets/train/qwen2.5-vl-7b-dflash_video.jsonl \
        --frames-dir datasets/train/nextqa_frames \
        --max-samples 4000 --per-video 1 \
        --min-pixels 50176 --max-pixels 200704

    # 3) 35B 生成(多卡 device_map=auto, pixels 留空走模型自带, 见 memory
    #    dflash-35b-vl-stage2-6gpu-launch)
    python scripts/gen_video_dflash_data.py \
        --model /path/to/Qwen3.5-35B-VL --device-map balanced \
        --video-dir datasets/train/nextqa/NExTVideo \
        --out datasets/train/qwen3.5-35b-vl-dflash_video.jsonl \
        --frames-dir datasets/train/nextqa_frames_35b \
        --max-samples 4000 --per-video 1

注意: --min-pixels/--max-pixels 必须与训练 train_dflash.py 用的值完全一致, 否则每帧
image_pad 数变, grid 对不上 input_ids, target forward 崩(dflash_target_model.py 已加
校验断言会提前抓到)。
"""

import argparse
import hashlib
import io
import json
import os
import sys

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

# 复用评测端的自适应帧数策略(MAX_FRAMES=48, >30s 按 duration/30 线性), 保证与评测一致。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_video_eval_datasets import adaptive_num_frames  # noqa: E402

# 与评测端 export_video_eval_datasets.ANSWER_SUFFIX 完全一致: 诱导模型先描述视频再逐步
# 推理最后给字母, 产生中长输出(否则多选只输出一个字母 ~2 token, decode 太短体现不出 SD 加速)。
ANSWER_SUFFIX = (
    "\nFirst describe what you observe in the video relevant to the question, "
    "then reason step by step, and finally answer with the option letter."
)


def build_question(row) -> str:
    """把 NExTQA MC 一行拼成题干 + (A)..(E) 选项 + ANSWER_SUFFIX, 对齐评测端格式。"""
    q = str(row["question"]).strip()
    if not q.endswith("?"):
        q = q + "?"
    opts = []
    for i in range(5):
        cand = row.get(f"a{i}")
        if cand is None or (isinstance(cand, float)):
            continue
        cand = str(cand).strip()
        if cand:
            opts.append(f"({chr(65 + i)}) {cand}")
    return f"{q}\n{' '.join(opts)}{ANSWER_SUFFIX}"


def get_video_duration(path: str) -> float:
    """用 cv2 取视频时长(秒); 读不出返回 60.0(落入 >30s 档)。对齐 export_video_eval_datasets。"""
    try:
        import cv2

        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        if fps > 0 and frame_count > 0:
            return frame_count / fps
    except Exception:
        pass
    return 60.0


def extract_frames(video_path: str, num_frames: int, out_dir: str, quality: int = 85):
    """均匀抽 num_frames 帧落盘成 jpg, 返回帧绝对路径 list。

    抽帧索引公式 [int(i*total/num_frames) for i in range(num_frames)] 与
    benchmark_sd_vlm.extract_video_frames 完全一致 -> 训练/评测抽到同样的帧。
    帧文件名带 video 标识 + 帧号, 同一视频重复调用幂等(已存在则跳过)。
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError(f"无法读取视频帧: {video_path}")

    vid_key = hashlib.md5(os.path.abspath(video_path).encode()).hexdigest()[:12]
    os.makedirs(out_dir, exist_ok=True)
    indices = [int(i * total / num_frames) for i in range(num_frames)]
    paths = []
    for j, idx in enumerate(indices):
        fp = os.path.join(out_dir, f"{vid_key}_n{num_frames}_f{j:02d}.jpg")
        if not os.path.exists(fp):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            Image.fromarray(rgb).save(fp, format="JPEG", quality=quality)
        paths.append(fp)
    cap.release()
    return paths


def find_video_file(video_dir: str, video_id) -> str:
    """NExTQA 的 video 字段是数字 id, 视频文件名为 <id>.mp4(解压后)。返回绝对路径或 None。"""
    for ext in (".mp4", ".webm", ".avi", ".mkv"):
        cand = os.path.join(video_dir, f"{video_id}{ext}")
        if os.path.exists(cand):
            return cand
    return None


def main():
    ap = argparse.ArgumentParser(description="生成 dflash VLM 视频训练数据(抽帧成多图)")
    ap.add_argument("--model", required=True, help="target 模型路径(Qwen2.5-VL-7B / Qwen3.5-35B-VL)")
    ap.add_argument("--video-dir", required=True, help="解压后的 NExTQA 视频目录(含 <id>.mp4)")
    ap.add_argument("--out", required=True, help="输出 dflash JSONL 路径")
    ap.add_argument("--frames-dir", required=True, help="抽帧 jpg 落盘目录(帧路径写进 image_files)")
    ap.add_argument("--mc-parquet", default=None,
                    help="NExTQA MC parquet 路径; 留空则自动从 HF 下载 lmms-lab/NExTQA")
    ap.add_argument("--max-samples", type=int, default=4000, help="最多生成多少条(质量过滤后)")
    ap.add_argument("--per-video", type=int, default=1, help="每个视频最多取几题(默认 1, 量小质优)")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="!=0 走采样; 1.0 保证生成段够长、token 多样(loss 信号足)")
    ap.add_argument("--max-len", type=int, default=4096, help="full_ids 截断长度(与训练 --max-length 对齐)")
    ap.add_argument("--min-gen-tokens", type=int, default=32, help="生成段短于此则丢弃(质量过滤)")
    ap.add_argument("--min-pixels", type=int, default=None,
                    help="processor min_pixels; 必须与训练一致! 7B 多帧建议 50176(=64*28*28)")
    ap.add_argument("--max-pixels", type=int, default=None,
                    help="processor max_pixels; 必须与训练一致! 7B 多帧建议 200704(=256*28*28) 控预算")
    ap.add_argument("--device-map", default="auto", help="HF device_map(35B 用 balanced 多卡)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-shards", type=int, default=1,
                    help="数据并行分片总数(多进程各跑一片, 用 CUDA_VISIBLE_DEVICES 隔离卡)")
    ap.add_argument("--shard-id", type=int, default=0,
                    help="当前进程的分片号 [0, num_shards)")
    args = ap.parse_args()

    import pandas as pd

    # 1) 载入 NExTQA MC parquet
    if args.mc_parquet:
        mc_path = args.mc_parquet
    else:
        from huggingface_hub import hf_hub_download

        mc_path = hf_hub_download(
            "lmms-lab/NExTQA", "MC/test-00000-of-00001.parquet", repo_type="dataset"
        )
    df = pd.read_parquet(mc_path)
    # 每视频限题数 + 打散(seed 固定可复现)
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    if args.per_video > 0:
        df = df.groupby("video", group_keys=False).head(args.per_video).reset_index(drop=True)

    # 数据并行分片: 第 shard_id 个进程只处理 iloc[shard_id::num_shards]。
    # max_samples 按分片均分(每片各自达标即可)。
    if args.num_shards > 1:
        df = df.iloc[args.shard_id :: args.num_shards].reset_index(drop=True)
        per_shard_samples = (args.max_samples + args.num_shards - 1) // args.num_shards
        print(
            f"[shard] shard {args.shard_id}/{args.num_shards}: {len(df)} 题, "
            f"目标 {per_shard_samples} 条"
        )
        args.max_samples = per_shard_samples
    print(f"[data] MC 处理 {len(df)} 条, 覆盖 {df['video'].nunique()} 视频")

    # 2) 载入 processor + 模型
    proc_kwargs = {}
    if args.min_pixels is not None:
        proc_kwargs["min_pixels"] = args.min_pixels
    if args.max_pixels is not None:
        proc_kwargs["max_pixels"] = args.max_pixels
    processor = AutoProcessor.from_pretrained(args.model, use_fast=True, **proc_kwargs)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, device_map=args.device_map, torch_dtype="auto"
    ).eval()
    eos_id = processor.tokenizer.eos_token_id

    # 3) 断点续跑: 已写出的行数(out 已存在则续写)
    done = 0
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = sum(1 for _ in f)
        print(f"[resume] {args.out} 已有 {done} 条, 续写")

    written = done
    skipped = 0
    fout = open(args.out, "a")
    try:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="generate"):
            if written >= args.max_samples:
                break
            video_path = find_video_file(args.video_dir, row["video"])
            if video_path is None:
                skipped += 1
                continue

            # a) 自适应帧数(复用评测策略) + 抽帧落盘
            num_frames = adaptive_num_frames(get_video_duration(video_path))
            try:
                frame_paths = extract_frames(video_path, num_frames, args.frames_dir)
            except Exception as e:
                print(f"[skip] 抽帧失败 {video_path}: {e}")
                skipped += 1
                continue
            if len(frame_paths) == 0:
                skipped += 1
                continue

            # b) 构造 prompt(多帧 image 占位 + 题干), apply_chat_template
            question = build_question(row)
            content = [{"type": "image"} for _ in frame_paths]
            content.append({"type": "text", "text": question})
            text = processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )

            # c) processor 现编码(多图), target generate
            images = [Image.open(p).convert("RGB") for p in frame_paths]
            try:
                inputs = processor(
                    images=images, text=text, return_tensors="pt"
                ).to(model.device)
            except Exception as e:
                print(f"[skip] processor 失败 video={row['video']}: {e}")
                skipped += 1
                continue
            prompt_len = int(inputs["input_ids"].shape[-1])
            if prompt_len >= args.max_len:
                skipped += 1  # 仅 prompt 就超长(帧太多), 没有生成空间, 丢弃
                continue

            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    do_sample=args.temperature != 0,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                )
            full_ids = out_ids[0].cpu()

            # d) EOS 裁剪(保留首个 EOS, 与 ViSpec gen_stage2 一致)
            gen_part = full_ids[prompt_len:]
            eos_pos = (gen_part == eos_id).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                full_ids = torch.cat(
                    [full_ids[:prompt_len], gen_part[: eos_pos[0].item() + 1]]
                )

            # e) 质量过滤 + 截断 + loss_mask
            gen_len = len(full_ids) - prompt_len
            if gen_len < args.min_gen_tokens:
                skipped += 1
                continue
            full_ids = full_ids[: args.max_len]
            gen_len = len(full_ids) - prompt_len
            loss_mask = [0] * prompt_len + [1] * gen_len

            fout.write(
                json.dumps(
                    {
                        "input_ids": full_ids.tolist(),
                        "loss_mask": loss_mask,
                        "image_files": frame_paths,
                    }
                )
                + "\n"
            )
            fout.flush()
            written += 1
    finally:
        fout.close()

    print(f"[done] 写出 {written} 条(本次 {written - done}), 跳过 {skipped} 条 -> {args.out}")


if __name__ == "__main__":
    main()
