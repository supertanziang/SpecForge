#!/usr/bin/env python
"""把视频评测集 (Video-MME / LongVideoBench) 的标准索引转成 DFlash 纯 PyTorch
评测脚本 (scripts/eval_dflash_spec_video.py) 能直接吃的「预 tokenize JSONL」。

为什么需要它:
  eval_dflash_spec_video.py 是唯一会真正执行 video_adapter.compress 的评测入口
  (vLLM/SGLang 后端不加载也不执行 adapter), 但它只吃预 tokenize 的 JSONL
  (input_ids + image_files), 不吃 datasets/eval/{ds}_index.jsonl 这种「video_path +
  question」的标准索引。本脚本做这一层转换:

    标准索引 (video_path / question) --抽帧--> N 帧 jpg --chat 模板--> tokenize
      --> {input_ids(仅 prompt), loss_mask(全0占位), image_files} 一行

  抽帧策略 / 抽帧索引公式 / chat 模板与训练数据生成 (gen_video_dflash_data.py) 完全
  一致, 故训练分布与评测分布对齐。注意:

  - 评测只跑 prompt + 在线让 target forward, 不需要 target 生成续写, 所以这里
    不加载 target、不调 model.generate, 只用 processor 取 prompt 的 input_ids。
  - loss_mask 全 0 仅为占位 (OnlineMMDataset 要求该字段存在); spec_generate 不读它。
  - 索引里的 question 字段已含选项 + ANSWER_SUFFIX (见 export_video_eval_datasets.py),
    这里直接使用, 不再追加后缀。
  - min/max_pixels 必须与跑 eval 时一致, 否则 eval 的 vision-token assert 会崩。

用法:
  conda activate dflash
  python scripts/export_video_eval_to_dflash_jsonl.py \
      --target-model-path model/Qwen2.5-VL-7B-Instruct \
      --datasets videomme,longvideobench \
      --min-pixels 50176 --max-pixels 200704 \
      --max-length 4096 --trust-remote-code
"""

import argparse
import json
import os
import sys

from PIL import Image
from transformers import AutoProcessor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
EVAL_DIR = os.path.join(ROOT_DIR, "datasets", "eval")

# 复用现成工具, 不重写抽帧/帧数策略。
sys.path.insert(0, SCRIPT_DIR)
from export_video_eval_datasets import adaptive_num_frames  # noqa: E402
from gen_video_dflash_data import extract_frames, get_video_duration  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="export video eval index -> dflash jsonl")
    p.add_argument("--target-model-path", required=True, help="VLM 路径, 取其 processor")
    p.add_argument(
        "--datasets",
        default="videomme,longvideobench",
        help="逗号分隔, 对应 datasets/eval/{name}_index.jsonl",
    )
    p.add_argument("--min-pixels", type=int, default=50176)
    p.add_argument("--max-pixels", type=int, default=200704)
    p.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="prompt 超过此长度的样本跳过 (与 eval 的 --max-length 对齐)",
    )
    p.add_argument(
        "--max-samples", type=int, default=0, help="0=全部; 否则每个数据集最多导出这么多条"
    )
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    processor = AutoProcessor.from_pretrained(
        args.target_model_path,
        trust_remote_code=args.trust_remote_code,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )

    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        index_path = os.path.join(EVAL_DIR, f"{ds}_index.jsonl")
        if not os.path.exists(index_path):
            print(f"[skip] 找不到索引: {index_path}")
            continue
        frames_dir = os.path.join(EVAL_DIR, f"{ds}_frames")
        out_path = os.path.join(EVAL_DIR, f"{ds}_dflash_video.jsonl")

        rows = [json.loads(l) for l in open(index_path) if l.strip()]
        print(f"\n[{ds}] {len(rows)} 条索引 -> {out_path}")

        written = 0
        skipped_novideo = 0
        skipped_long = 0
        skipped_err = 0
        with open(out_path, "w") as fout:
            for row in rows:
                if args.max_samples and written >= args.max_samples:
                    break
                question = row.get("question")

                # 支持两种索引: video_path(视频,抽帧) 或 image_path(单图,直接用)
                video_path = row.get("video_path")
                image_path = row.get("image_path")
                if video_path and os.path.exists(video_path):
                    num_frames = adaptive_num_frames(get_video_duration(video_path))
                    try:
                        frame_paths = extract_frames(video_path, num_frames, frames_dir)
                    except Exception as e:  # noqa: BLE001
                        print(f"  [skip] 抽帧失败 {video_path}: {e}")
                        skipped_err += 1
                        continue
                    if not frame_paths:
                        skipped_err += 1
                        continue
                elif image_path and os.path.exists(image_path):
                    frame_paths = [image_path]
                else:
                    skipped_novideo += 1
                    continue

                # chat 模板: N 个 image 占位 + 题干 (question 已含选项与作答后缀)
                content = [{"type": "image"} for _ in frame_paths]
                content.append({"type": "text", "text": question})
                text = processor.apply_chat_template(
                    [{"role": "user", "content": content}],
                    tokenize=False,
                    add_generation_prompt=True,
                )

                images = [Image.open(p).convert("RGB") for p in frame_paths]
                try:
                    inputs = processor(images=images, text=text, return_tensors="pt")
                except Exception as e:  # noqa: BLE001
                    print(f"  [skip] processor 失败 id={row.get('id')}: {e}")
                    skipped_err += 1
                    continue

                input_ids = inputs["input_ids"][0]
                if len(input_ids) >= args.max_length:
                    skipped_long += 1
                    continue

                fout.write(
                    json.dumps(
                        {
                            "input_ids": input_ids.tolist(),
                            "loss_mask": [0] * len(input_ids),
                            "image_files": frame_paths,
                            "id": row.get("id"),
                            "num_frames": len(frame_paths),
                        }
                    )
                    + "\n"
                )
                written += 1

        print(
            f"[{ds}] 写出 {written} 条 -> {out_path}\n"
            f"        跳过: 无视频 {skipped_novideo}, 超长(>={args.max_length}) "
            f"{skipped_long}, 出错 {skipped_err}"
        )


if __name__ == "__main__":
    main()
