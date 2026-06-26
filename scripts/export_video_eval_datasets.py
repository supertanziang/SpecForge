"""
导出视频评测集 (Video-MME / LongVideoBench) 为「视频文件 + jsonl 索引」，
供 vllm 环境的 benchmark_sd_vlm.py 读取。

与图文版 export_eval_datasets.py 对齐:
  - 在 specforge 环境跑 (依赖 huggingface_hub / pandas);
  - 落盘成 mp4 + jsonl, 评测端只用 opencv 抽帧。

为什么不用 datasets.load_dataset / list_repo_files:
  - 这两个会拉取整个仓库的文件树 (Video-MME 视频仓库有 80w+ 文件), 极易触发
    HF 429 限流。这里改成: 只下载单个元数据文件 (parquet / lvb_val.json) +
    用 HfFileSystem.ls 非递归列视频目录 + hf_hub_download 单文件下载视频。

视频来源:
  - Video-MME: 元数据 lmms-lab/Video-MME (parquet), 视频 vid-modeling/videomme
    (仅 data/ 下约 155 个 YouTube 视频可用, 取交集)。
  - LongVideoBench: 元数据+视频均在 lccshunli/LongVideoBench
    (lvb_val.json + videos/videos/<id>.mp4)。

统一索引格式 (datasets/eval/<name>_index.jsonl), 每行:
    {"id", "question", "video_path", "duration", "num_frames", "dataset"}

帧数策略 (按时长自适应, 写入 num_frames 字段):
  - ≤10s: 4 帧 / 10-30s: 8 帧 / >30s: 16 帧 (上限, 控制 token 预算; 32 帧 x 长
    视频会超过 8k 上下文)

用法 (在 specforge 环境):
    python scripts/export_video_eval_datasets.py
    python scripts/export_video_eval_datasets.py --datasets videomme --max-per-dataset 50
    python scripts/export_video_eval_datasets.py --datasets longvideobench --max-per-dataset 50 --force
"""

import argparse
import json
import os
import random

from huggingface_hub import hf_hub_download, HfFileSystem

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
EVAL_DIR = os.path.join(ROOT_DIR, "datasets", "eval")

VIDEOMME_VIDEO_REPO = "vid-modeling/videomme"
VIDEOMME_META_REPO = "lmms-lab/Video-MME"
LVB_REPO = "lccshunli/LongVideoBench"

# 多选题默认只输出一个选项字母 (~2 token), decode 太短无法体现投机解码加速 (加速的是
# decode 阶段)。追加此后缀诱导模型先描述视频内容再逐步推理, 产生中长输出 (对齐图文版
# textvqa 的做法), 这样 speedup / accept_rate 才有意义。
ANSWER_SUFFIX = (
    "\nFirst describe what you observe in the video relevant to the question, "
    "then reason step by step, and finally answer with the option letter."
)


def adaptive_num_frames(duration_sec: float) -> int:
    if duration_sec <= 10:
        return 4
    elif duration_sec <= 30:
        return 8
    else:
        return 16


def _get_video_duration(path: str) -> float:
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


def export_videomme(max_n: int):
    """Video-MME: parquet 元数据 + vid-modeling/videomme 的 data/ 视频子集 (取交集)。"""
    import pandas as pd

    # 1) 下载元数据 parquet (单文件, 不触发限流)
    parquet_path = hf_hub_download(
        repo_id=VIDEOMME_META_REPO, repo_type="dataset",
        filename="videomme/test-00000-of-00001.parquet",
    )
    df = pd.read_parquet(parquet_path)

    # 2) 非递归列出视频仓库 data/ 下可用的 mp4 (约 155 个 YouTube 视频)
    fs = HfFileSystem()
    entries = fs.ls(f"datasets/{VIDEOMME_VIDEO_REPO}/data", detail=False)
    available = {
        e.split("/")[-1][:-4] for e in entries if e.endswith(".mp4")
    }
    print(f"  [videomme] {len(available)} 个视频可用 (data/ 下)")

    # 3) 元数据按 videoID 分组, 只保留有视频的, 每视频取第一题
    rows_by_video = {}
    for _, row in df.iterrows():
        vid = row["videoID"]
        if vid in available and vid not in rows_by_video:
            rows_by_video[vid] = row

    video_ids = list(rows_by_video.keys())
    random.Random(42).shuffle(video_ids)

    video_dir = os.path.join(EVAL_DIR, "videomme_videos")
    os.makedirs(video_dir, exist_ok=True)

    items = []
    for vid in video_ids:
        if max_n > 0 and len(items) >= max_n:
            break

        local_path = os.path.join(video_dir, f"{vid}.mp4")
        if not os.path.exists(local_path):
            print(f"  [download] videomme/{vid}")
            try:
                src = hf_hub_download(
                    repo_id=VIDEOMME_VIDEO_REPO, repo_type="dataset",
                    filename=f"data/{vid}.mp4",
                )
                os.symlink(src, local_path)
            except Exception as e:
                print(f"  [skip] {vid}: {str(e)[:80]}")
                continue

        duration_sec = _get_video_duration(local_path)
        num_frames = adaptive_num_frames(duration_sec)

        row = rows_by_video[vid]
        opts_str = " ".join(list(row["options"]))
        prompt = f"{row['question']}\n{opts_str}{ANSWER_SUFFIX}"

        items.append({
            "id": str(row["question_id"]),
            "question": prompt,
            "video_path": local_path,
            "duration": round(duration_sec, 1),
            "num_frames": num_frames,
            "dataset": "videomme",
        })

    return items


def export_longvideobench(max_n: int):
    """LongVideoBench: lvb_val.json 元数据 + videos/videos/<id>.mp4 单文件下载。"""
    meta_path = hf_hub_download(
        repo_id=LVB_REPO, repo_type="dataset", filename="lvb_val.json",
    )
    with open(meta_path) as f:
        rows = json.load(f)

    random.Random(42).shuffle(rows)

    video_dir = os.path.join(EVAL_DIR, "longvideobench_videos")
    os.makedirs(video_dir, exist_ok=True)

    items = []
    for row in rows:
        if max_n > 0 and len(items) >= max_n:
            break

        vid = row["video_id"]
        local_path = os.path.join(video_dir, f"{vid}.mp4")
        if not os.path.exists(local_path):
            print(f"  [download] lvb/{vid}")
            try:
                src = hf_hub_download(
                    repo_id=LVB_REPO, repo_type="dataset",
                    filename=f"videos/videos/{vid}.mp4",
                )
                os.symlink(src, local_path)
            except Exception as e:
                print(f"  [skip] {vid}: {str(e)[:80]}")
                continue

        duration_sec = row.get("duration") or _get_video_duration(local_path)
        num_frames = adaptive_num_frames(duration_sec)

        # candidates 是选项列表
        cands = row.get("candidates", [])
        opts_str = " ".join(
            f"({chr(65+i)}) {c}" for i, c in enumerate(cands)
        )
        prompt = f"{row['question']}\n{opts_str}{ANSWER_SUFFIX}"

        items.append({
            "id": str(row["id"]),
            "question": prompt,
            "video_path": local_path,
            "duration": round(float(duration_sec), 1),
            "num_frames": num_frames,
            "dataset": "longvideobench",
        })

    return items


EXPORTERS = {
    "videomme": export_videomme,
    "longvideobench": export_longvideobench,
}


def main():
    parser = argparse.ArgumentParser(
        description="Export Video-MME / LongVideoBench to mp4 + jsonl index (run in specforge env)"
    )
    parser.add_argument(
        "--datasets", type=str, default="videomme,longvideobench",
        help="逗号分隔, 子集 of {videomme,longvideobench}",
    )
    parser.add_argument(
        "--max-per-dataset", type=int, default=50,
        help="每个集最多导出多少条 (评测时再取前 N); <=0 表示全量",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="索引已存在时也重新导出 (默认存在则跳过)",
    )
    args = parser.parse_args()

    names = [d.strip() for d in args.datasets.split(",") if d.strip()]
    for name in names:
        if name not in EXPORTERS:
            raise ValueError(f"未知数据集: {name} (支持 {list(EXPORTERS)})")

        index_path = os.path.join(EVAL_DIR, f"{name}_index.jsonl")
        if os.path.exists(index_path) and not args.force:
            with open(index_path) as f:
                n = sum(1 for _ in f)
            print(f"[skip] {name}: 索引已存在 ({n} 条) -> {index_path} (--force 重导)")
            continue

        print(f"[export] {name} ...")
        items = EXPORTERS[name](args.max_per_dataset)
        with open(index_path, "w") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"[export] {name}: {len(items)} 条 -> {index_path}")


if __name__ == "__main__":
    main()
