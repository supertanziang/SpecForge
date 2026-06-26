"""
把图文评测集 (MM-Vet / ScienceQA / MME / TextVQA) 统一导出成
「图片文件 + jsonl 索引」, 供 vLLM 环境的 benchmark_sd_vlm.py 读取。

为什么要预导出:
  - 评测引擎 vLLM 只装在 `vllm` conda 环境, 而该环境的 `datasets` 库是坏的
    (load_from_disk 都 import 不了)。所以读原始 Arrow/HF 数据这一步必须在
    `specforge` 环境做, 落盘成 jpg + jsonl 后, 评测端只用 PIL 读图。

统一索引格式 (datasets/eval/<name>_index.jsonl), 每行:
    {"id": ..., "question": <拼好的完整 prompt>, "image_path": ..., "dataset": <name>}

每个集都用 shuffle(seed=42) 打乱后再导出 (与 ViSpec 评测对齐), 保证取前 N 条有代表性。

用法 (在 specforge 环境):
    python scripts/export_eval_datasets.py                       # 三个集各导出 <=200 条
    python scripts/export_eval_datasets.py --datasets mmvet,sqa  # 只导部分
    python scripts/export_eval_datasets.py --max-per-dataset 100 --force
"""

import argparse
import json
import os
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
EVAL_DIR = os.path.join(ROOT_DIR, "datasets", "eval")

SQA_LETTERS = "ABCDEFGHIJ"


def _save_image(image, path: str):
    image.convert("RGB").save(path, "JPEG", quality=95)


def export_mmvet(max_n: int):
    """MM-Vet: Arrow, 字段 {id, image, question}。"""
    from datasets import load_from_disk

    ds = load_from_disk(os.path.join(EVAL_DIR, "mmvet"))
    rows = list(ds)
    random.Random(42).shuffle(rows)
    if max_n > 0:
        rows = rows[:max_n]

    img_dir = os.path.join(EVAL_DIR, "mmvet_images")
    os.makedirs(img_dir, exist_ok=True)
    items = []
    for i, row in enumerate(rows):
        qid = row.get("id", str(i))
        img_path = os.path.join(img_dir, f"{qid}.jpg")
        _save_image(row["image"], img_path)
        items.append(
            {
                "id": qid,
                "question": row["question"],
                "image_path": img_path,
                "dataset": "mmvet",
            }
        )
    return items


def export_mme(max_n: int):
    """MME: llava_mme.jsonl + 图片散落在 MME_Benchmark/<category>/[images/]<name>。"""
    from PIL import Image

    mme_root = os.path.join(EVAL_DIR, "MME")
    img_root = os.path.join(
        mme_root, "MME_Benchmark_release_version", "MME_Benchmark"
    )
    # 递归建立 文件名 -> 完整路径 索引 (jsonl 里 image 只有文件名)
    name2path = {}
    for dp, _, fs in os.walk(img_root):
        for fn in fs:
            if fn.lower().endswith((".jpg", ".png", ".jpeg")):
                name2path.setdefault(fn, os.path.join(dp, fn))

    rows = []
    with open(os.path.join(mme_root, "llava_mme.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    random.Random(42).shuffle(rows)
    if max_n > 0:
        rows = rows[:max_n]

    img_dir = os.path.join(EVAL_DIR, "mme_images")
    os.makedirs(img_dir, exist_ok=True)
    items = []
    for row in rows:
        qid = str(row["question_id"])
        src = name2path.get(row["image"])
        if src is None:
            continue
        img_path = os.path.join(img_dir, f"{qid}.jpg")
        _save_image(Image.open(src), img_path)
        # text 只取首行 (MME 问题自带 "Please answer yes or no")
        question = row["text"].partition("\n")[0]
        items.append(
            {
                "id": qid,
                "question": question,
                "image_path": img_path,
                "dataset": "mme",
            }
        )
    return items


def export_textvqa(max_n: int):
    """TextVQA: Arrow, 字段 {image, question, answers, ...}。

    TextVQA 本身 GT 是短答案 (OCR 问答), 但这里评测的是**速度/加速比**, 实际输出
    长度由 prompt 决定 —— 故在问题后追加一句, 诱导模型读图中文字并给出详细解释,
    产生中长输出 (与 mme 的 3-token yes/no 形成对照, 更能体现投机解码价值)。
    """
    from datasets import load_from_disk

    ds = load_from_disk(os.path.join(EVAL_DIR, "textvqa"))
    rows = [r for r in ds if r.get("image") is not None]
    random.Random(42).shuffle(rows)
    if max_n > 0:
        rows = rows[:max_n]

    suffix = (
        " Read the text in the image, answer the question, "
        "and explain your reasoning in detail."
    )
    img_dir = os.path.join(EVAL_DIR, "textvqa_images")
    os.makedirs(img_dir, exist_ok=True)
    items = []
    for i, row in enumerate(rows):
        qid = str(row.get("question_id", f"textvqa_{i}"))
        img_path = os.path.join(img_dir, f"{qid}.jpg")
        _save_image(row["image"], img_path)
        items.append(
            {
                "id": qid,
                "question": row["question"].strip() + suffix,
                "image_path": img_path,
                "dataset": "textvqa",
            }
        )
    return items


def export_sqa(max_n: int):
    """ScienceQA: test split, 过滤无图题, 把 question + choices 拼成多选 prompt。"""
    from datasets import load_from_disk

    ds = load_from_disk(os.path.join(EVAL_DIR, "sqa", "dataset"))
    split = ds["test"] if "test" in ds else ds[list(ds.keys())[0]]

    rows = [r for r in split if r.get("image") is not None]
    random.Random(42).shuffle(rows)
    if max_n > 0:
        rows = rows[:max_n]

    img_dir = os.path.join(EVAL_DIR, "sqa_images")
    os.makedirs(img_dir, exist_ok=True)
    items = []
    for i, row in enumerate(rows):
        qid = f"sqa_{i}"
        img_path = os.path.join(img_dir, f"{qid}.jpg")
        _save_image(row["image"], img_path)

        choices = row.get("choices") or []
        opts = " ".join(
            f"({SQA_LETTERS[j]}) {c}" for j, c in enumerate(choices)
        )
        hint = (row.get("hint") or "").strip()
        parts = []
        if hint:
            parts.append(f"Context: {hint}")
        parts.append(f"Question: {row['question']}")
        if opts:
            parts.append(f"Options: {opts}")
            parts.append("Answer with the option's letter and a brief explanation.")
        question = "\n".join(parts)

        items.append(
            {
                "id": qid,
                "question": question,
                "image_path": img_path,
                "dataset": "sqa",
            }
        )
    return items


EXPORTERS = {
    "mmvet": export_mmvet,
    "mme": export_mme,
    "sqa": export_sqa,
    "textvqa": export_textvqa,
}


def main():
    parser = argparse.ArgumentParser(
        description="Export MM-Vet / ScienceQA / MME / TextVQA to jpg + jsonl index (run in specforge env)"
    )
    parser.add_argument(
        "--datasets", type=str, default="mmvet,sqa,mme",
        help="逗号分隔, 子集 of {mmvet,sqa,mme,textvqa}",
    )
    parser.add_argument(
        "--max-per-dataset", type=int, default=200,
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
