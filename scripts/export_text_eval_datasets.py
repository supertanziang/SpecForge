"""
把纯文本评测集 (MT-Bench / GSM8K / HumanEval) 统一导出成 jsonl 索引,
供 benchmark_sd.py (纯文本投机解码评测) 读取。

与图文版 export_eval_datasets.py 的区别:
  - 纯文本, 不导图片, 索引每行只有 {id, question, dataset};
  - 输出到 datasets/eval/<name>_text.jsonl (区别于图文版的 <name>_index.jsonl)。

统一索引格式 (datasets/eval/<name>_text.jsonl), 每行:
    {"id": ..., "question": <拼好的完整 prompt>, "dataset": <name>}

每个集 shuffle(seed=42) 后取前 N 条 (与图文版对齐), 保证取样有代表性。

数据来源:
  - mtbench  : 本地 benchmarks/mtbench.jsonl (取 turns[0], 单轮);
  - gsm8k    : HF load_dataset('gsm8k','main')['test'] (question 用 "Question: ...\nAnswer:", zero-shot);
  - humaneval: HF load_dataset('openai/openai_humaneval')['test'] (取 prompt, 代码补全)。

须在有 datasets 库的环境跑 (如 dflash / specforge 环境):
    conda activate dflash
    python scripts/export_text_eval_datasets.py                          # 三集各 <=200 条
    python scripts/export_text_eval_datasets.py --datasets gsm8k         # 只导部分
    python scripts/export_text_eval_datasets.py --max-per-dataset 100 --force
"""

import argparse
import json
import os
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
EVAL_DIR = os.path.join(ROOT_DIR, "datasets", "eval")
MTBENCH_PATH = os.path.join(ROOT_DIR, "benchmarks", "mtbench.jsonl")


def export_mtbench(max_n: int):
    """MT-Bench: 本地 benchmarks/mtbench.jsonl, 字段 {question_id, category, turns}。

    取 turns[0] 作 question (单轮, 对齐 benchmark_sd.py 原行为)。
    """
    rows = []
    with open(MTBENCH_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    random.Random(42).shuffle(rows)
    if max_n > 0:
        rows = rows[:max_n]
    items = []
    for i, row in enumerate(rows):
        turns = row.get("turns") or []
        if not turns:
            continue
        items.append(
            {
                "id": str(row.get("question_id", i)),
                "question": turns[0],
                "dataset": "mtbench",
            }
        )
    return items


def export_gsm8k(max_n: int):
    """GSM8K: HF gsm8k/main test split。

    question 用 "Question: {q}\\nAnswer:" (对齐 benchmarker/gsm8k.py 的
    get_one_example(..., include_answer=False)); zero-shot, 不加 few-shot
    (thinking 模式靠模型自己推理, 加 few-shot 会拉长 prompt 偏离 README setup)。
    """
    from datasets import load_dataset

    ds = load_dataset("gsm8k", "main")["test"]
    rows = list(ds)
    random.Random(42).shuffle(rows)
    if max_n > 0:
        rows = rows[:max_n]
    items = []
    for i, row in enumerate(rows):
        items.append(
            {
                "id": f"gsm8k_{i}",
                "question": "Question: " + row["question"] + "\nAnswer:",
                "dataset": "gsm8k",
            }
        )
    return items


def export_humaneval(max_n: int):
    """HumanEval: HF openai/openai_humaneval test split, question 用 prompt (代码补全)。"""
    from datasets import load_dataset

    ds = load_dataset("openai/openai_humaneval")["test"]
    rows = list(ds)
    random.Random(42).shuffle(rows)
    if max_n > 0:
        rows = rows[:max_n]
    items = []
    for i, row in enumerate(rows):
        items.append(
            {
                "id": str(row.get("task_id", f"humaneval_{i}")).replace("/", "_"),
                "question": row["prompt"],
                "dataset": "humaneval",
            }
        )
    return items


EXPORTERS = {
    "mtbench": export_mtbench,
    "gsm8k": export_gsm8k,
    "humaneval": export_humaneval,
}


def main():
    parser = argparse.ArgumentParser(
        description="Export MT-Bench / GSM8K / HumanEval to jsonl (run in env with datasets lib, e.g. dflash)"
    )
    parser.add_argument(
        "--datasets", type=str, default="mtbench,gsm8k,humaneval",
        help="逗号分隔, 子集 of {mtbench,gsm8k,humaneval}",
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

    os.makedirs(EVAL_DIR, exist_ok=True)
    names = [d.strip() for d in args.datasets.split(",") if d.strip()]
    for name in names:
        if name not in EXPORTERS:
            raise ValueError(f"未知数据集: {name} (支持 {list(EXPORTERS)})")

        index_path = os.path.join(EVAL_DIR, f"{name}_text.jsonl")
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
