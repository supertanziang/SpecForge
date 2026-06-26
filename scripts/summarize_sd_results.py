#!/usr/bin/env python3
# ==============================================================================
# 投机解码 (dflash / eagle3) 评测结果汇总脚本
# ------------------------------------------------------------------------------
# 扫描 results/ 下所有 benchmark_sd*.py 产出的 JSON, 抽取每个 (模型, 引擎, 方法,
# 数据集) 组合的核心指标 (speedup / accept_length / accept_rate), 汇总成一张
# 便于横向对比的表。解决 results/ 下 JSON 越攒越多、肉眼难比的问题。
#
# 关键处理:
#   - accept_length 字段常为 null (vllm 不直接吐), 此时用
#     accepted_tokens / draft_tokens * num_speculative_tokens + 1 回退计算;
#   - engine 字段在早期 JSON 缺失, 退而从文件名推断;
#   - draft_model_path 里抽出 step / ckpt 名, 方便区分同一模型不同 ckpt。
#
# 用法:
#   python scripts/summarize_sd_results.py                  # 汇总全部
#   python scripts/summarize_sd_results.py --filter 35b     # 只看名字含 35b 的
#   python scripts/summarize_sd_results.py --method dflash  # 只看 dflash
#   python scripts/summarize_sd_results.py --results-dir results --csv out.csv
# ==============================================================================
import argparse
import glob
import json
import os
import re


def infer_engine(d, fname):
    """engine 字段缺失时从文件名推断 (sglang / vllm), 都没有则返回 '?'。"""
    eng = d.get("engine")
    if eng:
        return eng
    low = fname.lower()
    if "sglang" in low:
        return "sglang"
    if "vllm" in low:
        return "vllm"
    return "?"


def short_model(path):
    """从 model_path 取末段, 去掉 -padfix 后缀噪声, 作为模型短名。"""
    if not path:
        return "?"
    name = os.path.basename(path.rstrip("/"))
    return name.replace("-padfix", "")


def short_draft(path):
    """从 draft_model_path 抽出可辨识的 ckpt 标识 (优先 step, 否则末段目录名)。"""
    if not path:
        return "-"
    m = re.search(r"(epoch_\d+_step_\d+|step_\d+)", path)
    if m:
        return m.group(1)
    return os.path.basename(path.rstrip("/"))


def accept_length(summary, num_spec):
    """取 accept_length; 为 null 时用 accepted/draft*spec+1 回退。返回 float 或 None。"""
    al = summary.get("accept_length")
    if al is not None:
        return float(al)
    acc = summary.get("accepted_tokens")
    drf = summary.get("draft_tokens")
    if acc and drf and num_spec:
        return acc / drf * num_spec + 1.0
    return None


def collect(results_dir):
    """扫描目录, 返回 [row, ...]。每行是一个 (文件, 数据集) 的指标字典。"""
    rows = []
    for f in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if not isinstance(d, dict) or "summaries" not in d:
            continue  # 跳过非 benchmark JSON
        fname = os.path.basename(f)
        base = {
            "file": fname,
            "model": short_model(d.get("model_path")),
            "draft": short_draft(d.get("draft_model_path")),
            "engine": infer_engine(d, fname),
            "method": d.get("speculative_method") or "?",
            "spec": d.get("num_speculative_tokens"),
            "temp": d.get("temperature"),
            "tp": d.get("tensor_parallel_size"),
            "mtime": os.path.getmtime(f),
        }
        for s in d["summaries"]:
            row = dict(base)
            row["dataset"] = s.get("dataset", "?")
            row["nq"] = s.get("num_sd")
            row["speedup"] = s.get("speedup")
            row["accept_rate"] = s.get("accept_rate")
            row["accept_len"] = accept_length(s, base["spec"])
            rows.append(row)
    return rows


def fmt(v, spec="{:.2f}"):
    return spec.format(v) if isinstance(v, (int, float)) else "-"


def print_table(rows):
    """按 model -> draft -> engine 分组, 时间升序打印。"""
    if not rows:
        print("没有匹配的结果。")
        return
    # 排序: 模型名, draft, engine, 时间
    rows.sort(key=lambda r: (r["model"], r["draft"], r["engine"], r["mtime"], r["dataset"]))

    hdr = (f"{'model':<22} {'draft/ckpt':<20} {'eng':<6} {'method':<7} "
           f"{'spec':>4} {'T':>4} {'TP':>3} {'dataset':<14} {'nq':>4} "
           f"{'speedup':>8} {'acc_len':>8} {'acc_rate':>9}")
    print(hdr)
    print("-" * len(hdr))
    prev_key = None
    for r in rows:
        key = (r["model"], r["draft"], r["engine"], r["file"])
        # 同一文件内多个数据集时, 重复列只在首行显示, 后续留空, 便于阅读
        if key != prev_key:
            model, draft, eng, method = r["model"], r["draft"], r["engine"], r["method"]
            spec = str(r["spec"]) if r["spec"] is not None else "-"
            temp = fmt(r["temp"], "{:.1f}")
            tp = str(r["tp"]) if r["tp"] is not None else "-"
            prev_key = key
        else:
            model = draft = eng = method = spec = temp = tp = ""
        print(f"{model:<22} {draft:<20} {eng:<6} {method:<7} "
              f"{spec:>4} {temp:>4} {tp:>3} {r['dataset']:<14} "
              f"{(r['nq'] if r['nq'] is not None else '-'):>4} "
              f"{fmt(r['speedup'], '{:.2f}x'):>8} {fmt(r['accept_len']):>8} "
              f"{fmt(r['accept_rate'], '{:.3f}'):>9}")


def write_csv(rows, path):
    import csv
    cols = ["model", "draft", "engine", "method", "spec", "temp", "tp",
            "dataset", "nq", "speedup", "accept_len", "accept_rate", "file"]
    with open(path, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["model"], r["draft"], r["engine"], r["mtime"])):
            w.writerow(r)
    print(f"\nCSV 已写出: {path}  ({len(rows)} 行)")


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description="汇总投机解码评测 JSON 结果")
    ap.add_argument("--results-dir", default=os.path.join(here, "results"),
                    help="评测 JSON 所在目录 (默认 <repo>/results)")
    ap.add_argument("--filter", default=None,
                    help="只保留 model/draft/file 名含该子串的行 (大小写不敏感)")
    ap.add_argument("--method", default=None, choices=["dflash", "eagle3"],
                    help="只看指定投机方法")
    ap.add_argument("--engine", default=None, choices=["vllm", "sglang"],
                    help="只看指定推理引擎")
    ap.add_argument("--csv", default=None, help="额外写出一份 CSV")
    args = ap.parse_args()

    rows = collect(args.results_dir)
    if args.filter:
        key = args.filter.lower()
        rows = [r for r in rows
                if key in r["model"].lower() or key in r["draft"].lower()
                or key in r["file"].lower()]
    if args.method:
        rows = [r for r in rows if r["method"] == args.method]
    if args.engine:
        rows = [r for r in rows if r["engine"] == args.engine]

    print(f"\n扫描目录: {args.results_dir}")
    print(f"匹配 {len(set(r['file'] for r in rows))} 个文件, {len(rows)} 条 (文件×数据集) 记录\n")
    print_table(rows)
    if args.csv:
        write_csv(rows, args.csv)


if __name__ == "__main__":
    main()
