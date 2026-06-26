"""
Benchmark speculative decoding (DFlash) vs normal decoding on MT-Bench questions.

推理引擎是超参数: --engine vllm|sglang
  - vllm   (装在 `vllm` conda 环境): 用 --speculative-config '{"method":"dflash",...}',
           接受率从 Prometheus counter (num_accepted/num_draft) 算。
  - sglang (装在 `sglang` conda 环境, 0.5.12+ 原生支持 DFLASH): 用
           --speculative-algorithm DFLASH --speculative-draft-model-path ...
           --speculative-dflash-block-size N; 接受长度/接受率直接读 Gauge
           sglang:spec_accept_length / sglang:spec_accept_rate。

thinking (reasoning) 模式: --enable-thinking 时请求带
  chat_template_kwargs={"enable_thinking":true,"thinking":true} (Qwen3 系),
  sglang 端须配 --reasoning-parser qwen3。默认 max_tokens=4096 (对齐 DFlash README)。
  注意: Llama 等没有 <think> 模板的模型不支持 thinking, 传了也会被 chat template
  忽略 (用 --no-thinking 显式关闭更清晰)。

Usage:
    # vLLM 引擎 (文本 DFlash, 经典 Llama target)
    conda activate vllm
    python benchmark_sd.py --engine vllm \
        --model-path /prj/.../Meta-Llama-3.1-8B-Instruct \
        --draft-model-path z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat --no-thinking

    # SGLang 引擎 + thinking + max 4096 (Qwen3 系 target)
    conda activate sglang
    python benchmark_sd.py --engine sglang \
        --model-path ./model/Qwen3.5-35B-A3B \
        --draft-model-path ./model/Qwen3.5-35B-A3B-DFlash \
        --enable-thinking --max-tokens 4096
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MTBENCH_PATH = os.path.join(SCRIPT_DIR, "benchmarks", "mtbench.jsonl")
EVAL_DIR = os.path.join(SCRIPT_DIR, "datasets", "eval")

# 支持的纯文本评测集 (超参数 --datasets 从这里取)。索引由
# scripts/export_text_eval_datasets.py 预导出成 datasets/eval/<name>_text.jsonl。
SUPPORTED_DATASETS = {
    "mtbench": "MT-Bench (多轮对话, 取首轮)",
    "gsm8k": "GSM8K (小学数学应用题)",
    "humaneval": "HumanEval (Python 代码补全)",
}

# ============================================================================
# 超参数默认值 (改这里即可改默认行为; 命令行仍可覆盖)
# ============================================================================
# 默认 target 指向 Qwen3.5-35B-A3B (sglang dflash 评测主力); 跑经典 Llama 文本
# dflash 时用 --model-path 覆盖, 并配 --no-thinking --engine vllm。
DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, "model", "Qwen3.5-35B-A3B")
DEFAULT_DRAFT_MODEL_PATH = os.path.join(SCRIPT_DIR, "model", "Qwen3.5-35B-A3B-DFlash")
# 评测集 (逗号分隔, 取值见 SUPPORTED_DATASETS) 与每集取样条数
DEFAULT_DATASETS = "mtbench,gsm8k,humaneval"
# 推理引擎: "vllm" (装在 vllm 环境) 或 "sglang" (装在 sglang/dflash 环境, 0.5.12+ 支持 DFLASH)。
DEFAULT_ENGINE = "sglang"
# reasoning(thinking)解析器, 仅 sglang 用。Qwen3 系用 qwen3。
DEFAULT_REASONING_PARSER = "qwen3"
# 是否开启 thinking(reasoning)模式。
DEFAULT_ENABLE_THINKING = True
# 透传给 sglang 的额外 flag (空格分隔)。Qwen3.5-35B-A3B 这类 hybrid MoE (含 linear/mamba
# attention) 跑 DFlash 必须加; 纯文本 Llama 等非 MoE 模型设为 "" 即可。
DEFAULT_SGLANG_EXTRA_ARGS = "--attention-backend fa3 --mamba-scheduler-strategy extra_buffer"
DEFAULT_NUM_SPECULATIVE_TOKENS = 5
DEFAULT_SPECULATIVE_METHOD = "dflash"
# 默认 4096: 对齐 DFlash README 的评测设置 (thinking enabled, max output length 4096)。
DEFAULT_MAX_TOKENS = 4096
DEFAULT_NUM_QUESTIONS = 50
DEFAULT_GPU_MEMORY_UTILIZATION = 0.90
# 35B + thinking 输出可达 4096, max_model_len 给足 (prompt + 4096 输出)
DEFAULT_MAX_MODEL_LEN = 8192
DEFAULT_TENSOR_PARALLEL_SIZE = 2
DEFAULT_DATA_PARALLEL_SIZE = 1
DEFAULT_CONCURRENCY = 1
DEFAULT_PORT = 30000
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results")


def dataset_index_path(name: str) -> str:
    return os.path.join(EVAL_DIR, f"{name}_text.jsonl")


def load_text_questions(
    name: str, num_questions: Optional[int] = None
) -> List[Dict[str, Any]]:
    """从预导出的 jsonl 索引加载某个纯文本评测集, 返回 [{id, question, dataset}]。

    索引由 scripts/export_text_eval_datasets.py 导出。
    """
    index_path = dataset_index_path(name)
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"找不到 {name} 索引 {index_path}。\n"
            "请先导出 (需 datasets 库, 如 dflash 环境):\n"
            f"  python scripts/export_text_eval_datasets.py --datasets {name}"
        )
    items: List[Dict[str, Any]] = []
    with open(index_path, "r") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if num_questions is not None and len(items) >= num_questions:
                break
            row = json.loads(line)
            items.append(
                {
                    "id": row.get("id", str(idx)),
                    "question": row["question"],
                    "dataset": row.get("dataset", name),
                }
            )
    if not items:
        raise ValueError(f"{name} 索引为空: {index_path}")
    return items


def load_mtbench_questions(path: str, num_questions: Optional[int] = None) -> List[str]:
    questions = []
    with open(path, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            if "turns" in item and len(item["turns"]) > 0:
                questions.append(item["turns"][0])
    if num_questions:
        questions = questions[:num_questions]
    return questions


def wait_for_server(port: int, proc: subprocess.Popen, timeout: int = 1200) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            return False
        try:
            resp = requests.get(f"http://127.0.0.1:{port}/health", timeout=3)
            if resp.status_code == 200:
                return True
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(3)
    return False


def _build_vllm_cmd(
    model_path, port, draft_model_path, speculative_method, num_speculative_tokens,
    gpu_memory_utilization, max_model_len, tensor_parallel_size, data_parallel_size=1,
    max_num_seqs=None,
):
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port),
        "--dtype", "bfloat16",
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--trust-remote-code",
        "--max-model-len", str(max_model_len),
        "--served-model-name", "default",
    ]
    # 单卡 (TP=1) 跑 hybrid MoE+Mamba (如 Qwen3.5-35B-A3B) 时, 默认 max_num_seqs=256
    # 会超出可用的 Mamba cache block 数, CUDA graph capture 报 ValueError 起不来;
    # 评测 concurrency 很小, 调低即可。仅在显式传入时附加, 不影响多卡默认行为。
    if max_num_seqs is not None:
        cmd += ["--max-num-seqs", str(max_num_seqs)]
    if tensor_parallel_size > 1:
        cmd += ["--tensor-parallel-size", str(tensor_parallel_size)]
    if data_parallel_size > 1:
        cmd += ["--data-parallel-size", str(data_parallel_size)]
    if draft_model_path:
        spec_config: Dict[str, Any] = {
            "method": speculative_method,
            "model": draft_model_path,
            "num_speculative_tokens": num_speculative_tokens,
        }
        cmd += ["--speculative-config", json.dumps(spec_config)]
    return cmd


def _build_sglang_cmd(
    model_path, port, draft_model_path, speculative_method, num_speculative_tokens,
    gpu_memory_utilization, max_model_len, tensor_parallel_size,
    reasoning_parser, enable_thinking, sglang_extra_args=None, data_parallel_size=1,
):
    # sglang 0.5.12+ 原生支持 DFLASH (一组独立 flag, 非 vLLM 的 JSON config)。
    # --speculative-dflash-block-size 是 DFLASH 下 num_draft_tokens 的别名。
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--port", str(port),
        "--dtype", "bfloat16",
        "--mem-fraction-static", str(gpu_memory_utilization),
        "--trust-remote-code",
        "--context-length", str(max_model_len),
        "--served-model-name", "default",
        # 暴露 Prometheus /metrics (否则 spec_accept_length/rate 读不到)
        "--enable-metrics",
    ]
    if tensor_parallel_size > 1:
        cmd += ["--tp-size", str(tensor_parallel_size)]
    # 数据并行: 起 N 个 engine 副本 (每副本占 tp-size 张卡), 总占卡 = dp * tp。
    # 仅在 concurrency >= 副本数时多卡才都忙起来。
    if data_parallel_size > 1:
        cmd += ["--dp-size", str(data_parallel_size)]
    if enable_thinking and reasoning_parser:
        cmd += ["--reasoning-parser", reasoning_parser]
    if draft_model_path:
        if speculative_method.lower() == "dflash":
            cmd += [
                "--speculative-algorithm", "DFLASH",
                "--speculative-draft-model-path", draft_model_path,
                "--speculative-dflash-block-size", str(num_speculative_tokens),
            ]
        else:
            # eagle3 等: sglang 对 EAGLE 类算法有互斥校验——给了 num_draft_tokens/eagle_topk
            # 就必须同时给 num_steps, 否则 AssertionError (它本想三者全 None 时自动选)。
            # 故把三个投机参数一起给全 (与 benchmark_sd_vlm.py 的 eagle3 分支一致):
            #   num_steps=投机迭代步数, eagle_topk=每步候选分支数, num_draft_tokens=tree 总节点数。
            cmd += [
                "--speculative-algorithm", speculative_method.upper(),
                "--speculative-draft-model-path", draft_model_path,
                "--speculative-num-steps", "5",
                "--speculative-eagle-topk", "4",
                "--speculative-num-draft-tokens", "8",
            ]
    # 透传额外 sglang flag (model-specific, 如 hybrid MoE 的 fa3/mamba)
    if sglang_extra_args:
        cmd += list(sglang_extra_args)
    return cmd


def launch_server(
    model_path: str,
    port: int,
    draft_model_path: Optional[str] = None,
    speculative_method: str = "dflash",
    num_speculative_tokens: int = 5,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 4096,
    tensor_parallel_size: int = 1,
    engine: str = "vllm",
    reasoning_parser: str = "qwen3",
    enable_thinking: bool = False,
    log_path: Optional[str] = None,
    sglang_extra_args: Optional[List[str]] = None,
    data_parallel_size: int = 1,
    max_num_seqs: Optional[int] = None,
) -> subprocess.Popen:
    if engine == "sglang":
        cmd = _build_sglang_cmd(
            model_path, port, draft_model_path, speculative_method,
            num_speculative_tokens, gpu_memory_utilization, max_model_len,
            tensor_parallel_size, reasoning_parser, enable_thinking,
            sglang_extra_args, data_parallel_size,
        )
    else:
        cmd = _build_vllm_cmd(
            model_path, port, draft_model_path, speculative_method,
            num_speculative_tokens, gpu_memory_utilization, max_model_len,
            tensor_parallel_size, data_parallel_size, max_num_seqs,
        )

    label = "SD=ON" if draft_model_path else "SD=OFF"
    # sglang 在本机 (torch 2.9.1 + CuDNN 9.10) 会触发 CuDNN 版本检查直接退出, baseline
    # 和 dflash 两阶段都要绕过 (与 draft 无关)。SPEC_V2 仅 dflash 需要。不覆盖用户已设的值。
    env = os.environ.copy()
    if engine == "sglang":
        env.setdefault("SGLANG_DISABLE_CUDNN_CHECK", "1")
        if draft_model_path and speculative_method.lower() == "dflash":
            env.setdefault("SGLANG_ENABLE_SPEC_V2", "1")
    print(f"  Launching {engine} server on port {port} ({label})...", flush=True)
    print(f"    cmd: {' '.join(cmd)}", flush=True)
    # server 日志重定向到文件 (而非 PIPE), 避免父进程未及时读管道导致回压死锁,
    # 还能 `tail -f` 看启动进度。
    if log_path:
        log_fh = open(log_path, "w")
        proc = subprocess.Popen(
            cmd, stdout=log_fh, stderr=subprocess.STDOUT, text=True, env=env,
        )
        proc._log_path = log_path
        proc._log_fh = log_fh
    else:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
        )
    return proc


def kill_server(proc: subprocess.Popen):
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    time.sleep(3)


def query_server(
    port: int,
    question: str,
    max_tokens: int = 512,
    timeout: int = 600,
    temperature: float = 0.0,
    enable_thinking: bool = False,
) -> Dict:
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": question}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    # thinking (reasoning) 模式: Qwen3 系靠 chat template 的 enable_thinking 开关插入
    # <think> 段。两个键都给以兼容 vllm/sglang 不同实现。Llama 等无 <think> 模板的
    # 模型会忽略这两个键。
    if enable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": True, "thinking": True}
    if temperature > 0:
        payload["seed"] = 1234

    start = time.perf_counter()
    resp = requests.post(url, json=payload, timeout=timeout)
    elapsed = time.perf_counter() - start

    resp.raise_for_status()
    data = resp.json()

    usage = data.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)

    return {
        "elapsed": elapsed,
        "completion_tokens": completion_tokens,
        "tokens_per_sec": completion_tokens / elapsed if elapsed > 0 else 0,
    }


def run_benchmark(
    port: int,
    items: List[Dict[str, Any]],
    max_tokens: int,
    label: str,
    warmup: bool = True,
    concurrency: int = 1,
    temperature: float = 0.0,
    enable_thinking: bool = False,
) -> Tuple[List[Dict], float]:
    """跑一个数据集, 返回 (每条结果列表, 墙钟总耗时秒)。

    concurrency=1: 顺序单请求 (测单请求延迟/加速比, 对齐 DFlash README setup)。
    concurrency>1: 线程池并发 (测高并发系统吞吐)。吞吐统一用墙钟总耗时算。
    """
    results: List[Dict] = []
    print(f"\n{'='*60}")
    print(f"  Benchmarking: {label}  ({len(items)} questions, concurrency={concurrency}, T={temperature})")
    print(f"{'='*60}")

    if warmup:
        print("  Warming up (first request may take a while for kernel compilation)...", flush=True)
        warm_q = items[0]["question"]
        for i in range(3):
            try:
                query_server(port, warm_q, max_tokens=64, timeout=600,
                             temperature=temperature, enable_thinking=enable_thinking)
                print(f"    Warmup {i+1}/3 done.", flush=True)
            except Exception as e:
                print(f"    Warmup {i+1}/3 failed: {e}", flush=True)

    def _one(item: Dict[str, Any]) -> Dict:
        try:
            return query_server(port, item["question"], max_tokens,
                                 temperature=temperature, enable_thinking=enable_thinking)
        except Exception as e:
            return {"elapsed": 0, "completion_tokens": 0, "tokens_per_sec": 0, "error": str(e)}

    if concurrency <= 1:
        wall_start = time.perf_counter()
        for i, item in enumerate(items):
            short_q = item["question"][:55].replace("\n", " ")
            short_q = short_q + "..." if len(item["question"]) > 55 else short_q
            print(f"  [{i+1:3d}/{len(items)}] {short_q}", end="", flush=True)
            result = _one(item)
            if "error" in result:
                print(f"  -> ERROR: {result['error']}")
            else:
                print(f"  -> {result['completion_tokens']:3d} tok, {result['tokens_per_sec']:.1f} tok/s")
            results.append(result)
        wall_time = time.perf_counter() - wall_start
    else:
        results = [None] * len(items)
        done = 0
        wall_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = {ex.submit(_one, item): i for i, item in enumerate(items)}
            for fut in as_completed(futs):
                i = futs[fut]
                result = fut.result()
                results[i] = result
                done += 1
                if "error" in result:
                    print(f"  [{done:3d}/{len(items)}] -> ERROR: {result['error']}", flush=True)
                else:
                    print(f"  [{done:3d}/{len(items)}] -> {result['completion_tokens']:3d} tok", flush=True)
        wall_time = time.perf_counter() - wall_start

    return results, wall_time


def fetch_spec_decode_metrics(port: int, engine: str = "vllm") -> Dict[str, float]:
    """从 Prometheus /metrics 抓投机解码指标。

    vllm: counter num_accepted/num_draft tokens + draft_acceptance_rate Gauge。
    sglang: Gauge sglang:spec_accept_length (平均接受长度) / sglang:spec_accept_rate。
    """
    metrics: Dict[str, float] = {}
    try:
        resp = requests.get(f"http://127.0.0.1:{port}/metrics", timeout=10)
        if resp.status_code != 200:
            return metrics
        for line in resp.text.splitlines():
            if line.startswith("#"):
                continue
            if engine == "sglang":
                if "sglang:spec_accept_length" in line:
                    metrics["accept_length"] = float(line.split()[-1])
                elif "sglang:spec_accept_rate" in line:
                    metrics["accept_rate"] = float(line.split()[-1])
            else:
                if "spec_decode_draft_acceptance_rate" in line:
                    metrics["draft_acceptance_rate"] = float(line.split()[-1])
                elif "spec_decode_efficiency" in line:
                    metrics["system_efficiency"] = float(line.split()[-1])
                elif "spec_decode_num_accepted_tokens_total" in line or \
                     "num_accepted_tokens_total" in line:
                    metrics["accepted_tokens"] = float(line.split()[-1])
                elif "spec_decode_num_draft_tokens_total" in line or \
                     "num_draft_tokens_total" in line:
                    metrics["draft_tokens"] = float(line.split()[-1])
    except Exception:
        pass
    return metrics


def spec_metrics_delta(
    before: Dict[str, float], after: Dict[str, float], engine: str = "vllm"
) -> Dict[str, float]:
    """算出单个数据集区间内的接受率 / 接受长度。

    vllm: 用前后两次 counter 差值算接受率, 并由 num_speculative_tokens 换算接受长度。
    sglang: spec_accept_* 是 Gauge(最近值)无法差值, 直接取该集跑完后的当前值。
    """
    out: Dict[str, float] = {}
    if engine == "sglang":
        if "accept_length" in after:
            out["accept_length"] = after["accept_length"]
        if "accept_rate" in after:
            out["accept_rate"] = after["accept_rate"]
        return out
    acc = after.get("accepted_tokens", 0.0) - before.get("accepted_tokens", 0.0)
    drf = after.get("draft_tokens", 0.0) - before.get("draft_tokens", 0.0)
    if drf > 0:
        out["accepted_tokens"] = acc
        out["draft_tokens"] = drf
        out["accept_rate"] = acc / drf
    return out


def summarize_one(
    name: str,
    no_sd_results: List[Dict],
    sd_results: List[Dict],
    spec_metrics: Dict[str, float],
    no_sd_wall: float,
    sd_wall: float,
) -> Dict[str, Any]:
    """汇总单个数据集的 baseline / SD 指标 (吞吐用墙钟总耗时算)。"""
    no_sd_tokens = sum(r["completion_tokens"] for r in no_sd_results)
    no_sd_tps = no_sd_tokens / no_sd_wall if no_sd_wall > 0 else 0
    sd_tokens = sum(r["completion_tokens"] for r in sd_results)
    sd_tps = sd_tokens / sd_wall if sd_wall > 0 else 0
    speedup = sd_tps / no_sd_tps if no_sd_tps > 0 else 0
    return {
        "dataset": name,
        "num_sd": len(sd_results),
        "baseline_tps": no_sd_tps,
        "sd_tps": sd_tps,
        "speedup": speedup,
        "accept_rate": spec_metrics.get("accept_rate"),
        "accept_length": spec_metrics.get("accept_length"),
        "accepted_tokens": spec_metrics.get("accepted_tokens"),
        "draft_tokens": spec_metrics.get("draft_tokens"),
        "baseline_tokens": no_sd_tokens,
        "baseline_time": no_sd_wall,
        "sd_tokens": sd_tokens,
        "sd_time": sd_wall,
    }


def print_final_table(summaries: List[Dict[str, Any]], engine: str = "vllm",
                      num_speculative_tokens: int = 5):
    """打印每个数据集 + 总计的加速比 / 接受率 / 接受长度表格。"""

    def _accept_length(s: Dict[str, Any]) -> Optional[float]:
        if s.get("accept_length") is not None:
            return s["accept_length"]
        if s.get("accept_rate") is not None:
            return s["accept_rate"] * num_speculative_tokens + 1.0
        return None

    print("\n")
    print("=" * 90)
    print(f"        BENCHMARK RESULTS (DFlash, {engine})  —  加速比 / 接受率 / 接受长度")
    print("=" * 90)
    print(f"  {'Dataset':<12} {'N':>4} {'Baseline':>11} {'DFlash':>11} "
          f"{'Speedup':>9} {'AcceptRate':>11} {'AcceptLen':>10}")
    print(f"  {'':<12} {'':>4} {'tok/s':>11} {'tok/s':>11} {'':>9} "
          f"{'(acc/draft)':>11} {'(tok/步)':>10}")
    print("-" * 90)

    tot_b_tok = tot_b_time = tot_s_tok = tot_s_time = 0.0
    tot_acc = tot_drf = 0.0
    al_vals: List[float] = []
    for s in summaries:
        ar = s["accept_rate"]
        ar_str = f"{ar:.4f}" if ar is not None else "N/A"
        al = _accept_length(s)
        al_str = f"{al:.2f}" if al is not None else "N/A"
        if al is not None:
            al_vals.append(al)
        print(
            f"  {s['dataset']:<12} {s['num_sd']:>4d} "
            f"{s['baseline_tps']:>11.1f} {s['sd_tps']:>11.1f} "
            f"{s['speedup']:>8.2f}x {ar_str:>11} {al_str:>10}"
        )
        tot_b_tok += s["baseline_tokens"]; tot_b_time += s["baseline_time"]
        tot_s_tok += s["sd_tokens"]; tot_s_time += s["sd_time"]
        if s["accepted_tokens"] is not None and s["draft_tokens"] is not None:
            tot_acc += s["accepted_tokens"]; tot_drf += s["draft_tokens"]

    print("-" * 90)
    overall_b_tps = tot_b_tok / tot_b_time if tot_b_time > 0 else 0
    overall_s_tps = tot_s_tok / tot_s_time if tot_s_time > 0 else 0
    overall_speedup = overall_s_tps / overall_b_tps if overall_b_tps > 0 else 0
    if tot_drf > 0:
        overall_ar = f"{tot_acc / tot_drf:.4f}"
        overall_al = f"{tot_acc / tot_drf * num_speculative_tokens + 1.0:.2f}"
    elif al_vals:
        overall_ar = "N/A"
        overall_al = f"{sum(al_vals) / len(al_vals):.2f}"
    else:
        overall_ar = overall_al = "N/A"
    print(
        f"  {'OVERALL':<12} {'':>4} "
        f"{overall_b_tps:>11.1f} {overall_s_tps:>11.1f} "
        f"{overall_speedup:>8.2f}x {overall_ar:>11} {overall_al:>10}"
    )
    print("=" * 90)
    print()


def _dump_server_tail(proc: subprocess.Popen, n: int = 6000):
    log_path = getattr(proc, "_log_path", None)
    if log_path and os.path.exists(log_path):
        try:
            with open(log_path, "r") as f:
                output = f.read()
            if output and len(output) > n:
                print("  ... (truncated) ...")
                print(output[-n:])
            elif output:
                print(output)
        except Exception:
            pass
        return
    if proc.stdout:
        os.set_blocking(proc.stdout.fileno(), False)
        try:
            output = proc.stdout.read()
            if output and len(output) > n:
                print("  ... (truncated) ...")
                print(output[-n:])
            elif output:
                print(output)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Benchmark DFlash SD vs no-SD on MT-Bench (vllm/sglang)")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--draft-model-path", type=str, default=DEFAULT_DRAFT_MODEL_PATH)
    parser.add_argument(
        "--engine", type=str, default=DEFAULT_ENGINE, choices=["vllm", "sglang"],
        help="推理引擎: vllm (vllm 环境) 或 sglang (sglang 环境, 0.5.12+ 支持 DFLASH)",
    )
    parser.add_argument(
        "--reasoning-parser", type=str, default=DEFAULT_REASONING_PARSER,
        help="reasoning(thinking)解析器, 仅 sglang 用; Qwen3 系用 qwen3",
    )
    parser.add_argument(
        "--enable-thinking", dest="enable_thinking", action="store_true",
        default=DEFAULT_ENABLE_THINKING,
        help="开启 thinking(reasoning)模式 (Qwen3 系有效; Llama 无 <think> 模板会忽略)",
    )
    parser.add_argument(
        "--no-thinking", dest="enable_thinking", action="store_false",
        help="关闭 thinking 模式 (Llama 等非 reasoning 模型用这个)",
    )
    parser.add_argument(
        "--sglang-extra-args", type=str, default=DEFAULT_SGLANG_EXTRA_ARGS,
        help="透传给 sglang server 的额外 flag (空格分隔), 仅 --engine sglang 用。"
             "Qwen3.5-35B-A3B 等 hybrid MoE 须设 "
             "'--attention-backend fa3 --mamba-scheduler-strategy extra_buffer'",
    )
    parser.add_argument("--num-speculative-tokens", type=int, default=DEFAULT_NUM_SPECULATIVE_TOKENS)
    parser.add_argument("--speculative-method", type=str, default=DEFAULT_SPECULATIVE_METHOD,
                        help="投机解码方法 (dflash, eagle3, ...)")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="采样温度。0.0=贪心 (按 argmax 严格匹配接受); >0=采样。"
             "thinking 模式 Qwen 推荐采样 (如 0.6)",
    )
    parser.add_argument(
        "--datasets", type=str, default=DEFAULT_DATASETS,
        help=f"逗号分隔的评测集 (超参数)。支持: {','.join(SUPPORTED_DATASETS)}",
    )
    parser.add_argument("--num-questions", type=int, default=DEFAULT_NUM_QUESTIONS,
                        help="每个数据集取多少条")
    parser.add_argument("--gpu-memory-utilization", type=float, default=DEFAULT_GPU_MEMORY_UTILIZATION)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=DEFAULT_TENSOR_PARALLEL_SIZE,
        help="张量并行卡数; 35B 等大模型单卡放不下时设 >1 (该模型 KV head 限制 TP<=2)",
    )
    parser.add_argument(
        "--data-parallel-size", type=int, default=DEFAULT_DATA_PARALLEL_SIZE,
        help="数据并行副本数 (每副本占 tp-size 张卡); 总占卡 = dp * tp。"
             "仅在 --concurrency >= 副本数时多卡才都忙起来",
    )
    parser.add_argument(
        "--max-num-seqs", type=int, default=None,
        help="vLLM 单批最大并发序列数 (默认不传, 用 vLLM 默认 256)。单卡跑 hybrid "
             "MoE+Mamba (如 35B-A3B TP=1) 时须调低 (如 64), 否则 Mamba cache block "
             "不足导致 CUDA graph capture 失败起不来。",
    )
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY,
        help="同时在飞的请求数; 1=顺序单请求 (测单请求延迟/加速比, 对齐 README), "
             ">1=并发打流 (测高并发吞吐, 让 data-parallel 多副本同时干活)",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--skip-baseline", action="store_true",
        help="跳过 baseline, 只跑 SD",
    )
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    # 把 --sglang-extra-args 字符串拆成 list (空串 -> 空 list)
    sglang_extra = args.sglang_extra_args.split() if args.sglang_extra_args else []

    # 解析数据集超参, 预加载每个集的题目
    dataset_names = [d.strip() for d in args.datasets.split(",") if d.strip()]
    for name in dataset_names:
        if name not in SUPPORTED_DATASETS:
            print(f"ERROR: 不支持的数据集 '{name}'。支持: {list(SUPPORTED_DATASETS)}",
                  file=sys.stderr)
            sys.exit(1)
    dataset_items = {
        name: load_text_questions(name, args.num_questions) for name in dataset_names
    }

    # 引擎按 --engine 检测对应库 (vllm 装在 vllm 环境, sglang 装在 sglang/dflash 环境)
    if args.engine == "sglang":
        try:
            import sglang
            engine_version = sglang.__version__
        except Exception:
            print("ERROR: 未找到 sglang。--engine sglang 须在 `sglang`/`dflash` conda 环境 (0.5.12+)。",
                  file=sys.stderr)
            sys.exit(1)
    else:
        try:
            import vllm
            engine_version = vllm.__version__
        except Exception:
            print("ERROR: 未找到 vllm。--engine vllm 须在 `vllm` conda 环境。", file=sys.stderr)
            sys.exit(1)

    print("=" * 78)
    print("     Multi-Dataset DFlash Speculative Decoding Benchmark (text)")
    print("=" * 78)
    print(f"  Target model: {args.model_path}")
    print(f"  Draft model:  {args.draft_model_path}")
    print(f"  Datasets:     " + ", ".join(
        f"{n}({len(dataset_items[n])})" for n in dataset_names))
    print(f"  Per-dataset:  {args.num_questions} questions")
    print(f"  Max tokens:   {args.max_tokens}")
    print(f"  Temperature:  {args.temperature}  ({'greedy/argmax' if args.temperature == 0 else 'sampling'})")
    print(f"  Thinking:     {'ON' if args.enable_thinking else 'OFF'}"
          f"{' (reasoning-parser=' + args.reasoning_parser + ')' if args.engine == 'sglang' and args.enable_thinking else ''}")
    print(f"  Spec tokens:  {args.num_speculative_tokens}")
    print(f"  Method:       {args.speculative_method}")
    print(f"  Parallel:     TP={args.tensor_parallel_size}  concurrency={args.concurrency}")
    print(f"  Engine:       {args.engine} {engine_version}")
    print("=" * 78)
    print()

    baseline_by_ds: Dict[str, List[Dict]] = {n: [] for n in dataset_names}
    sd_by_ds: Dict[str, List[Dict]] = {n: [] for n in dataset_names}
    spec_by_ds: Dict[str, Dict[str, float]] = {n: {} for n in dataset_names}
    baseline_wall: Dict[str, float] = {n: 0.0 for n in dataset_names}
    sd_wall: Dict[str, float] = {n: 0.0 for n in dataset_names}

    os.makedirs(args.output_dir, exist_ok=True)
    launch_ts = time.strftime("%Y%m%d_%H%M%S")
    baseline_log = os.path.join(args.output_dir, f"sdtext_{args.engine}_baseline_{launch_ts}.log")
    sd_log = os.path.join(args.output_dir, f"sdtext_{args.engine}_sd_{launch_ts}.log")

    # ====== Phase 1: Baseline (no speculative decoding) — 单 server 跑完所有集 ======
    if not args.skip_baseline:
        print("Phase 1/2: Running baseline (no speculative decoding)")
        baseline_proc = launch_server(
            args.model_path, args.port,
            draft_model_path=None,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            data_parallel_size=args.data_parallel_size,
            engine=args.engine,
            reasoning_parser=args.reasoning_parser,
            enable_thinking=args.enable_thinking,
            log_path=baseline_log,
            sglang_extra_args=sglang_extra,
            max_num_seqs=args.max_num_seqs,
        )
        print(f"  Waiting for server to be ready... [server log: {baseline_log}]", flush=True)
        if not wait_for_server(args.port, baseline_proc):
            print("  ERROR: Baseline server failed to start!")
            _dump_server_tail(baseline_proc)
            kill_server(baseline_proc)
            return
        print("  Server ready!")
        for di, name in enumerate(dataset_names):
            baseline_by_ds[name], baseline_wall[name] = run_benchmark(
                args.port, dataset_items[name], args.max_tokens,
                f"[baseline] {name}", warmup=(di == 0),
                concurrency=args.concurrency,
                temperature=args.temperature,
                enable_thinking=args.enable_thinking,
            )
        print("\n  Shutting down baseline server...", flush=True)
        kill_server(baseline_proc)
        print("  Baseline server stopped. Waiting for GPU memory release...")
        time.sleep(5)
    else:
        print("Phase 1/2: SKIPPED baseline (--skip-baseline)")

    # ====== Phase 2: With DFlash — 单 server 跑完所有集, 每集跑完读 Gauge ======
    print("\nPhase 2/2: Running with DFlash speculative decoding")
    sd_proc = launch_server(
        args.model_path, args.port,
        draft_model_path=args.draft_model_path,
        speculative_method=args.speculative_method,
        num_speculative_tokens=args.num_speculative_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        data_parallel_size=args.data_parallel_size,
        engine=args.engine,
        reasoning_parser=args.reasoning_parser,
        enable_thinking=args.enable_thinking,
        log_path=sd_log,
        sglang_extra_args=sglang_extra,
        max_num_seqs=args.max_num_seqs,
    )
    print(f"  Waiting for server to be ready... [server log: {sd_log}]", flush=True)
    if not wait_for_server(args.port, sd_proc):
        print("  ERROR: SD server failed to start!")
        _dump_server_tail(sd_proc)
        kill_server(sd_proc)
        return
    print("  Server ready!")

    for di, name in enumerate(dataset_names):
        # 每集开跑前抓一次 (vllm 用差值; sglang 是 Gauge, 用 after 即可)
        before = fetch_spec_decode_metrics(args.port, engine=args.engine)
        sd_by_ds[name], sd_wall[name] = run_benchmark(
            args.port, dataset_items[name], args.max_tokens,
            f"[{args.speculative_method}] {name}", warmup=(di == 0),
            concurrency=args.concurrency,
            temperature=args.temperature,
            enable_thinking=args.enable_thinking,
        )
        after = fetch_spec_decode_metrics(args.port, engine=args.engine)
        spec_by_ds[name] = spec_metrics_delta(before, after, engine=args.engine)

    print("\n  Shutting down SD server...", flush=True)
    kill_server(sd_proc)

    # ====== Summary: 每个集 + 总计 ======
    summaries = [
        summarize_one(
            name, baseline_by_ds[name], sd_by_ds[name], spec_by_ds[name],
            baseline_wall[name], sd_wall[name],
        )
        for name in dataset_names
    ]
    print_final_table(summaries, engine=args.engine,
                      num_speculative_tokens=args.num_speculative_tokens)

    # 落盘结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_tag = os.path.basename(args.model_path.rstrip("/")) or "model"
    tag = "-".join(dataset_names)
    out_file = os.path.join(
        args.output_dir,
        f"{model_tag}-{args.engine}-{args.speculative_method}_{tag}_{timestamp}.json",
    )
    with open(out_file, "w") as f:
        json.dump(
            {
                "engine": args.engine,
                "engine_version": engine_version,
                "model_path": args.model_path,
                "draft_model_path": args.draft_model_path,
                "speculative_method": args.speculative_method,
                "enable_thinking": args.enable_thinking,
                "reasoning_parser": args.reasoning_parser if args.engine == "sglang" else None,
                "datasets": dataset_names,
                "num_questions": args.num_questions,
                "num_speculative_tokens": args.num_speculative_tokens,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "tensor_parallel_size": args.tensor_parallel_size,
                "data_parallel_size": args.data_parallel_size,
                "concurrency": args.concurrency,
                "summaries": summaries,
                "per_question": {
                    name: {"baseline": baseline_by_ds[name], "sd": sd_by_ds[name]}
                    for name in dataset_names
                },
            },
            f,
            indent=2,
        )
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
