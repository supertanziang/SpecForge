"""
Benchmark speculative decoding (DFlash) vs normal decoding for a *vision-language*
target model (Qwen2.5-VL-7B) on multiple image+text benchmarks (MM-Vet / ScienceQA / MME).

与纯文本版 benchmark_sd.py 的区别:
  - 读取一个或多个图文评测集 (datasets/eval/<name>_index.jsonl, 预导出的图片 + jsonl);
  - 每条样本把图片编码成 base64 data URL, 按 OpenAI Vision chat 格式 (image_url + text) 发送;
  - 启动 vLLM server 时打开多模态 (--limit-mm-per-prompt image=1);
  - DFlash draft 端不读图像, 只消费 target 的 hidden states (1D 位置), 因此可以直接挂上 Qwen2.5-VL target。

双阶段对照 (每阶段只启动 1 次 server, 顺序跑完所有数据集):
  Phase 1: baseline (无 speculative decoding)
  Phase 2: with DFlash speculative decoding
每个数据集单独汇总 baseline vs SD 的吞吐 (tok/s)、加速比、draft 接受率/接受长度。

推理引擎是超参数: --engine vllm|sglang
  - vllm   (装在 `vllm` conda 环境): 用 --speculative-config '{"method":"dflash",...}',
           接受率从 Prometheus counter (num_accepted/num_draft) 差值算。
  - sglang (装在 `sglang` conda 环境, 0.5.12+ 原生支持 DFLASH): 用
           --speculative-algorithm DFLASH --speculative-draft-model-path ...
           --speculative-dflash-block-size N; 接受长度/接受率直接读 Gauge
           sglang:spec_accept_length / sglang:spec_accept_rate。
两个引擎的 chat 接口都是 OpenAI 兼容 /v1/chat/completions。

thinking (reasoning) 模式: --enable-thinking 时, 请求里带
  chat_template_kwargs={"enable_thinking":true,"thinking":true} (Qwen3 系),
  sglang 端须配 --reasoning-parser qwen3。默认 max_tokens=4096 (thinking 输出更长)。

数据集是超参数: --datasets mmvet,sqa,mme  (每个集取 --num-questions 条)。
支持的集见 SUPPORTED_DATASETS; 索引由 specforge 环境的 scripts/export_eval_datasets.py 预导出
(因为评测引擎只装在各自环境, 而 datasets 库在那些环境不可用)。

重要: vllm 只装在 `vllm` 环境, sglang 只装在 `sglang` 环境 (specforge 环境两者都没有)。
按 --engine 切到对应环境再跑, 或用随附的 examples/run_*.sh。手动示例:

    # vLLM 引擎
    conda activate vllm
    python benchmark_sd_vlm.py --engine vllm \
        --model-path ./model/Qwen2.5-VL-7B-Instruct \
        --draft-model-path ./outputs/qwen2.5-vl-7b-dflash/epoch_6_step_50940 \
        --datasets mmvet,sqa,mme --num-questions 50

    # SGLang 引擎 + thinking + max 4096 (对齐 DFlash README setup)
    conda activate sglang
    python benchmark_sd_vlm.py --engine sglang \
        --model-path ./model/Qwen3.5-35B-A3B \
        --draft-model-path ./model/Qwen3.5-35B-A3B-DFlash \
        --enable-thinking --max-tokens 4096 \
        --datasets mmvet,sqa,mme --num-questions 50
"""

import argparse
import base64
import io
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(SCRIPT_DIR, "datasets", "eval")

# 支持的图文评测集 (超参数 --datasets 从这里取)。索引由 specforge 环境的
# scripts/export_eval_datasets.py 预导出成 datasets/eval/<name>_index.jsonl。
SUPPORTED_DATASETS = {
    "mmvet": "MM-Vet (6 类多模态综合能力)",
    "sqa": "ScienceQA (带图科学多选题)",
    "mme": "MME (Yes/No 感知/认知)",
    "textvqa": "TextVQA (场景文字 OCR 问答, prompt 诱导中长输出)",
    "videomme": "Video-MME (视频理解多选问答)",
    "longvideobench": "LongVideoBench (长视频推理问答)",
}

VIDEO_DATASETS = {"videomme", "longvideobench"}

# ============================================================================
# 超参数配置区: 所有命令行参数的默认值都集中在这里, 改这里即可改默认行为;
# 命令行 (如 --num-questions 100) 仍可临时覆盖任意一项。
# ============================================================================
# 模型路径
DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, "model", "Qwen2.5-VL-7B-Instruct")
DEFAULT_DRAFT_MODEL_PATH = os.path.join(
    SCRIPT_DIR, "outputs", "qwen2.5-vl-7b-dflash", "epoch_6_step_50940"
)
# 评测集 (逗号分隔, 取值见 SUPPORTED_DATASETS) 与每集取样条数
DEFAULT_DATASETS = "mmvet,sqa,mme"
DEFAULT_NUM_QUESTIONS = 50
# 投机解码 (speculative decoding) 方法。本脚本把它透传给 vLLM 的
# --speculative-config 的 "method" 字段, 不同方法对 draft 模型的要求不同:
#
#   本项目训练/验证过、可直接用的两种 (draft 权重见 outputs/ 或 scripts/convert_vispec_to_*.py):
#     "dflash"  — DFlash: draft 端只消费 target 的 hidden states (1D 位置),
#                 不读图像, 因此能直接挂在 Qwen2.5-VL 这类 VLM target 上 (本脚本默认)。
#     "eagle3"  — EAGLE-3: 复用 target 多层特征训练轻量 draft head, 接受率通常较高。
#
#   vLLM 引擎还支持下列方法 (本脚本可透传, 但需自备对应 draft 权重, 且未必适配 VLM target):
#     "eagle"         — EAGLE v1 (eagle3 的前代)。
#     "ngram"         — n-gram prompt lookup, 无需 draft 模型, 靠 prompt 内重复片段命中。
#     "medusa"        — Medusa 多头并行预测。
#     "mlp_speculator"— MLP speculator (IBM)。
#     "draft_model"   — 经典小模型草稿 (用一个独立小 LM 当 draft)。
# 取值最终以所用 vLLM 版本支持的为准; 选了引擎不认的 method 会在 server 启动时报错。
DEFAULT_NUM_SPECULATIVE_TOKENS = 5
DEFAULT_SPECULATIVE_METHOD = "dflash"
# 推理引擎: "vllm" (装在 vllm 环境) 或 "sglang" (装在 sglang 环境, 0.5.12+ 原生支持 DFLASH)。
DEFAULT_ENGINE = "sglang"
# reasoning (thinking) 解析器, 仅 sglang 用 (--reasoning-parser)。Qwen3 系用 qwen3。
DEFAULT_REASONING_PARSER = "qwen3"
# 是否开启 thinking (reasoning) 模式: 请求里带 chat_template_kwargs={"enable_thinking":True}。
DEFAULT_ENABLE_THINKING = True
# sglang 跑 Qwen3.5-35B-A3B 这类 hybrid MoE (含 linear/mamba attention) DFlash 时必须加的
# 额外 flag (空格分隔)。否则 server 会因 radix cache 与 mamba 不兼容、或 attention backend
# 不对而启动失败。纯文本 Llama 等模型可设为 "" (空)。配套 env 见 launch_server。
DEFAULT_SGLANG_EXTRA_ARGS = "--attention-backend fa3 --mamba-scheduler-strategy extra_buffer"
# 生成与显存
# 默认 4096: 对齐 DFlash README 的评测设置 (thinking enabled, max output length 4096)。
DEFAULT_MAX_TOKENS = 4096
# 采样温度。0.0=贪心 (greedy, 投机解码按 argmax 严格匹配才接受); >0=采样解码
# (按 speculative sampling 概率接受, 接受率对应训练时优化的 expected acceptance)。
# 注意: eagle3/dflash 训练优化的是采样接受率, 贪心严格匹配率可能远低于它,
# 故贪心 (T=0) 评测出的接受率可能偏低, 想对齐训练指标用 T=1.0。
DEFAULT_TEMPERATURE = 0.0
DEFAULT_GPU_MEMORY_UTILIZATION = 0.90
DEFAULT_MAX_MODEL_LEN = 8192
# 显卡: GPU_IDS 指定用哪几张卡 (如 "2" 或 "2,3", None=继承环境/全部, 通过
# CUDA_VISIBLE_DEVICES 注入子进程); TP/DP 指定并行方式, 总占卡数 = TP * DP。
DEFAULT_GPU_IDS = None
DEFAULT_TENSOR_PARALLEL_SIZE = 1
DEFAULT_DATA_PARALLEL_SIZE = 1
# 并发: 1=顺序单请求 (测延迟/加速比), >1=线程池高并发 (测系统吞吐)
DEFAULT_CONCURRENCY = 1
# 服务端口与结果目录
DEFAULT_PORT = 30000
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results")


def dataset_index_path(name: str) -> str:
    return os.path.join(EVAL_DIR, f"{name}_index.jsonl")


def load_questions(
    name: str, num_questions: Optional[int] = None
) -> List[Dict[str, Any]]:
    """从预导出的 jsonl 索引加载某个评测集。

    图文集返回 [{id, question, image_path, dataset}]。
    视频集返回 [{id, question, video_path, num_frames, duration, dataset}]。
    """
    index_path = dataset_index_path(name)
    if not os.path.exists(index_path):
        export_script = (
            "scripts/export_video_eval_datasets.py" if name in VIDEO_DATASETS
            else "scripts/export_eval_datasets.py"
        )
        raise FileNotFoundError(
            f"找不到 {name} 索引 {index_path}。\n"
            "请先在 specforge 环境导出 (datasets 库在 vllm 环境不可用):\n"
            f"  conda activate specforge\n"
            f"  python {export_script} --datasets {name}"
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
            item: Dict[str, Any] = {
                "id": row.get("id", str(idx)),
                "question": row["question"],
                "dataset": row.get("dataset", name),
            }
            if "video_path" in row:
                item["video_path"] = row["video_path"]
                item["num_frames"] = row.get("num_frames", 8)
                item["duration"] = row.get("duration", 0)
            else:
                item["image_path"] = row["image_path"]
            items.append(item)
    if not items:
        raise ValueError(f"{name} 索引为空: {index_path}")
    return items


def encode_image_to_data_url(image_path: str) -> str:
    """把图片文件编码成 base64 data URL (JPEG)。"""
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def extract_video_frames(video_path: str, num_frames: int) -> List[str]:
    """从视频均匀抽帧, 返回 base64 data URL 列表。用 opencv-python-headless。"""
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"无法读取视频帧: {video_path}")

    indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    data_urls = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        data_urls.append(f"data:image/jpeg;base64,{b64}")
    cap.release()
    return data_urls


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
    gpu_memory_utilization, max_model_len, tensor_parallel_size, data_parallel_size,
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
        # 多模态: 每条 prompt 最多 N 张图 (视频帧作为多图送入, 需要放大限制)
        "--limit-mm-per-prompt", json.dumps({"image": 32}),
    ]
    # 单卡 (TP=1) 跑 hybrid MoE+Mamba (如 Qwen3.5-35B-A3B) 时, 默认 max_num_seqs=256
    # 会超出可用 Mamba cache block 数, CUDA graph capture 报 ValueError 起不来;
    # 评测 concurrency 很小, 调低即可。仅在显式传入时附加, 不影响多卡默认行为。
    if max_num_seqs is not None:
        cmd += ["--max-num-seqs", str(max_num_seqs)]
    # 大模型 (如 Qwen3.5-35B-A3B, 72GB) 单卡放不下时, 用张量并行跨多卡切分。
    if tensor_parallel_size > 1:
        cmd += ["--tensor-parallel-size", str(tensor_parallel_size)]
    # 数据并行: 起 N 个 engine 副本 (每副本占 tensor_parallel_size 张卡), vLLM 自带
    # 负载均衡把并发请求分摊到各副本。总占卡数 = data_parallel_size * tensor_parallel_size。
    # 注意: DP 只有在「并发请求 >= 副本数」时才会让多卡都忙起来 (见 --concurrency)。
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
    gpu_memory_utilization, max_model_len, tensor_parallel_size, data_parallel_size,
    reasoning_parser, enable_thinking, sglang_extra_args=None,
):
    # sglang 0.5.12+ 原生支持 DFLASH。它没有 vLLM 的 --speculative-config JSON,
    # 而是一组独立 flag; --speculative-dflash-block-size 是 DFLASH 下 num_draft_tokens 的别名。
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--port", str(port),
        "--dtype", "bfloat16",
        # sglang 用 mem-fraction-static 而非 gpu-memory-utilization (语义相近: 静态显存占比)
        "--mem-fraction-static", str(gpu_memory_utilization),
        "--trust-remote-code",
        "--context-length", str(max_model_len),
        "--served-model-name", "default",
        # 暴露 Prometheus /metrics (否则 spec_accept_length/rate 读不到, 接受率/接受长度为 N/A)
        "--enable-metrics",
    ]
    # 张量并行 / 数据并行 (sglang 用 --tp-size / --dp-size)
    if tensor_parallel_size > 1:
        cmd += ["--tp-size", str(tensor_parallel_size)]
    if data_parallel_size > 1:
        cmd += ["--dp-size", str(data_parallel_size)]
    # thinking (reasoning) 模式须在 server 端挂 reasoning 解析器, 否则 <think> 不会被解析
    if enable_thinking and reasoning_parser:
        cmd += ["--reasoning-parser", reasoning_parser]
    if draft_model_path:
        if speculative_method.lower() == "dflash":
            cmd += [
                "--speculative-algorithm", "DFLASH",
                "--speculative-draft-model-path", draft_model_path,
                # block size = verify window length = num_speculative_tokens 的别名
                "--speculative-dflash-block-size", str(num_speculative_tokens),
            ]
        else:
            # eagle3 等: sglang 用大写算法名。注意 sglang 对 EAGLE 类算法有互斥校验
            # (server_args.py): 若给了 speculative_num_draft_tokens / eagle_topk, 就必须
            # 同时给 speculative_num_steps, 否则 AssertionError (它本想在三者全 None 时自动
            # 选参数)。故这里把 eagle3 的三个投机参数一起给全:
            #   num_steps   = 投机迭代步数 (draft 自回归生成 draft tree 的深度)
            #   eagle_topk  = 每步保留的候选分支数 (tree 宽度); >1 时 page_size>1 须配
            #                 flashinfer/fa3 backend (sglang 默认 flashinfer + page_size=1, 安全)
            #   num_draft_tokens = 一次 verify 的 draft token 总数 (tree 总节点数, 须 >= topk)
            # 用较小的 (5, 4, 8) 树参数: topk=8/num_draft_tokens=16 在并发 56 + 多模态大图
            # + mem-fraction-static 偏高时会把显存撑爆 (实测 CUDA OOM, scheduler 崩)。
            # eagle3 与 dflash 算法不同, spec token 数无法严格对齐, 故不复用 num_speculative_tokens。
            cmd += [
                "--speculative-algorithm", speculative_method.upper(),
                "--speculative-draft-model-path", draft_model_path,
                "--speculative-num-steps", "5",
                "--speculative-eagle-topk", "4",
                "--speculative-num-draft-tokens", "8",
            ]
    # 透传额外 sglang flag (model-specific)。例如 Qwen3.5-35B-A3B 这类 hybrid MoE
    # (含 linear/mamba attention) 跑 DFLASH 必须加:
    #   --attention-backend fa3 --mamba-scheduler-strategy extra_buffer
    # (否则 server 启动会因 radix cache 与 mamba 不兼容而报错)。
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
    max_model_len: int = 8192,
    tensor_parallel_size: int = 1,
    data_parallel_size: int = 1,
    gpu_ids: Optional[str] = None,
    log_path: Optional[str] = None,
    engine: str = "vllm",
    reasoning_parser: str = "qwen3",
    enable_thinking: bool = False,
    sglang_extra_args: Optional[List[str]] = None,
    max_num_seqs: Optional[int] = None,
) -> subprocess.Popen:
    if engine == "sglang":
        cmd = _build_sglang_cmd(
            model_path, port, draft_model_path, speculative_method,
            num_speculative_tokens, gpu_memory_utilization, max_model_len,
            tensor_parallel_size, data_parallel_size, reasoning_parser, enable_thinking,
            sglang_extra_args,
        )
    else:
        cmd = _build_vllm_cmd(
            model_path, port, draft_model_path, speculative_method,
            num_speculative_tokens, gpu_memory_utilization, max_model_len,
            tensor_parallel_size, data_parallel_size, max_num_seqs,
        )

    label = "SD=ON" if draft_model_path else "SD=OFF"
    # 指定用哪几张卡: 通过 CUDA_VISIBLE_DEVICES 注入子进程。None 时继承父进程环境。
    env = os.environ.copy()
    if gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    # sglang 在本机 (torch 2.9.1 + CuDNN 9.10) 会触发 CuDNN 版本检查直接退出, baseline
    # 和 dflash 两阶段都要绕过 (与 draft 无关)。SPEC_V2 仅 dflash 需要。不覆盖用户已设的值。
    if engine == "sglang":
        env.setdefault("SGLANG_DISABLE_CUDNN_CHECK", "1")
        if draft_model_path and speculative_method.lower() == "dflash":
            env.setdefault("SGLANG_ENABLE_SPEC_V2", "1")
    gpu_label = gpu_ids if gpu_ids else env.get("CUDA_VISIBLE_DEVICES", "all")
    print(f"  Launching {engine} server on port {port} ({label}, GPU={gpu_label})...", flush=True)
    print(f"    cmd: {' '.join(cmd)}", flush=True)
    # server 的 stdout/stderr 重定向到日志文件, 而不是 subprocess.PIPE。
    # 原因 (踩过的坑): 用 PIPE 时父进程在 server ready 之前不读管道, DP=N 会起 N
    # 个 EngineCore, 启动日志量是单卡的 N 倍, 很快写满 64KB 管道缓冲 → 所有
    # EngineCore 卡在 pipe_write → server 永远起不来 (经典 PIPE 死锁)。写文件则
    # 无回压, 任意并行度都安全, 还能 `tail -f` 实时看 server 启动进度。
    if log_path:
        log_fh = open(log_path, "w")
        proc = subprocess.Popen(
            cmd, stdout=log_fh, stderr=subprocess.STDOUT, text=True, env=env,
        )
        proc._log_path = log_path  # 供 _dump_server_tail 读取
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
    item: Dict[str, Any],
    max_tokens: int = 512,
    timeout: int = 600,
    temperature: float = 0.0,
    enable_thinking: bool = False,
) -> Dict:
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    if "video_path" in item:
        frame_urls = extract_video_frames(item["video_path"], item.get("num_frames", 8))
        content = [
            {"type": "image_url", "image_url": {"url": u}} for u in frame_urls
        ] + [{"type": "text", "text": item["question"]}]
    else:
        data_url = encode_image_to_data_url(item["image_path"])
        content = [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": item["question"]},
        ]
    payload = {
        "model": "default",
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    # thinking (reasoning) 模式: Qwen3 系靠 chat template 的 enable_thinking 开关插入
    # <think> 段。vllm/sglang 都支持把 chat_template_kwargs 透传给 tokenizer 的
    # apply_chat_template; 两个键都给以兼容不同实现 (sglang 这版会把 enable_thinking
    # 归一成 thinking)。
    if enable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": True, "thinking": True}
    # 采样解码 (temperature>0) 时固定 seed, 保证 baseline 与 SD 两阶段、以及多次
    # 运行之间可复现 (否则随机性会污染加速比/接受率对比)。
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

    concurrency=1: 顺序单请求 (测单请求延迟/加速比)。
    concurrency>1: 用线程池同时在飞 N 条请求 (测高并发系统吞吐); 只有这样
                   data-parallel 的多个 engine 副本才会同时干活。

    吞吐统一用墙钟总耗时 (wall_time) 计算, 而非各请求 elapsed 之和——并发下
    请求时间互相重叠, 求和会重复计时, 只有墙钟才反映系统真实吞吐。
    """
    results: List[Dict] = []
    print(f"\n{'='*60}")
    print(f"  Benchmarking: {label}  ({len(items)} questions, concurrency={concurrency}, T={temperature})")
    print(f"{'='*60}")

    if warmup:
        print("  Warming up (first request may take a while for kernel compilation)...", flush=True)
        warm = items[0]
        for i in range(3):
            try:
                query_server(port, warm, max_tokens=64, timeout=600,
                             temperature=temperature, enable_thinking=enable_thinking)
                print(f"    Warmup {i+1}/3 done.", flush=True)
            except Exception as e:
                print(f"    Warmup {i+1}/3 failed: {e}", flush=True)

    def _one(i: int, item: Dict[str, Any]) -> Dict:
        try:
            return query_server(port, item, max_tokens, temperature=temperature,
                                 enable_thinking=enable_thinking)
        except Exception as e:
            return {"elapsed": 0, "completion_tokens": 0, "tokens_per_sec": 0, "error": str(e)}

    if concurrency <= 1:
        # 顺序: 保留逐条进度打印
        wall_start = time.perf_counter()
        for i, item in enumerate(items):
            short_q = item["question"][:50].replace("\n", " ")
            short_q = short_q + "..." if len(item["question"]) > 50 else short_q
            print(f"  [{i+1:3d}/{len(items)}] {short_q}", end="", flush=True)
            result = _one(i, item)
            if "error" in result:
                print(f"  -> ERROR: {result['error']}")
            else:
                print(f"  -> {result['completion_tokens']:3d} tok, {result['tokens_per_sec']:.1f} tok/s")
            results.append(result)
        wall_time = time.perf_counter() - wall_start
    else:
        # 并发: 线程池同时打 concurrency 条, 完成一条打印一次
        results = [None] * len(items)
        done = 0
        wall_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = {ex.submit(_one, i, item): i for i, item in enumerate(items)}
            for fut in as_completed(futs):
                i = futs[fut]
                result = fut.result()
                results[i] = result
                done += 1
                if "error" in result:
                    print(f"  [{done:3d}/{len(items)}] -> ERROR: {result['error']}", flush=True)
                else:
                    print(
                        f"  [{done:3d}/{len(items)}] -> "
                        f"{result['completion_tokens']:3d} tok",
                        flush=True,
                    )
        wall_time = time.perf_counter() - wall_start

    return results, wall_time


def fetch_spec_decode_metrics(port: int, engine: str = "vllm") -> Dict[str, float]:
    """从 Prometheus /metrics 端点抓取投机解码指标。

    vllm: 抓累计 counter (num_accepted/num_draft tokens), 是 server 启动至今的累计值;
          想得到单个数据集的接受率, 需在该集前后各取一次用差值 (见 spec_metrics_delta)。
    sglang: 抓 Gauge sglang:spec_accept_length (平均接受长度) 和 sglang:spec_accept_rate
            (接受率)。注意是 Gauge(最近值)而非 counter, 不能做差值, 直接取当前值。
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
                # sglang:spec_accept_length{...} <val>  /  sglang:spec_accept_rate{...} <val>
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
    """汇总单个数据集的 baseline / SD 指标。

    吞吐用墙钟总耗时 (no_sd_wall / sd_wall) 计算: 并发下各请求时间重叠,
    求和会重复计时, 只有墙钟反映系统真实 tok/s。
    """
    no_sd_tokens = sum(r["completion_tokens"] for r in no_sd_results)
    no_sd_time = no_sd_wall
    no_sd_tps = no_sd_tokens / no_sd_time if no_sd_time > 0 else 0

    sd_tokens = sum(r["completion_tokens"] for r in sd_results)
    sd_time = sd_wall
    sd_tps = sd_tokens / sd_time if sd_time > 0 else 0

    speedup = sd_tps / no_sd_tps if no_sd_tps > 0 else 0
    accept_rate = spec_metrics.get("accept_rate")
    # 接受长度 (平均每步被接受的 token 数, 含保底的 1 个): sglang 直接给出;
    # vllm 没有该 Gauge, 由 accepted/draft 换算: accept_length = accepted/draft * spec_tokens + 1
    # (这里 summarize 拿不到 spec_tokens, 故 vllm 的换算放到 print 时按全局 spec_tokens 做)。
    accept_length = spec_metrics.get("accept_length")

    return {
        "dataset": name,
        "num_baseline": len(no_sd_results),
        "num_sd": len(sd_results),
        "baseline_tps": no_sd_tps,
        "sd_tps": sd_tps,
        "speedup": speedup,
        "accept_rate": accept_rate,
        "accept_length": accept_length,
        "accepted_tokens": spec_metrics.get("accepted_tokens"),
        "draft_tokens": spec_metrics.get("draft_tokens"),
        "baseline_tokens": no_sd_tokens,
        "baseline_time": no_sd_time,
        "sd_tokens": sd_tokens,
        "sd_time": sd_time,
    }


def print_final_table(summaries: List[Dict[str, Any]], engine: str = "vllm",
                       num_speculative_tokens: int = 5):
    """打印每个数据集 + 总计的加速比 / 接受率 / 接受长度表格。"""

    def _accept_length(s: Dict[str, Any]) -> Optional[float]:
        # sglang: 直接读到的 accept_length; vllm: 由接受率换算 acc/draft*spec_tokens + 1
        if s.get("accept_length") is not None:
            return s["accept_length"]
        if s.get("accept_rate") is not None:
            return s["accept_rate"] * num_speculative_tokens + 1.0
        return None

    print("\n")
    print("=" * 90)
    print(f"        BENCHMARK RESULTS (DFlash, {engine})  —  加速比 / 接受率 / 接受长度")
    print("=" * 90)
    header = (
        f"  {'Dataset':<10} {'N':>4} {'Baseline':>11} {'DFlash':>11} "
        f"{'Speedup':>9} {'AcceptRate':>11} {'AcceptLen':>10}"
    )
    print(header)
    print(f"  {'':<10} {'':>4} {'tok/s':>11} {'tok/s':>11} {'':>9} "
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
            f"  {s['dataset']:<10} {s['num_sd']:>4d} "
            f"{s['baseline_tps']:>11.1f} {s['sd_tps']:>11.1f} "
            f"{s['speedup']:>8.2f}x {ar_str:>11} {al_str:>10}"
        )
        tot_b_tok += s["baseline_tokens"]
        tot_b_time += s["baseline_time"]
        tot_s_tok += s["sd_tokens"]
        tot_s_time += s["sd_time"]
        if s["accepted_tokens"] is not None and s["draft_tokens"] is not None:
            tot_acc += s["accepted_tokens"]
            tot_drf += s["draft_tokens"]

    print("-" * 90)
    overall_b_tps = tot_b_tok / tot_b_time if tot_b_time > 0 else 0
    overall_s_tps = tot_s_tok / tot_s_time if tot_s_time > 0 else 0
    overall_speedup = overall_s_tps / overall_b_tps if overall_b_tps > 0 else 0
    if tot_drf > 0:
        # vllm: 用全局 acc/draft 算总接受率与接受长度
        overall_ar = f"{tot_acc / tot_drf:.4f}"
        overall_al = f"{tot_acc / tot_drf * num_speculative_tokens + 1.0:.2f}"
    elif al_vals:
        # sglang: Gauge 无法跨集求和, 用各集接受长度的简单平均
        overall_ar = "N/A"
        overall_al = f"{sum(al_vals) / len(al_vals):.2f}"
    else:
        overall_ar = overall_al = "N/A"
    print(
        f"  {'OVERALL':<10} {'':>4} "
        f"{overall_b_tps:>11.1f} {overall_s_tps:>11.1f} "
        f"{overall_speedup:>8.2f}x {overall_ar:>11} {overall_al:>10}"
    )
    print("=" * 90)
    print()


def _dump_server_tail(proc: subprocess.Popen, n: int = 6000):
    # 优先从重定向的日志文件读尾部 (DP 模式走这条); 否则回退到 PIPE。
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
    parser = argparse.ArgumentParser(
        description="Benchmark DFlash SD vs no-SD on multiple image+text benchmarks for Qwen2.5-VL (vLLM)"
    )
    parser.add_argument(
        "--model-path", type=str,
        default=DEFAULT_MODEL_PATH,
        help="VLM target 模型路径",
    )
    parser.add_argument(
        "--draft-model-path", type=str,
        default=DEFAULT_DRAFT_MODEL_PATH,
        help="DFlash draft 模型路径 (训练产出的 checkpoint)",
    )
    parser.add_argument(
        "--engine", type=str, default=DEFAULT_ENGINE, choices=["vllm", "sglang"],
        help="推理引擎: vllm (装在 vllm 环境) 或 sglang (装在 sglang 环境, 0.5.12+ "
             "原生支持 DFLASH)。两者 chat 接口都是 OpenAI 兼容",
    )
    parser.add_argument(
        "--reasoning-parser", type=str, default=DEFAULT_REASONING_PARSER,
        help="reasoning(thinking)解析器, 仅 sglang 用 (--reasoning-parser); Qwen3 系用 qwen3",
    )
    parser.add_argument(
        "--enable-thinking", dest="enable_thinking", action="store_true",
        default=DEFAULT_ENABLE_THINKING,
        help="开启 thinking(reasoning)模式 (请求带 chat_template_kwargs.enable_thinking)",
    )
    parser.add_argument(
        "--no-thinking", dest="enable_thinking", action="store_false",
        help="关闭 thinking 模式",
    )
    parser.add_argument(
        "--sglang-extra-args", type=str, default=DEFAULT_SGLANG_EXTRA_ARGS,
        help="透传给 sglang server 的额外 flag (空格分隔), 仅 --engine sglang 用。"
             "默认带 Qwen3.5-35B-A3B hybrid MoE DFlash 必需的 "
             "'--attention-backend fa3 --mamba-scheduler-strategy extra_buffer'; "
             "纯文本/非 MoE 模型可传空串 ''",
    )
    parser.add_argument(
        "--datasets", type=str, default=DEFAULT_DATASETS,
        help=f"逗号分隔的评测集 (超参数)。支持: {','.join(SUPPORTED_DATASETS)}",
    )
    parser.add_argument(
        "--num-questions", type=int, default=DEFAULT_NUM_QUESTIONS,
        help="每个数据集取多少条",
    )
    parser.add_argument("--num-speculative-tokens", type=int, default=DEFAULT_NUM_SPECULATIVE_TOKENS)
    parser.add_argument(
        "--speculative-method", type=str, default=DEFAULT_SPECULATIVE_METHOD,
        help="投机解码方法 (透传给 vLLM --speculative-config 的 method)。"
             "本项目验证过: dflash (默认, 适配 VLM)、eagle3; "
             "引擎另支持 eagle/ngram/medusa/mlp_speculator/draft_model (需自备 draft 权重)。"
             "详见脚本顶部 DEFAULT_SPECULATIVE_METHOD 处的注释",
    )
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE,
        help="采样温度。0.0=贪心 (投机解码按 argmax 严格匹配接受); "
             ">0=采样 (按 speculative sampling 概率接受, 对应训练优化的 expected "
             "acceptance)。eagle3/dflash 贪心严格匹配率常远低于采样接受率, "
             "想对齐训练指标/看真实接受率用 1.0",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=DEFAULT_GPU_MEMORY_UTILIZATION)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument(
        "--gpu-ids", type=str, default=DEFAULT_GPU_IDS,
        help="指定用哪几张卡 (注入 CUDA_VISIBLE_DEVICES), 如 '2' 或 '2,3'; "
             "不指定则用全部可见卡。须保证可见卡数 >= TP*DP",
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=DEFAULT_TENSOR_PARALLEL_SIZE,
        help="张量并行卡数; 35B 等大模型单卡放不下时设 >1 (如 2)。"
             "注意 attention head 数须能被整除 (Qwen2.5-VL-7B 是 28, 只能 1/2/4)",
    )
    parser.add_argument(
        "--data-parallel-size", type=int, default=DEFAULT_DATA_PARALLEL_SIZE,
        help="数据并行副本数; 小模型高并发吞吐评测时铺满多卡 (如 8)。"
             "总占卡数 = data_parallel_size * tensor_parallel_size。"
             "仅在 --concurrency >= 副本数时多卡才会都忙起来",
    )
    parser.add_argument(
        "--max-num-seqs", type=int, default=None,
        help="vLLM 单批最大并发序列数 (默认不传, 用 vLLM 默认 256)。单卡跑 hybrid "
             "MoE+Mamba (如 35B-A3B TP=1) 时须调低 (如 64), 否则 Mamba cache block "
             "不足导致 CUDA graph capture 失败起不来。",
    )
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY,
        help="同时在飞的请求数; 1=顺序单请求 (测单请求延迟/加速比), "
             ">1=并发打流 (测高并发系统吞吐, 让 data-parallel 多副本同时干活)。"
             "建议设为 data_parallel_size 的若干倍 (如 DP=8 时设 64)",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--skip-baseline", action="store_true",
        help="跳过 baseline, 只跑 SD (例如 baseline 已经测过)",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=DEFAULT_OUTPUT_DIR,
    )
    args = parser.parse_args()

    # 把 --sglang-extra-args 字符串拆成 list (空串 -> 空 list)
    sglang_extra = args.sglang_extra_args.split() if args.sglang_extra_args else []

    # (specforge 环境两者都没有)。导入失败给出明确的环境提示。
    if args.engine == "sglang":
        try:
            import sglang
            engine_version = sglang.__version__
        except Exception:
            print(
                "ERROR: 未找到 sglang。--engine sglang 必须在 `sglang` conda 环境下运行。\n"
                "       请先 `conda activate sglang` (需 0.5.12+ 才原生支持 DFLASH)。",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        try:
            import vllm
            engine_version = vllm.__version__
        except Exception:
            print(
                "ERROR: 未找到 vllm。--engine vllm 必须在 `vllm` conda 环境下运行。\n"
                "       请先 `conda activate vllm`, 或使用 "
                "examples/run_qwen2_5_vl_7b_dflash_eval.sh。",
                file=sys.stderr,
            )
            sys.exit(1)

    # 解析数据集超参, 预加载每个集的题目
    dataset_names = [d.strip() for d in args.datasets.split(",") if d.strip()]
    for name in dataset_names:
        if name not in SUPPORTED_DATASETS:
            print(
                f"ERROR: 不支持的数据集 '{name}'。支持: {list(SUPPORTED_DATASETS)}",
                file=sys.stderr,
            )
            sys.exit(1)
    dataset_items = {
        name: load_questions(name, args.num_questions) for name in dataset_names
    }

    print("=" * 78)
    print("     Multi-Dataset DFlash Speculative Decoding Benchmark (image+text)")
    print("=" * 78)
    print(f"  Target model: {args.model_path}")
    print(f"  Draft model:  {args.draft_model_path}")
    print(f"  Datasets:     " + ", ".join(
        f"{n}({len(dataset_items[n])})" for n in dataset_names
    ))
    print(f"  Per-dataset:  {args.num_questions} questions")
    print(f"  Max tokens:   {args.max_tokens}")
    print(f"  Temperature:  {args.temperature}  ({'greedy/argmax' if args.temperature == 0 else 'sampling'})")
    print(f"  Thinking:     {'ON' if args.enable_thinking else 'OFF'}"
          f"{' (reasoning-parser=' + args.reasoning_parser + ')' if args.engine == 'sglang' and args.enable_thinking else ''}")
    print(f"  Spec tokens:  {args.num_speculative_tokens}")
    print(f"  Method:       {args.speculative_method}")
    print(f"  Parallel:     TP={args.tensor_parallel_size} DP={args.data_parallel_size} "
          f"(共 {args.tensor_parallel_size * args.data_parallel_size} 卡)  "
          f"concurrency={args.concurrency}")
    print(f"  GPU:          {args.gpu_ids if args.gpu_ids else '(all visible)'}")
    print(f"  Engine:       {args.engine} {engine_version}")
    print("=" * 78)
    print()

    if args.data_parallel_size > 1 and args.concurrency < args.data_parallel_size:
        print(
            f"  ⚠️  警告: data-parallel-size={args.data_parallel_size} 但 "
            f"concurrency={args.concurrency} < 副本数。\n"
            f"      并发不足时多数副本空转, 多卡吞吐不会提升。建议 "
            f"--concurrency >= {args.data_parallel_size} (如 {args.data_parallel_size * 8})。\n",
            flush=True,
        )

    baseline_by_ds: Dict[str, List[Dict]] = {n: [] for n in dataset_names}
    sd_by_ds: Dict[str, List[Dict]] = {n: [] for n in dataset_names}
    spec_by_ds: Dict[str, Dict[str, float]] = {n: {} for n in dataset_names}
    # 每集墙钟耗时 (并发下吞吐必须按墙钟算, 不能用各请求 elapsed 求和)
    baseline_wall: Dict[str, float] = {n: 0.0 for n in dataset_names}
    sd_wall: Dict[str, float] = {n: 0.0 for n in dataset_names}

    # server 日志目录: 把 vLLM server 的 stdout 落到文件 (而非 PIPE), 避免 DP
    # 多副本日志撑爆管道导致死锁。用 launch 时间戳命名, 可 `tail -f` 看启动进度。
    os.makedirs(args.output_dir, exist_ok=True)
    launch_ts = time.strftime("%Y%m%d_%H%M%S")
    baseline_log = os.path.join(
        args.output_dir, f"server_{args.speculative_method}_baseline_{launch_ts}.log"
    )
    sd_log = os.path.join(
        args.output_dir, f"server_{args.speculative_method}_sd_{launch_ts}.log"
    )

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
            gpu_ids=args.gpu_ids,
            log_path=baseline_log,
            engine=args.engine,
            reasoning_parser=args.reasoning_parser,
            enable_thinking=args.enable_thinking,
            sglang_extra_args=sglang_extra,
            max_num_seqs=args.max_num_seqs,
        )
        print(f"  Waiting for server to be ready (this may take a few minutes)... [server log: {baseline_log}]", flush=True)
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

    # ====== Phase 2: With DFlash — 单 server 跑完所有集, 每集前后抓 counter 算接受率 ======
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
        gpu_ids=args.gpu_ids,
        log_path=sd_log,
        engine=args.engine,
        reasoning_parser=args.reasoning_parser,
        enable_thinking=args.enable_thinking,
        sglang_extra_args=sglang_extra,
        max_num_seqs=args.max_num_seqs,
    )
    print(f"  Waiting for server to be ready (this may take a few minutes)... [server log: {sd_log}]", flush=True)
    if not wait_for_server(args.port, sd_proc):
        print("  ERROR: SD server failed to start!")
        _dump_server_tail(sd_proc)
        kill_server(sd_proc)
        return
    print("  Server ready!")

    for di, name in enumerate(dataset_names):
        # warmup 只在第一个数据集之前做一次 (编译 kernel)。warmup 自身也会产生
        # draft tokens, 所以放在取基线 counter 之前, 保证每集接受率不含 warmup。
        if di == 0:
            warm = dataset_items[name][0]
            print(f"\n  Warming up before first dataset ({name})...", flush=True)
            for k in range(3):
                try:
                    query_server(args.port, warm, max_tokens=64, timeout=600,
                                 temperature=args.temperature,
                                 enable_thinking=args.enable_thinking)
                    print(f"    Warmup {k+1}/3 done.", flush=True)
                except Exception as e:
                    print(f"    Warmup {k+1}/3 failed: {e}", flush=True)

        # vllm: 该集开跑前后各抓一次累计 counter, 差值即本集的 accepted/draft tokens。
        # sglang: spec_accept_* 是 Gauge(最近值), before 抓了也用不上, 直接取 after。
        before = fetch_spec_decode_metrics(args.port, engine=args.engine)
        sd_by_ds[name], sd_wall[name] = run_benchmark(
            args.port, dataset_items[name], args.max_tokens,
            f"[{args.speculative_method}] {name}", warmup=False,
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
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    tag = "-".join(dataset_names)
    # 文件名带引擎名, 区分 vllm / sglang 两套结果
    model_tag = os.path.basename(args.model_path.rstrip("/")) or "model"
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
                    name: {
                        "baseline": baseline_by_ds[name],
                        "sd": sd_by_ds[name],
                    }
                    for name in dataset_names
                },
            },
            f,
            indent=2,
        )
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
