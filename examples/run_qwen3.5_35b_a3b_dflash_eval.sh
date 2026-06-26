#!/bin/bash
# ==============================================================================
# Qwen3.5-35B-A3B DFlash 投机解码 评测脚本 (vLLM, 图文多数据集)
# ------------------------------------------------------------------------------
# 测 z-lab/Qwen3.5-35B-A3B-DFlash 这个 draft model (DM) 搭配 target model (TM)
# Qwen/Qwen3.5-35B-A3B 的加速比。TM 是 VLM MoE (Qwen3_5MoeForConditionalGeneration),
# 所以沿用 VLM 版评测脚本 benchmark_sd_vlm.py (图文多数据集)。
#
# 与 7B 版 (run_qwen2_5_vl_7b_dflash_eval.sh) 的区别:
#   - TM 是 72GB 的 35B-A3B MoE, 单卡 A100-80GB 放权重后 KV cache 很紧, 默认用
#     张量并行 TP=2 跨两卡, 因此 GPU 默认 "0,1", TENSOR_PARALLEL_SIZE 默认 2;
#   - 模型路径指向 model/Qwen3.5-35B-A3B (软链接复用) 和 model/Qwen3.5-35B-A3B-DFlash。
#
# 关键: 评测引擎 vLLM 只装在名为 `vllm` 的 conda 环境里; 数据集索引导出要在
#       `specforge` 环境 (datasets 库在 vllm 环境不可用)。本脚本会自动处理两者。
#
# 用法 (需在分到 A100-80GB 的计算节点上跑, 例如已 bsub 进去的 8 卡节点):
#     bash examples/run_qwen3.5_35b_a3b_dflash_eval.sh                       # 三集各 50 条, TP=2
#     DATASETS=mmvet NUM_QUESTIONS=50 bash examples/run_qwen3.5_35b_a3b_dflash_eval.sh
#     GPU=0,1,2,3 TENSOR_PARALLEL_SIZE=4 bash examples/run_qwen3.5_35b_a3b_dflash_eval.sh
# ==============================================================================
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")
cd "$ROOT_DIR"

CONDA_ROOT="${CONDA_ROOT:-/prj/corp/crd/morpheus/lasvegas/china-scratch/ziantan/minianaconda3}"
VLLM_ENV="${VLLM_ENV:-vllm}"
SPECFORGE_ENV="${SPECFORGE_ENV:-specforge}"

# 评测集超参 (逗号分隔, 支持 mmvet,sqa,mme,textvqa) 与每集条数
DATASETS="${DATASETS:-mmvet,sqa,mme}"
NUM_QUESTIONS="${NUM_QUESTIONS:-50}"

if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
else
    echo "ERROR: 找不到 conda.sh ($CONDA_ROOT/etc/profile.d/conda.sh)" >&2
    exit 1
fi

# ------------------------------------------------------------------------------
# 1. (specforge 环境) 缺任一数据集索引时, 导出图片 + jsonl 索引
# ------------------------------------------------------------------------------
need_export=0
IFS=',' read -ra _DS <<< "$DATASETS"
for ds in "${_DS[@]}"; do
    if [ ! -f "$ROOT_DIR/datasets/eval/${ds}_index.jsonl" ]; then
        need_export=1
    fi
done

if [ "$need_export" = "1" ]; then
    echo "[eval] 部分数据集索引缺失, 在 $SPECFORGE_ENV 环境导出: $DATASETS"
    conda activate "$SPECFORGE_ENV"
    python "$ROOT_DIR/scripts/export_eval_datasets.py" \
        --datasets "$DATASETS" --max-per-dataset 200
    conda deactivate
fi

# ------------------------------------------------------------------------------
# 2. 切换到 vllm conda 环境 (评测必须在此环境跑)
# ------------------------------------------------------------------------------
echo "[eval] conda activate $VLLM_ENV"
conda activate "$VLLM_ENV"

python -c "import vllm; print('[eval] vllm', vllm.__version__)" || {
    echo "ERROR: '$VLLM_ENV' 环境里没有 vllm" >&2
    exit 1
}

# ------------------------------------------------------------------------------
# 3. 评测参数 (均可用同名环境变量覆盖)
# ------------------------------------------------------------------------------
TARGET_MODEL="${TARGET_MODEL:-$ROOT_DIR/model/Qwen3.5-35B-A3B}"
CKPT="${CKPT:-$ROOT_DIR/model/Qwen3.5-35B-A3B-DFlash}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-5}"
MAX_TOKENS="${MAX_TOKENS:-512}"
# 35B-A3B 72GB: 默认 TP=2 跨两卡。GPU 列表与 TP 卡数要一致。
GPU="${GPU:-0,1}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
PORT="${PORT:-30000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

export CUDA_VISIBLE_DEVICES="$GPU"

echo "[eval] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (TP=$TENSOR_PARALLEL_SIZE)"
echo "[eval] target=$TARGET_MODEL"
echo "[eval] draft (dflash DM)=$CKPT"
echo "[eval] datasets=$DATASETS  num_questions(每集)=$NUM_QUESTIONS  spec_tokens=$NUM_SPEC_TOKENS"

# ------------------------------------------------------------------------------
# 4. 跑评测 (baseline vs DFlash 双阶段对照, 多数据集)
# ------------------------------------------------------------------------------
python "$ROOT_DIR/benchmark_sd_vlm.py" \
    --model-path "$TARGET_MODEL" \
    --draft-model-path "$CKPT" \
    --datasets "$DATASETS" \
    --num-questions "$NUM_QUESTIONS" \
    --num-speculative-tokens "$NUM_SPEC_TOKENS" \
    --max-tokens "$MAX_TOKENS" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --port "$PORT"
