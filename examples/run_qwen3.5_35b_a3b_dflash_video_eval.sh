#!/bin/bash
# 评测 Qwen3.5-35B-A3B 在视频数据集 (Video-MME / LongVideoBench) 上的 DFlash 加速效果。
# 对比两个 draft: stage2 (VLM训练) vs 原始 DFlash (纯文本训练)。
#
# 用法: conda activate vllm && bash examples/run_qwen3.5_35b_a3b_dflash_video_eval.sh
#
# 前置: 先在 specforge 环境导出视频数据集:
#   conda activate specforge
#   python scripts/export_video_eval_datasets.py --datasets videomme,longvideobench --max-per-dataset 50

set -e
cd "$(dirname "$0")/.."

MODEL="./model/Qwen3.5-35B-A3B"
DRAFT_STAGE2="./outputs/qwen3.5-35b-a3b-dflash-stage2/epoch_2_step_42000"
DRAFT_ORIGINAL="./model/Qwen3.5-35B-A3B-DFlash"
DATASETS="videomme,longvideobench"
NUM_Q=50
TP=2
ENGINE=vllm

echo "=========================================="
echo "  35B Video Eval: Stage2 DFlash draft"
echo "=========================================="
python benchmark_sd_vlm.py \
    --engine $ENGINE \
    --model-path $MODEL \
    --draft-model-path $DRAFT_STAGE2 \
    --datasets $DATASETS \
    --num-questions $NUM_Q \
    --tensor-parallel-size $TP \
    --num-speculative-tokens 5 \
    --enable-thinking \
    --max-tokens 4096 \
    --max-model-len 32768 \
    --temperature 0.0 \
    --port 30010

echo ""
echo "=========================================="
echo "  35B Video Eval: Original DFlash draft"
echo "=========================================="
python benchmark_sd_vlm.py \
    --engine $ENGINE \
    --model-path $MODEL \
    --draft-model-path $DRAFT_ORIGINAL \
    --datasets $DATASETS \
    --num-questions $NUM_Q \
    --tensor-parallel-size $TP \
    --num-speculative-tokens 5 \
    --enable-thinking \
    --max-tokens 4096 \
    --max-model-len 32768 \
    --temperature 0.0 \
    --port 30010

echo ""
echo "########## 35B VIDEO EVAL DONE ##########"
