#!/bin/bash
# 评测 Qwen2.5-VL-7B 在视频数据集 (Video-MME / LongVideoBench) 上的 DFlash 加速效果。
#
# 用法: conda activate vllm && bash examples/run_qwen2.5_7b_vl_dflash_video_eval.sh
#
# 前置: 先在 specforge 环境导出视频数据集:
#   conda activate specforge
#   python scripts/export_video_eval_datasets.py --datasets videomme,longvideobench --max-per-dataset 50

set -e
cd "$(dirname "$0")/.."

MODEL="./model/Qwen2.5-VL-7B-Instruct"
DRAFT_STAGE2="./qingyw/outputs/qwen2.5-vl-7b-dflash-stage2/epoch_6_step_50940"
DATASETS="videomme,longvideobench"
NUM_Q=50
TP=1
ENGINE=vllm

echo "=========================================="
echo "  7B Video Eval: Stage2 DFlash draft"
echo "=========================================="
python benchmark_sd_vlm.py \
    --engine $ENGINE \
    --model-path $MODEL \
    --draft-model-path $DRAFT_STAGE2 \
    --datasets $DATASETS \
    --num-questions $NUM_Q \
    --tensor-parallel-size $TP \
    --num-speculative-tokens 5 \
    --max-tokens 512 \
    --max-model-len 32768 \
    --temperature 0.0 \
    --port 30000

echo ""
echo "########## 7B VIDEO EVAL DONE ##########"
