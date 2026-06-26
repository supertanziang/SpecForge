#!/bin/bash
# ==============================================================================
# Qwen2.5-VL-7B Eagle3 在线训练 —— VL target 喂纯文本 (--vlm-text-data, 8卡)
# ------------------------------------------------------------------------------
# 基于 run_qwen2.5_7b_vl_eagle3_online.sh 修改, 对标 dflash 的 --vlm-text-data:
#   - 数据换成 datasets/train/ultrachat_train.jsonl (纯文本 ShareGPT conversations, 无图)
#   - 加 --vlm-text-data: target 仍是 Qwen2.5-VL(VLM), 但数据走非 VLM 在线 tokenize 路径,
#     target forward 在 pixel_values=None 时退化为纯文本。
#   - 去掉 --vlm-online-mm 和预生成 vocab_mapping (纯文本路径在线自动生成 vocab_mapping)。
#   - 去掉 --min-pixels/--max-pixels (纯文本用不到; processor 仍会构建但不喂图)。
#   - 加 --chat-template qwen (纯文本路径需要)。
#   - 输出目录独立, 不覆盖带图训练的产物。
#
# 环境: dflash conda env (含 flash_attn 2.8.3, specforge 包, transformers 4.57)
#
# 用法:
#   conda activate dflash
#   bash examples/run_qwen2.5_7b_vl_eagle3_text_online.sh        # 默认8卡
#   CUDA_VISIBLE_DEVICES=0 bash examples/run_qwen2.5_7b_vl_eagle3_text_online.sh 1  # 单卡冒烟
# ==============================================================================
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/$USER/cache/compiled_kernels_eagle3

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
DEFAULT_NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
NUM_GPUS=${1:-$DEFAULT_NUM_GPUS}

# fa = flash_attn (需 dflash 环境), 比 flex_attention 更快; 可手动覆盖为 flex_attention/sdpa
ATTENTION_BACKEND=${2:-fa}

TRAIN_JSONL="$ROOT_DIR/datasets/train/ultrachat_train.jsonl"

echo "[eagle3_text_online] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES NUM_GPUS=$NUM_GPUS ATTN=$ATTENTION_BACKEND"

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path $ROOT_DIR/model/Qwen2.5-VL-7B-Instruct \
    --target-model-backend custom \
    --draft-model-config $ROOT_DIR/configs/qwen2-5-vl-7b-eagle3.json \
    --train-data-path "$TRAIN_JSONL" \
    --is-vlm \
    --vlm-text-data \
    --chat-template qwen \
    --output-dir $ROOT_DIR/$USER/outputs/qwen2.5-vl-7b-eagle3-text \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --max-length 4096 \
    --attention-backend $ATTENTION_BACKEND \
    --log-interval 50 \
    --save-interval 5000 \
    --report-to tensorboard \
    --cache-dir $ROOT_DIR/$USER/cache \
    --embedding-key model.embed_tokens.weight \
    --tp-size 1
