#!/bin/bash
# 基于 run_qwen3_8b_dflash_online.sh 修改：
#   - GPU 8 -> 2 (本机只有 2 张 A100-80GB)
#   - 训练数据 perfectblend(不存在) -> sharegpt_train.jsonl
#   - target model -> 本地 HF cache 路径(离线)
#   - target-model-backend sglang -> hf (2 卡环境更稳妥, target+draft 同卡)
#   - report-to wandb -> tensorboard (日志写到 outputs/.../runs)
#   - batch-size 4 -> 2 (2 卡 hf backend 显存考虑)

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels
export SPECFORGE_DATA_NUM_PROC=32
export HF_HUB_OFFLINE=1
export SGLANG_DISABLE_CUDNN_CHECK=1

NUM_GPUS=${1:-2}
ATTENTION_BACKEND=${2:-flex_attention}

TARGET_MODEL_PATH=/prj/corp/crd/morpheus/lasvegas/china-scratch/ziantan/EAGLE/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_dflash.py \
    --target-model-path $TARGET_MODEL_PATH \
    --draft-config-path $ROOT_DIR/configs/qwen3-8b-dflash.json \
    --train-data-path $ROOT_DIR/cache/dataset/sharegpt_train.jsonl \
    --output-dir $ROOT_DIR/outputs/qwen3-8b-dflash-sharegpt \
    --num-epochs 6 \
    --batch-size 2 \
    --learning-rate 6e-4 \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --max-length 3072 \
    --chat-template qwen \
    --attention-backend $ATTENTION_BACKEND \
    --loss-decay-gamma 7.0 \
    --log-interval 50 \
    --save-interval 1000 \
    --report-to tensorboard \
    --target-model-backend hf \
    --block-size 16 \
    --num-anchors 512
