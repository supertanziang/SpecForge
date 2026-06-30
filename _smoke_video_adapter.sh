#!/bin/bash
# 短训练验证: video adapter 损失是否下降。单卡, 视频数据, log 频繁。
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$SCRIPT_DIR
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/$USER/cache/compiled_kernels
export SPECFORGE_DATA_NUM_PROC=8
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "[smoke] video-adapter short training on GPU $CUDA_VISIBLE_DEVICES"

torchrun \
    --standalone \
    --nproc_per_node 1 \
    $ROOT_DIR/scripts/train_dflash.py \
    --target-model-path $ROOT_DIR/model/Qwen2.5-VL-7B-Instruct \
    --target-model-backend hf \
    --draft-config-path $ROOT_DIR/configs/qwen2.5-vl-7b-dflash-video-adapter.json \
    --train-data-path $ROOT_DIR/datasets/train/qwen2.5-vl-7b-dflash_video.jsonl \
    --output-dir $ROOT_DIR/$USER/outputs/video-adapter-smoke \
    --is-vlm \
    --min-pixels 50176 \
    --max-pixels 200704 \
    --mask-token-id 151657 \
    --num-epochs 1 \
    --batch-size 1 \
    --learning-rate 6e-4 \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --max-length 4096 \
    --attention-backend sdpa \
    --num-anchors 128 \
    --loss-decay-gamma 7.0 \
    --log-interval 5 \
    --save-interval 99999 \
    --report-to tensorboard \
    --block-size 16
