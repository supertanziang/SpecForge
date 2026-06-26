#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels
export SPECFORGE_DATA_NUM_PROC=32

# 默认用全部 8 张卡;调试时手动覆盖,例如 CUDA_VISIBLE_DEVICES=0 bash run_xxx.sh 1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# NUM_GPUS 默认与 CUDA_VISIBLE_DEVICES 对齐(逗号数 + 1)
DEFAULT_NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
NUM_GPUS=${1:-$DEFAULT_NUM_GPUS}

ATTENTION_BACKEND=${2:-flex_attention}

echo "[run_qwen2_5_vl_7b_dflash_online] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES NUM_GPUS=$NUM_GPUS ATTN=$ATTENTION_BACKEND"

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_dflash.py \
    --target-model-path $ROOT_DIR/model/Qwen2.5-VL-7B-Instruct \
    --target-model-backend hf \
    --draft-config-path $ROOT_DIR/configs/qwen2.5-vl-7b-dflash.json \
    --train-data-path $ROOT_DIR/datasets/train/qwen2.5-vl-7b-dflash_evalmix_only.jsonl \
    --output-dir $ROOT_DIR/$USER/outputs/qwen2.5-vl-7b-dflash \
    --is-vlm \
    --min-pixels 200704 \
    --max-pixels 1003520 \
    --mask-token-id 151657 \
    --num-epochs 6 \
    --batch-size 1 \
    --learning-rate 6e-4 \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --max-length 4096 \
    --attention-backend $ATTENTION_BACKEND \
    --num-anchors 512 \
    --loss-decay-gamma 7.0 \
    --log-interval 50 \
    --save-interval 7000 \
    --report-to tensorboard \
    --block-size 16
