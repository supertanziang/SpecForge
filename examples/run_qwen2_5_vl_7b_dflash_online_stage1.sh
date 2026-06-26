#!/bin/bash
# 纯文本训练 dflash draft, 但 target 仍是 Qwen2.5-VL-7B(VLM)。
# 基于 run_qwen2_5_vl_7b_dflash_online.sh 修改:
#   - 加 --vlm-text-data: VLM target 但喂纯文本 ShareGPT conversations(无图),
#     走 build_eagle3_dataset 在线 tokenize, 而非预 tokenize 的 OnlineMMDataset。
#   - --train-data-path 换成 perfectblend_train.jsonl(标准 conversations 格式)
#   - 加 --chat-template qwen(文本路径需要)
#   - 去掉 --min-pixels/--max-pixels(纯文本用不到)
#   - 保留 --is-vlm / --mask-token-id / --draft-config-path(target 仍是 VL,
#     draft 的 hidden 维度/target_layer_ids 不变)
#   - 输出目录独立, 不覆盖带图训练的产物

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/$USER/cache/compiled_kernels
export SPECFORGE_DATA_NUM_PROC=32

# 默认用全部 8 张卡;调试时手动覆盖,例如 CUDA_VISIBLE_DEVICES=0 bash run_xxx.sh 1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# NUM_GPUS 默认与 CUDA_VISIBLE_DEVICES 对齐(逗号数 + 1)
DEFAULT_NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
NUM_GPUS=${1:-$DEFAULT_NUM_GPUS}

ATTENTION_BACKEND=${2:-flex_attention}

echo "[run_qwen2_5_vl_7b_dflash_text_online] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES NUM_GPUS=$NUM_GPUS ATTN=$ATTENTION_BACKEND"

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_dflash.py \
    --target-model-path $ROOT_DIR/model/Qwen2.5-VL-7B-Instruct \
    --target-model-backend hf \
    --draft-config-path $ROOT_DIR/configs/qwen2.5-vl-7b-dflash.json \
    --train-data-path $ROOT_DIR/datasets/train/ultrachat_train.jsonl \
    --output-dir $ROOT_DIR/$USER/outputs/qwen2.5-vl-7b-dflash-text \
    --is-vlm \
    --vlm-text-data \
    --chat-template qwen \
    --mask-token-id 151657 \
    --num-epochs 6 \
    --batch-size 2 \
    --learning-rate 6e-4 \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --max-length 4096 \
    --attention-backend $ATTENTION_BACKEND \
    --num-anchors 512 \
    --loss-decay-gamma 7.0 \
    --log-interval 50 \
    --save-interval 10000 \
    --report-to tensorboard \
    --block-size 16
