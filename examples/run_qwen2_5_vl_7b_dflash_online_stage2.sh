#!/bin/bash
# Stage2: 带图继续训练 dflash draft, 接力纯文本 stage1 训出来的权重。
# 基于 run_qwen2_5_vl_7b_dflash_online.sh 修改:
#   - 加 --init-draft-path: 用 stage1(纯文本)的最终 ckpt 初始化 draft 权重,
#     只 load 权重, 不恢复 optimizer/scheduler/step(全新 output-dir 重新开训)。
#     注意: 不能用 --resume(那是恢复同一 output-dir 的训练, 且会覆盖 --init-draft-path)。
#   - --train-data-path 换成 qwen2.5-vl-7b-dflash.jsonl(预 tokenize 带图数据)
#   - --output-dir 独立, 不覆盖 stage1 或带图 evalmix 的产物
#   - draft 结构(hidden_size/num_hidden_layers/block_size/target_layer_ids/mask_token_id)
#     与 stage1 完全一致, 所以权重可直接接力。

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/$USER/cache/compiled_kernels
export SPECFORGE_DATA_NUM_PROC=32

# stage1(纯文本)训出来的最终 ckpt, 用于初始化 draft 权重。
INIT_DRAFT_PATH=${INIT_DRAFT_PATH:-$ROOT_DIR/qingyw/outputs/qwen2.5-vl-7b-dflash-text/epoch_6_step_77934}

# 默认用全部 8 张卡;调试时手动覆盖,例如 CUDA_VISIBLE_DEVICES=0 bash run_xxx.sh 1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# NUM_GPUS 默认与 CUDA_VISIBLE_DEVICES 对齐(逗号数 + 1)
DEFAULT_NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
NUM_GPUS=${1:-$DEFAULT_NUM_GPUS}

ATTENTION_BACKEND=${2:-flex_attention}

echo "[run_qwen2_5_vl_7b_dflash_online_stage2] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES NUM_GPUS=$NUM_GPUS ATTN=$ATTENTION_BACKEND"
echo "[run_qwen2_5_vl_7b_dflash_online_stage2] INIT_DRAFT_PATH=$INIT_DRAFT_PATH"

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_dflash.py \
    --target-model-path $ROOT_DIR/model/Qwen2.5-VL-7B-Instruct \
    --target-model-backend hf \
    --draft-config-path $ROOT_DIR/configs/qwen2.5-vl-7b-dflash.json \
    --init-draft-path $INIT_DRAFT_PATH \
    --train-data-path $ROOT_DIR/datasets/train/qwen2.5-vl-7b-dflash.jsonl \
    --output-dir $ROOT_DIR/$USER/outputs/qwen2.5-vl-7b-dflash-stage2 \
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
