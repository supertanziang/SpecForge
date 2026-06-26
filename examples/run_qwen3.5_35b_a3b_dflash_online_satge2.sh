#!/bin/bash
# ==============================================================================
# Qwen3.5-35B-A3B-VL  dflash  Stage 2(图文)在线训练
# ------------------------------------------------------------------------------
# stage1(纯文本, sglang backend)见 run_qwen3.5_35b_a3b_dflash_online.sh。
# 本脚本是 stage2:在 stage1 的 draft 权重基础上,用图文数据继续训练同一个 35B draft。
#
# 与 stage1 / 7B-VL 脚本的关键差异:
#   - VLM 必须用 hf backend(sglang 的 dflash 采集路径不跑视觉塔,会拒绝 --is-vlm)。
#   - 35B-VL 是 Qwen3_5MoeForConditionalGeneration,需要 transformers 支持
#     qwen3_5_moe。当前 specforge/dflash 环境的 4.57.1 不支持 ——
#     请用克隆+升级后的环境(默认 specforge_vl, transformers>=5.8)运行:
#       conda activate specforge_vl
#   - 35B 用 Qwen3VLProcessor,不接受 min_pixels/max_pixels(用
#     size={longest_edge,shortest_edge}),所以这里不传 --min-pixels/--max-pixels。
#   - --mask-token-id 用 248070(35B vocab 内的空槽, 与 draft config 一致),
#     不是 7B 的 151657。
#   - chat-template 用 qwen3.5(thinking 模板),与 stage1 一致。
#
# 训练数据须由 35B-VL 当 target 重新生成(ViSpec gen_stage2_parallel.sh, PIXELS=none),
# 再用 SpecForge/scripts/convert_gen_mm_to_jsonl.py 转 jsonl ——
# 不能直接用 Qwen2.5-VL 编码的 gen_mm_combined(token id 与 35B vocab 错位)。
#
# 用法:
#   conda activate specforge_vl
#   bash examples/run_qwen3.5_35b_a3b_dflash_online_satge2.sh
#   # 调试单卡:CUDA_VISIBLE_DEVICES=0 bash examples/run_qwen3.5_35b_a3b_dflash_online_satge2.sh 1
# ==============================================================================

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
# Allow caller to override compile-cache dir (the in-tree default is shared and
# may be locked down by other users' files — see china-scratch/.cache/torchinductor_dflash).
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$ROOT_DIR/cache/compiled_kernels}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$TORCHINDUCTOR_CACHE_DIR/triton}
export SPECFORGE_DATA_NUM_PROC=32

# 默认用全部 8 张卡;35B 全参(hf 全量加载) + 视觉塔显存远大于 7B,
# 单卡跑不下,需多卡。调试时手动覆盖,例如 CUDA_VISIBLE_DEVICES=0,1 bash run_xxx.sh 2
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# NUM_GPUS 默认与 CUDA_VISIBLE_DEVICES 对齐(逗号数 + 1)
DEFAULT_NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
NUM_GPUS=${1:-$DEFAULT_NUM_GPUS}

ATTENTION_BACKEND=${2:-flex_attention}

# num_anchors 可被环境变量覆盖(单卡/小显存时调小, 默认 512)
NUM_ANCHORS=${NUM_ANCHORS:-512}
MAX_LENGTH=${MAX_LENGTH:-2048}
# num_epochs 可被环境变量覆盖。target 前向是冻结重复计算(占单 step ~92%),
# 每个 epoch 对同一条数据重算一遍 35B —— draft 是 473M 小模型收敛快,
# 3 个 epoch 通常已够,默认 3 以省一半墙钟。需要更充分训练时设 NUM_EPOCHS=6。
NUM_EPOCHS=${NUM_EPOCHS:-3}

# stage1 产出的 draft checkpoint —— stage2 从它加载初始权重(新 output-dir, 不续 optimizer)。
# stage1 跑完后按真实 last checkpoint 路径调整(形如 epoch_N 或 epoch_N_step_M)。
INIT_DRAFT_PATH=${INIT_DRAFT_PATH:-$ROOT_DIR/outputs/qwen3.5-35a-a3b-dflash-opc}

echo "[run_qwen3.5_35b_a3b_dflash_online_satge2] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES NUM_GPUS=$NUM_GPUS ATTN=$ATTENTION_BACKEND"
echo "[run_qwen3.5_35b_a3b_dflash_online_satge2] init draft weights from: $INIT_DRAFT_PATH"

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_dflash.py \
    --target-model-path $ROOT_DIR/model/Qwen3.5-35B-A3B \
    --target-model-backend hf \
    --embedding-key model.language_model.embed_tokens.weight \
    --lm-head-key lm_head.weight \
    --draft-config-path $ROOT_DIR/configs/qwen3.5-35b-a3b-dflash.json \
    --train-data-path $ROOT_DIR/datasets/train/qwen3.5-35b-vl-dflash_stage2.jsonl \
    --output-dir $ROOT_DIR/outputs/qwen3.5-35b-a3b-dflash-stage2 \
    --init-draft-path $INIT_DRAFT_PATH \
    --is-vlm \
    --mask-token-id 248070 \
    --num-epochs $NUM_EPOCHS \
    --batch-size 1 \
    --learning-rate 6e-4 \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --max-length $MAX_LENGTH \
    --chat-template qwen3.5 \
    --attention-backend $ATTENTION_BACKEND \
    --num-anchors $NUM_ANCHORS \
    --loss-decay-gamma 7.0 \
    --log-interval 50 \
    --save-interval 7000 \
    --report-to tensorboard \
    --block-size 16
