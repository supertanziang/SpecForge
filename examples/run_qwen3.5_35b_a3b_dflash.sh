#!/bin/bash
# ==============================================================================
# Qwen3.5-35B-A3B  dflash  训练 (单脚本三阶段)
# ------------------------------------------------------------------------------
# 用 STAGE 变量切换三个阶段, 阶段间权重接力 (后一阶段 --init-draft-path 加载前一
# 阶段最终 ckpt, 只 load 权重不续 optimizer, 全新 output-dir 重新开训):
#
#   STAGE=1  纯文本  sglang backend 在线采集 (ultrachat/opc), 非 VLM, 4 卡。
#   STAGE=2  图文    hf backend (VLM 必须用 hf, sglang 采集路径不跑视觉塔),
#                    8 卡, 接力 stage1 权重。
#   STAGE=3  视频    hf backend, 接力 stage2。35B target 单卡装不下,
#                    每个 dp rank 用 2 卡 (PER_RANK_DEVICES=2, device_map=balanced)。
#                    两条子路线由 FREEZE_BACKBONE 切换:
#                      FREEZE_BACKBONE=1  adapter 预热 (video-adapter config, 冻结
#                                         backbone 只训 adapter, stage2 图文数据,
#                                         lr=1e-3, 3 epoch)。
#                      FREEZE_BACKBONE=0  视频联合训练 (普通 dflash config 无 adapter,
#                                         视频多帧数据补分布, lr=3e-4, 10 epoch)。[默认]
#
# 关键差异说明:
#   - 35B-VL 是 Qwen3_5MoeForConditionalGeneration, 需 transformers 支持 qwen3_5_moe。
#     specforge/dflash 环境的 4.57.1 不支持 —— 请用升级后的环境 (transformers>=5.8):
#       conda activate specforge_vl
#   - 35B 用 Qwen3VLProcessor, 不接受 min_pixels/max_pixels, 故不传 (走模型自带配置)。
#   - --mask-token-id 248070 (35B vocab 内空槽), 不是 7B 的 151657。仅 VLM 阶段需要。
#   - chat-template 用 qwen3.5 (thinking 模板)。
#   - stage2/3 训练数据须由 35B-VL 当 target 重新生成 (ViSpec gen_*, PIXELS=none),
#     再用 scripts/convert_gen_mm_to_jsonl.py 转 jsonl —— 不能直接用 Qwen2.5-VL 编码
#     的数据 (token id 与 35B vocab 错位)。
#
# 用法:
#   conda activate specforge_vl
#   STAGE=1 nohup bash examples/run_qwen3.5_35b_a3b_dflash.sh > logs/train_35b_stage1.log 2>&1 &
#   STAGE=2 nohup bash examples/run_qwen3.5_35b_a3b_dflash.sh > logs/train_35b_stage2.log 2>&1 &
#   # stage3 两条子路线:
#   STAGE=3 FREEZE_BACKBONE=1 nohup bash examples/run_qwen3.5_35b_a3b_dflash.sh > logs/train_35b_stage3_frozen.log 2>&1 &
#   STAGE=3 nohup bash examples/run_qwen3.5_35b_a3b_dflash.sh > logs/train_35b_stage3_video.log 2>&1 &
#   # 覆盖接力 ckpt: INIT_DRAFT_PATH=/path STAGE=2 bash ...
#   # 覆盖轮数/长度: NUM_EPOCHS=6 MAX_LENGTH=4096 STAGE=2 bash ...
#
# 调试 NCCL 集合通信错位 (若 watchdog timeout 死锁):
#   先 `export SPECFORGE_STEP_BARRIER=1` 再启动, 让错位在出问题那一 step 立即暴露。
# ==============================================================================

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$ROOT_DIR/cache/hf_datasets}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$ROOT_DIR/cache/compiled_kernels}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$TORCHINDUCTOR_CACHE_DIR/triton}
export SPECFORGE_DATA_NUM_PROC=32

STAGE=${STAGE:-2}
ATTENTION_BACKEND=${2:-flex_attention}
NUM_ANCHORS=${NUM_ANCHORS:-512}

DRAFT_CONFIG=$ROOT_DIR/configs/qwen3.5-35b-a3b-dflash.json

# ---- 各阶段差异 ----
case "$STAGE" in
  1)  # 纯文本, sglang 在线采集, 非 VLM, 4 卡
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
    NUM_GPUS=${1:-$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')}
    TRAIN_DATA=$ROOT_DIR/cache/dataset/opc_train_regen_first_turn.jsonl
    OUTPUT_DIR=$ROOT_DIR/outputs/qwen3.5-35a-a3b-dflash-opc
    INIT_DRAFT_PATH=${INIT_DRAFT_PATH:-}
    BACKEND_ARGS="--target-model-backend sglang --sglang-mem-fraction-static 0.5 --resume"
    VLM_ARGS=""
    NUM_EPOCHS=${NUM_EPOCHS:-10}
    BATCH_SIZE=2
    LEARNING_RATE=6e-4
    LOSS_DECAY_GAMMA=7.0
    SAVE_INTERVAL=10000
    BLOCK_SIZE=16
    MAX_LENGTH=${MAX_LENGTH:-4096}
    ;;
  2)  # 图文, hf backend, 8 卡
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
    NUM_GPUS=${1:-$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')}
    TRAIN_DATA=$ROOT_DIR/datasets/train/qwen3.5-35b-vl-dflash_stage2.jsonl
    OUTPUT_DIR=$ROOT_DIR/outputs/qwen3.5-35b-a3b-dflash-stage2
    INIT_DRAFT_PATH=${INIT_DRAFT_PATH:-$ROOT_DIR/outputs/qwen3.5-35a-a3b-dflash-opc}
    BACKEND_ARGS="--target-model-backend hf --lm-head-key lm_head.weight"
    VLM_ARGS="--is-vlm --mask-token-id 248070"
    NUM_EPOCHS=${NUM_EPOCHS:-3}
    BATCH_SIZE=1
    LEARNING_RATE=6e-4
    LOSS_DECAY_GAMMA=7.0
    SAVE_INTERVAL=7000
    BLOCK_SIZE=16
    MAX_LENGTH=${MAX_LENGTH:-2048}
    ;;
  3)  # video adapter, hf backend, 每 rank 2 卡装 target (两条子路线由 FREEZE_BACKBONE 切换)
    export SPECFORGE_PER_RANK_DEVICES=${SPECFORGE_PER_RANK_DEVICES:-2}
    export SPECFORGE_TARGET_DEVICE_MAP=${SPECFORGE_TARGET_DEVICE_MAP:-balanced}
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
    NUM_GPUS=${1:-$(( $(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}') / SPECFORGE_PER_RANK_DEVICES ))}
    VLM_ARGS="--is-vlm --mask-token-id 248070"
    BATCH_SIZE=1
    BLOCK_SIZE=10
    LOSS_DECAY_GAMMA=5.0
    SAVE_INTERVAL=3000
    MAX_LENGTH=${MAX_LENGTH:-4096}
    if [ "${FREEZE_BACKBONE:-0}" = "1" ]; then
      # adapter 预热: 冻结 backbone 只训 video_adapter, 用 stage2 图文数据。
      # 原理同 7B: 随机初始化的 resampler 在 kv_mode=draft_only 下会用噪声替换 draft KV
      #   冲垮 backbone -> NaN。先冻结 backbone 让 resampler 学会压视觉, 再联合微调。
      DRAFT_CONFIG=$ROOT_DIR/configs/qwen3.5-35b-a3b-dflash-video-adapter.json
      TRAIN_DATA=$ROOT_DIR/datasets/train/qwen3.5-35b-vl-dflash_stage2.jsonl
      OUTPUT_DIR=$ROOT_DIR/outputs/qwen3.5-35b-a3b-dflash-stage3-adapter-frozen
      INIT_DRAFT_PATH=${INIT_DRAFT_PATH:-$ROOT_DIR/outputs/qwen3.5-35b-a3b-dflash-stage2/epoch_2_step_49000}
      BACKEND_ARGS="--target-model-backend hf --lm-head-key lm_head.weight --freeze-draft-backbone --trust-remote-code"
      NUM_EPOCHS=${NUM_EPOCHS:-3}
      LEARNING_RATE=1e-3
    else
      # 视频联合训练: 用普通 dflash.json (无 adapter), 纯补视频多帧分布提升接受率。
      DRAFT_CONFIG=$ROOT_DIR/configs/qwen3.5-35b-a3b-dflash.json
      TRAIN_DATA=$ROOT_DIR/datasets/train/qwen3.5-35b-vl-dflash_video.jsonl
      OUTPUT_DIR=$ROOT_DIR/outputs/qwen3.5-35b-a3b-dflash-stage3-video
      INIT_DRAFT_PATH=${INIT_DRAFT_PATH:-$ROOT_DIR/outputs/qwen3.5-35b-a3b-dflash-stage2/epoch_2_step_49000}
      BACKEND_ARGS="--target-model-backend hf --lm-head-key lm_head.weight --trust-remote-code"
      NUM_EPOCHS=${NUM_EPOCHS:-10}
      LEARNING_RATE=3e-4
    fi
    ;;
  *)
    echo "[run_qwen3.5_35b_a3b_dflash] 未知 STAGE=$STAGE (应为 1/2/3)" >&2
    exit 1
    ;;
esac

# stage>=2 用 --init-draft-path 接力前一阶段权重 (stage1 走 --resume 同目录, 不传)
INIT_ARG=""
if [ -n "$INIT_DRAFT_PATH" ]; then
    INIT_ARG="--init-draft-path $INIT_DRAFT_PATH"
fi

echo "[run_qwen3.5_35b_a3b_dflash] STAGE=$STAGE CUDA=$CUDA_VISIBLE_DEVICES NUM_GPUS=$NUM_GPUS ATTN=$ATTENTION_BACKEND"
echo "[run_qwen3.5_35b_a3b_dflash] backend_args=$BACKEND_ARGS"
echo "[run_qwen3.5_35b_a3b_dflash] data=$TRAIN_DATA"
echo "[run_qwen3.5_35b_a3b_dflash] output=$OUTPUT_DIR"
echo "[run_qwen3.5_35b_a3b_dflash] init_draft=${INIT_DRAFT_PATH:-<none>}"
[ "$STAGE" = "3" ] && echo "[run_qwen3.5_35b_a3b_dflash] PER_RANK_DEVICES=$SPECFORGE_PER_RANK_DEVICES TARGET_DEVICE_MAP=$SPECFORGE_TARGET_DEVICE_MAP"

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_dflash.py \
    --target-model-path $ROOT_DIR/model/Qwen3.5-35B-A3B \
    $BACKEND_ARGS \
    --embedding-key model.language_model.embed_tokens.weight \
    --draft-config-path $DRAFT_CONFIG \
    $INIT_ARG \
    --train-data-path $TRAIN_DATA \
    --output-dir $OUTPUT_DIR \
    $VLM_ARGS \
    --num-epochs $NUM_EPOCHS \
    --batch-size $BATCH_SIZE \
    --learning-rate $LEARNING_RATE \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --max-length $MAX_LENGTH \
    --chat-template qwen3.5 \
    --attention-backend $ATTENTION_BACKEND \
    --num-anchors $NUM_ANCHORS \
    --loss-decay-gamma $LOSS_DECAY_GAMMA \
    --log-interval 50 \
    --save-interval $SAVE_INTERVAL \
    --report-to tensorboard \
    --block-size $BLOCK_SIZE
