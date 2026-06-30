#!/bin/bash
# ==============================================================================
# Qwen2.5-VL-7B  dflash  训练 (单脚本三阶段, 8卡)
# ------------------------------------------------------------------------------
# 用 STAGE 变量切换三个阶段, 阶段间权重接力 (后一阶段用 --init-draft-path 加载前
# 一阶段最终 ckpt, 只 load 权重不续 optimizer, 全新 output-dir 重新开训):
#
#   STAGE=1  纯文本  target 仍是 VL 但喂纯文本 ShareGPT (--vlm-text-data),
#                    在线 tokenize 走 build_eagle3_dataset。
#   STAGE=2  图文    预 tokenize 带图数据 (OnlineMMDataset), 接力 stage1 权重。
#   STAGE=3  视频    视频多帧 + video adapter (压缩 vision token 加速), 接力 stage2。
#                    两条子路线由 FREEZE_BACKBONE 切换:
#                      FREEZE_BACKBONE=1  adapter 预热 (冻结 backbone 只训 adapter,
#                                         用 stage2 图文数据, lr=1e-3, 3 epoch)。
#                      FREEZE_BACKBONE=0  联合训练 (backbone+adapter, 视频数据,
#                                         lr=3e-4, 10 epoch)。[默认]
#
# 用法:
#   STAGE=1 nohup bash examples/run_qwen2_5_vl_7b_dflash.sh > logs/train_stage1.log 2>&1 &
#   STAGE=2 nohup bash examples/run_qwen2_5_vl_7b_dflash.sh > logs/train_stage2.log 2>&1 &
#   # stage3 先预热 adapter, 再联合微调:
#   STAGE=3 FREEZE_BACKBONE=1 nohup bash examples/run_qwen2_5_vl_7b_dflash.sh > logs/train_stage3_frozen.log 2>&1 &
#   STAGE=3 INIT_DRAFT_PATH=<frozen ckpt> nohup bash examples/run_qwen2_5_vl_7b_dflash.sh > logs/train_stage3_video.log 2>&1 &
#   # 调试单卡: CUDA_VISIBLE_DEVICES=0 STAGE=2 bash examples/run_qwen2_5_vl_7b_dflash.sh 1
#   # 覆盖接力 ckpt: INIT_DRAFT_PATH=/path/to/ckpt STAGE=2 bash ...
#
# 调试 NCCL 集合通信错位 (若 watchdog timeout 死锁):
#   先 `export SPECFORGE_STEP_BARRIER=1` 再启动, 让错位在出问题那一 step 立即暴露。
# ==============================================================================

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/$USER/cache/compiled_kernels
export SPECFORGE_DATA_NUM_PROC=32

STAGE=${STAGE:-2}

# 8 卡; 调试时手动覆盖, 例如 CUDA_VISIBLE_DEVICES=0 bash run_xxx.sh 1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
DEFAULT_NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
NUM_GPUS=${1:-$DEFAULT_NUM_GPUS}
ATTENTION_BACKEND=${2:-flex_attention}

# ---- 各阶段差异 ----
case "$STAGE" in
  1)  # 纯文本预热
    DRAFT_CONFIG=$ROOT_DIR/configs/qwen2.5-vl-7b-dflash.json
    TRAIN_DATA=$ROOT_DIR/datasets/train/ultrachat_train.jsonl
    OUTPUT_DIR=$ROOT_DIR/$USER/outputs/qwen2.5-vl-7b-dflash-text
    INIT_DRAFT_PATH=${INIT_DRAFT_PATH:-}
    STAGE_ARGS="--vlm-text-data --chat-template qwen"
    NUM_EPOCHS=6
    BATCH_SIZE=2
    LEARNING_RATE=6e-4
    LOSS_DECAY_GAMMA=7.0
    SAVE_INTERVAL=10000
    BLOCK_SIZE=16
    ;;
  2)  # 图文
    DRAFT_CONFIG=$ROOT_DIR/configs/qwen2.5-vl-7b-dflash.json
    TRAIN_DATA=$ROOT_DIR/datasets/train/qwen2.5-vl-7b-dflash.jsonl
    OUTPUT_DIR=$ROOT_DIR/$USER/outputs/qwen2.5-vl-7b-dflash-stage2
    INIT_DRAFT_PATH=${INIT_DRAFT_PATH:-$ROOT_DIR/qingyw/outputs/qwen2.5-vl-7b-dflash-text/epoch_6_step_77934}
    STAGE_ARGS="--min-pixels 200704 --max-pixels 1003520"
    NUM_EPOCHS=6
    BATCH_SIZE=1
    LEARNING_RATE=6e-4
    LOSS_DECAY_GAMMA=7.0
    SAVE_INTERVAL=7000
    BLOCK_SIZE=16
    ;;
  3)  # video adapter (两条子路线, 由 FREEZE_BACKBONE 切换)
    DRAFT_CONFIG=$ROOT_DIR/configs/qwen2.5-vl-7b-dflash-video-adapter.json
    INIT_DRAFT_PATH=${INIT_DRAFT_PATH:-$ROOT_DIR/qingyw/outputs/qwen2.5-vl-7b-dflash-stage2/epoch_6_step_50940}
    BATCH_SIZE=1
    BLOCK_SIZE=10
    LOSS_DECAY_GAMMA=5.0
    SAVE_INTERVAL=3000
    if [ "${FREEZE_BACKBONE:-0}" = "1" ]; then
      # adapter 预热: 冻结 backbone 只训 video_adapter, 用 stage2 同分布图文数据。
      # 为什么: 随机初始化的 resampler 在 kv_mode=draft_only 下从 step1 就用噪声替换
      #   draft KV, 把 stage2 训好的 backbone 冲垮 -> loss 爆 NaN (gate_init=0 只拦
      #   fuse_mask 加法路径, 拦不住 KV 替换路径)。先冻结 backbone 让 resampler 学会
      #   "把视觉压成 backbone 认识的表示", 收敛后再用 FREEZE_BACKBONE=0 联合微调。
      TRAIN_DATA=$ROOT_DIR/datasets/train/qwen2.5-vl-7b-dflash.jsonl
      OUTPUT_DIR=$ROOT_DIR/$USER/outputs/qwen2.5-vl-7b-dflash-stage3-adapter-frozen
      STAGE_ARGS="--freeze-draft-backbone --min-pixels 200704 --max-pixels 1003520 --trust-remote-code"
      NUM_EPOCHS=3
      LEARNING_RATE=1e-3
    else
      # 联合训练: backbone + adapter 一起训, 视频多帧数据。
      # 注意: 若 adapter 是随机初始化(未先做预热), 直接联合训练大概率 NaN ——
      #   建议先 FREEZE_BACKBONE=1 预热, 再 INIT_DRAFT_PATH=<frozen ckpt> 联合微调。
      TRAIN_DATA=$ROOT_DIR/datasets/train/qwen2.5-vl-7b-dflash_video.jsonl
      OUTPUT_DIR=$ROOT_DIR/$USER/outputs/qwen2.5-vl-7b-dflash-stage3-video-adapter
      STAGE_ARGS="--min-pixels 50176 --max-pixels 200704 --trust-remote-code"
      NUM_EPOCHS=10
      LEARNING_RATE=3e-4
    fi
    ;;
  *)
    echo "[run_qwen2_5_vl_7b_dflash] 未知 STAGE=$STAGE (应为 1/2/3)" >&2
    exit 1
    ;;
esac

# stage>=2 用 --init-draft-path 接力前一阶段权重
INIT_ARG=""
if [ -n "$INIT_DRAFT_PATH" ]; then
    INIT_ARG="--init-draft-path $INIT_DRAFT_PATH"
fi

echo "[run_qwen2_5_vl_7b_dflash] STAGE=$STAGE CUDA=$CUDA_VISIBLE_DEVICES NUM_GPUS=$NUM_GPUS ATTN=$ATTENTION_BACKEND"
echo "[run_qwen2_5_vl_7b_dflash] config=$DRAFT_CONFIG"
echo "[run_qwen2_5_vl_7b_dflash] data=$TRAIN_DATA"
echo "[run_qwen2_5_vl_7b_dflash] output=$OUTPUT_DIR"
echo "[run_qwen2_5_vl_7b_dflash] init_draft=${INIT_DRAFT_PATH:-<none>}"

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_dflash.py \
    --target-model-path $ROOT_DIR/model/Qwen2.5-VL-7B-Instruct \
    --target-model-backend hf \
    --draft-config-path $DRAFT_CONFIG \
    $INIT_ARG \
    --train-data-path $TRAIN_DATA \
    --output-dir $OUTPUT_DIR \
    --is-vlm \
    --mask-token-id 151657 \
    $STAGE_ARGS \
    --num-epochs $NUM_EPOCHS \
    --batch-size $BATCH_SIZE \
    --learning-rate $LEARNING_RATE \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --max-length 4096 \
    --attention-backend $ATTENTION_BACKEND \
    --num-anchors 512 \
    --loss-decay-gamma $LOSS_DECAY_GAMMA \
    --log-interval 50 \
    --save-interval $SAVE_INTERVAL \
    --report-to tensorboard \
    --block-size $BLOCK_SIZE
