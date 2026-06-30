#!/bin/bash
# DFlash 35B (Qwen3.5-VL-A3B) video-adapter 投机解码评测 (纯 PyTorch)
#
# 与 7B 版同逻辑, 差异:
#   - 35B target 单卡装不下: device_map=balanced 跨多卡分片 (SPECFORGE_TARGET_DEVICE_MAP)
#   - 不传 --min-pixels/--max-pixels: Qwen3VLProcessor 不接受 (数据生成时也没传)
#   - conda 环境: specforge_vl (transformers>=5.8)
#
# 用法:
#   conda activate specforge_vl
#   DRAFT_PATH=.../epoch_X_step_Y bash examples/run_qwen3.5_35b_a3b_dflash_spec_video_eval.sh
#   # 跑图文集: DATA=datasets/train/qwen3.5-35b-vl-dflash_stage2.jsonl bash ...

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

# 35B target 跨多卡 (device_map=balanced 按 CUDA_VISIBLE_DEVICES 自动放置)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export SPECFORGE_TARGET_DEVICE_MAP=${SPECFORGE_TARGET_DEVICE_MAP:-balanced}

TARGET=${TARGET:-$ROOT_DIR/model/Qwen3.5-35B-A3B}
DRAFT_PATH=${DRAFT_PATH:-$ROOT_DIR/outputs/qwen3.5-35b-a3b-dflash-stage3-adapter-frozen/epoch_0_step_3000}
DATA=${DATA:-$ROOT_DIR/datasets/train/qwen3.5-35b-vl-dflash_video.jsonl}
NUM_SAMPLES=${NUM_SAMPLES:-20}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}

echo "[35b-spec-eval] TARGET=$TARGET  DEVICE_MAP=$SPECFORGE_TARGET_DEVICE_MAP"
echo "[35b-spec-eval] DRAFT=$DRAFT_PATH"
echo "[35b-spec-eval] DATA=$DATA  NUM_SAMPLES=$NUM_SAMPLES"

python $ROOT_DIR/scripts/eval_dflash_spec_video.py \
    --target-model-path $TARGET \
    --draft-path $DRAFT_PATH \
    --data-path $DATA \
    --is-vlm \
    --max-length 4096 \
    --max-new-tokens $MAX_NEW_TOKENS \
    --temperature 0.0 \
    --num-samples $NUM_SAMPLES \
    --modes on,off \
    --trust-remote-code
