#!/bin/bash
# DFlash 7B (Qwen2.5-VL) video-adapter 投机解码评测 (纯 PyTorch, 单卡)
#
# 对每条样本跑 开/关 adapter 两种模式, 对比 acceptance length / tokens-s /
# draft KV 压缩比, 验证 video_adapter 的 compress 是否带来加速、是否掉接受率。
#
# 关键: --min-pixels/--max-pixels 必须与生成训练/评测数据时一致 (50176/200704),
#   否则 processor 产出的视觉 token 数与 input_ids 里的占位符对不上 (脚本会 assert)。
#
# 用法:
#   conda activate dflash
#   DRAFT_PATH=.../epoch_0_step_3000 bash examples/run_qwen2_5_vl_7b_dflash_spec_video_eval.sh
#   # 默认数据=视频集; 跑图文集: DATA=datasets/train/qwen2.5-vl-7b-dflash.jsonl bash ...

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

TARGET=${TARGET:-/prj/corp/crd/morpheus/lasvegas/china-scratch/ziantan/ViSpec/model/Qwen2.5-VL-7B-Instruct}
DRAFT_PATH=${DRAFT_PATH:-$ROOT_DIR/ziantan/outputs/qwen2.5-vl-7b-dflash-stage3-adapter-frozen/epoch_0_step_3000}
DATA=${DATA:-$ROOT_DIR/datasets/train/qwen2.5-vl-7b-dflash_video.jsonl}
NUM_SAMPLES=${NUM_SAMPLES:-20}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}

echo "[7b-spec-eval] TARGET=$TARGET"
echo "[7b-spec-eval] DRAFT=$DRAFT_PATH"
echo "[7b-spec-eval] DATA=$DATA  NUM_SAMPLES=$NUM_SAMPLES"

python $ROOT_DIR/scripts/eval_dflash_spec_video.py \
    --target-model-path $TARGET \
    --draft-path $DRAFT_PATH \
    --data-path $DATA \
    --is-vlm \
    --min-pixels 50176 \
    --max-pixels 200704 \
    --max-length 4096 \
    --max-new-tokens $MAX_NEW_TOKENS \
    --temperature 0.0 \
    --num-samples $NUM_SAMPLES \
    --modes on,off \
    --trust-remote-code
