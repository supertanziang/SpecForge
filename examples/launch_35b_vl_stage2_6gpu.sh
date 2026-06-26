#!/bin/bash
# ==============================================================================
# Qwen3.5-35B-A3B-VL dflash stage2 训练启动脚本
#   - 6 卡 dp=3, 每 rank 用 2 张卡跑 target(device_map balanced)
#   - chunked lm_head(已在 specforge/core/dflash.py)
#   - num_epochs=3(target 重复计算是大头, draft 小模型 3 epoch 够)
# 用法(就这一条):
#   nohup bash examples/launch_35b_vl_stage2_6gpu.sh &
#   日志自动落到 results/stage2_35b_vl_<时间戳>.log
# ==============================================================================
ROOT_DIR=/prj/corp/crd/morpheus/lasvegas/china-scratch/ziantan/SpecForge
DFLASH_BIN=/prj/corp/crd/morpheus/lasvegas/china-scratch/ziantan/minianaconda3/envs/dflash/bin

TS=$(date +%Y%m%d_%H%M%S)
LOG=$ROOT_DIR/results/stage2_35b_vl_${TS}.log
mkdir -p "$(dirname "$LOG")"
echo "[launch] log=$LOG"

cd "$ROOT_DIR"
exec env -u CONDA_DEFAULT_ENV \
  PATH=$DFLASH_BIN:/usr/local/bin:/usr/bin:/bin \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
  SPECFORGE_PER_RANK_DEVICES=2 \
  SPECFORGE_TARGET_DEVICE_MAP=balanced \
  NUM_EPOCHS=3 \
  TORCHINDUCTOR_CACHE_DIR=/prj/corp/crd/morpheus/lasvegas/china-scratch/ziantan/.cache/torchinductor_dflash \
  TRITON_CACHE_DIR=/prj/corp/crd/morpheus/lasvegas/china-scratch/ziantan/.cache/triton_dflash \
  INIT_DRAFT_PATH=$ROOT_DIR/model/Qwen3.5-35B-A3B-DFlash \
  bash examples/run_qwen3.5_35b_a3b_dflash_online_satge2.sh 3 \
  > "$LOG" 2>&1
