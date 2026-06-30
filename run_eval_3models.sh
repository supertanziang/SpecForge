#!/bin/bash
# 串行评测 3 模型 × 4 数据集 (各30条), vllm 引擎, 无 vision adapter
# 指标: speedup (SD vs baseline tokens/s), accept_rate, accept_length
# 2张卡: 2.5VL单卡, 35B用TP=2, eagle3单卡 —— 串行跑避免抢卡
#
# 用法: conda activate vllm; nohup bash run_eval_3models.sh > logs/eval_3models.log 2>&1 &

set -x
cd /prj/corp/crd/morpheus/lasvegas/china-scratch/ziantan/SpecForge

# vllm 缓存导到 china-scratch 防 HOME 配额爆 (memory: vllm-home-quota-compile-cache)
export VLLM_CACHE_ROOT=/prj/corp/crd/morpheus/lasvegas/china-scratch/ziantan/.cache/vllm
export TORCHINDUCTOR_CACHE_DIR=/prj/corp/crd/morpheus/lasvegas/china-scratch/ziantan/.cache/torchinductor_vllm
export TRITON_CACHE_DIR=/prj/corp/crd/morpheus/lasvegas/china-scratch/ziantan/.cache/triton_vllm
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" results

DATASETS="sqa,textvqa,videomme,longvideobench"
NUM_Q=30
QWEN25=/prj/corp/crd/morpheus/lasvegas/china-scratch/ziantan/ViSpec/model/Qwen2.5-VL-7B-Instruct
QWEN25_DRAFT=$PWD/qingyw/outputs/qwen2.5-vl-7b-dflash-stage2/epoch_6_step_50940
M35B=$PWD/model/Qwen3.5-35B-A3B
M35B_DRAFT=$PWD/model/Qwen3.5-35B-A3B-DFlash

echo "########## [1/3] Qwen2.5-VL-7B dflash (单卡) ##########"
CUDA_VISIBLE_DEVICES=0 python benchmark_sd_vlm.py \
    --engine vllm --speculative-method dflash \
    --model-path $QWEN25 --draft-model-path $QWEN25_DRAFT \
    --datasets $DATASETS --num-questions $NUM_Q \
    --num-speculative-tokens 5 \
    --max-tokens 512 --temperature 0.0 \
    --max-model-len 32768 --gpu-memory-utilization 0.85 \
    --port 30021 --output-dir results 2>&1
echo "########## [1/3] done ##########"

echo "########## [2/3] Qwen3.5-35B-A3B dflash (TP=2) ##########"
CUDA_VISIBLE_DEVICES=0,1 python benchmark_sd_vlm.py \
    --engine vllm --speculative-method dflash \
    --model-path $M35B --draft-model-path $M35B_DRAFT \
    --datasets $DATASETS --num-questions $NUM_Q \
    --tensor-parallel-size 2 \
    --num-speculative-tokens 5 \
    --max-tokens 512 --temperature 0.0 \
    --max-model-len 32768 --gpu-memory-utilization 0.85 \
    --max-num-seqs 64 \
    --port 30022 --output-dir results 2>&1
echo "########## [2/3] done ##########"

echo "########## [3/3] eagle3 (单卡) ##########"
CUDA_VISIBLE_DEVICES=0 python benchmark_sd_vlm.py \
    --engine vllm --speculative-method eagle3 \
    --model-path $QWEN25 --draft-model-path $PWD/model/qwen2.5-vl-7b-eagle3-sgl-en-zh \
    --datasets $DATASETS --num-questions $NUM_Q \
    --num-speculative-tokens 5 \
    --max-tokens 512 --temperature 0.0 \
    --max-model-len 32768 --gpu-memory-utilization 0.85 \
    --port 30023 --output-dir results 2>&1
echo "########## [3/3] done ##########"
echo "ALL EVAL DONE"
