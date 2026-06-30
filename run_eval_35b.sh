#!/bin/bash
# 单独补跑 35B dflash (TP=2 + 禁custom-all-reduce, 已在benchmark脚本层加flag), 4数据集各30条
# 用法: conda activate vllm; nohup bash run_eval_35b.sh > logs/eval_35b.log 2>&1 &

set -x
cd /prj/corp/crd/morpheus/lasvegas/china-scratch/ziantan/SpecForge

export VLLM_CACHE_ROOT=/prj/corp/crd/morpheus/lasvegas/china-scratch/ziantan/.cache/vllm
export TORCHINDUCTOR_CACHE_DIR=/prj/corp/crd/morpheus/lasvegas/china-scratch/ziantan/.cache/torchinductor_vllm
export TRITON_CACHE_DIR=/prj/corp/crd/morpheus/lasvegas/china-scratch/ziantan/.cache/triton_vllm
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" results

CUDA_VISIBLE_DEVICES=0,1 python benchmark_sd_vlm.py \
    --engine vllm --speculative-method dflash \
    --model-path $PWD/model/Qwen3.5-35B-A3B \
    --draft-model-path $PWD/model/Qwen3.5-35B-A3B-DFlash \
    --datasets sqa,textvqa,videomme,longvideobench --num-questions 30 \
    --tensor-parallel-size 2 \
    --num-speculative-tokens 5 \
    --max-tokens 512 --temperature 0.0 \
    --max-model-len 16384 --gpu-memory-utilization 0.88 \
    --max-num-seqs 64 \
    --port 30042 --output-dir results 2>&1
echo "35B EVAL DONE"
