#!/bin/bash
# 补跑 35B(TP=1单卡) + eagle3(aux层修复) , vllm, 4数据集各30条
# 用法: conda activate vllm; nohup bash run_eval_rerun.sh > logs/eval_rerun.log 2>&1 &

set -x
cd /prj/corp/crd/morpheus/lasvegas/china-scratch/ziantan/SpecForge

export VLLM_CACHE_ROOT=/prj/corp/crd/morpheus/lasvegas/china-scratch/ziantan/.cache/vllm
export TORCHINDUCTOR_CACHE_DIR=/prj/corp/crd/morpheus/lasvegas/china-scratch/ziantan/.cache/torchinductor_vllm
export TRITON_CACHE_DIR=/prj/corp/crd/morpheus/lasvegas/china-scratch/ziantan/.cache/triton_vllm
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" results

DATASETS="sqa,textvqa,videomme,longvideobench"
NUM_Q=30
QWEN25=/prj/corp/crd/morpheus/lasvegas/china-scratch/ziantan/ViSpec/model/Qwen2.5-VL-7B-Instruct
M35B=$PWD/model/Qwen3.5-35B-A3B
M35B_DRAFT=$PWD/model/Qwen3.5-35B-A3B-DFlash

echo "########## [A] Qwen3.5-35B-A3B dflash (TP=1 单卡, 避开多卡all-reduce bug) ##########"
CUDA_VISIBLE_DEVICES=0 python benchmark_sd_vlm.py \
    --engine vllm --speculative-method dflash \
    --model-path $M35B --draft-model-path $M35B_DRAFT \
    --datasets $DATASETS --num-questions $NUM_Q \
    --tensor-parallel-size 1 \
    --num-speculative-tokens 5 \
    --max-tokens 512 --temperature 0.0 \
    --max-model-len 32768 --gpu-memory-utilization 0.90 \
    --max-num-seqs 64 \
    --port 30032 --output-dir results 2>&1
echo "########## [A] done ##########"

echo "########## [B] eagle3 (aux层=[1,13,24] 已修复) ##########"
CUDA_VISIBLE_DEVICES=0 python benchmark_sd_vlm.py \
    --engine vllm --speculative-method eagle3 \
    --model-path $QWEN25 --draft-model-path $PWD/model/qwen2.5-vl-7b-eagle3-sgl-en-zh \
    --datasets $DATASETS --num-questions $NUM_Q \
    --num-speculative-tokens 5 \
    --max-tokens 512 --temperature 0.0 \
    --max-model-len 32768 --gpu-memory-utilization 0.85 \
    --port 30033 --output-dir results 2>&1
echo "########## [B] done ##########"
echo "RERUN ALL DONE"
