#!/bin/bash
# 35B 视频数据生成 - 6卡6分片并行 (DP=6, 每片单卡装35B)
# 35B-A3B 单卡占 ~65GB, 80GB 卡够用 -> 每卡一个独立进程, 6x 加速。
# 输出到独立 shard 文件, 跑完用 cat 合并。
#
# 用法: nohup bash scripts/gen_video_35b_parallel.sh > logs/gen_video_35b_parallel.log 2>&1 &

cd /prj/corp/crd/morpheus/lasvegas/china-scratch/ziantan/SpecForge

NUM_SHARDS=6
COMMON="--model model/Qwen3.5-35B-A3B \
    --video-dir datasets/train/nextqa/NExTVideo \
    --frames-dir datasets/train/nextqa_frames_35b \
    --max-samples 2000 --per-video 4 \
    --max-new-tokens 512 --temperature 1.0 \
    --device-map cuda:0 --num-shards $NUM_SHARDS"

for i in 0 1 2 3 4 5; do
    CUDA_VISIBLE_DEVICES=$i python scripts/gen_video_dflash_data.py $COMMON \
        --shard-id $i --out datasets/train/qwen3.5-35b-vl-dflash_video_p$i.jsonl \
        > logs/gen_35b_p$i.log 2>&1 &
    echo "shard$i PID $! (GPU $i)"
done

wait
echo "=== 全部 shard 完成, 合并 ==="
cat datasets/train/qwen3.5-35b-vl-dflash_video_p*.jsonl \
    > datasets/train/qwen3.5-35b-vl-dflash_video_new.jsonl
echo "合并条数: $(wc -l < datasets/train/qwen3.5-35b-vl-dflash_video_new.jsonl)"
