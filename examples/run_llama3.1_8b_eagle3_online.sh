SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels
# train eagle3 for llama3.1-8b
NUM_GPUS=${1:-1}
TP_SIZE=${2:-1}
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-128}

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path /local/mnt/workspace/ziantan/model \
    --draft-model-config $ROOT_DIR/configs/llama3-8B-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/sharegpt_train.jsonl \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $ROOT_DIR/outputs/llama3-8b-eagle3-sharegpt \
    --num-epochs 5 \
    --batch-size 2 \
    --tp-size $TP_SIZE \
    --learning-rate 1e-4 \
    --max-length 1024 \
    --chat-template llama3 \
    --cache-dir $ROOT_DIR/cache \
    --attention-backend sdpa \
    --target-model-backend sglang \
    --log-interval 10 \
    --sglang-mem-fraction-static 0.2 \
    --report-to tensorboard
# max-lenth:输入输出最多2048个token