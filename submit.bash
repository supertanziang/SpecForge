NAME=$2

GPU_TYPE=NVIDIAA100_SXM4_80GB
NUM_GPU=8
if [ $# == 3 ]; then
    NUM_GPU=$3
fi

CONTAINER=/prj/corp/crd/morpheus/lasvegas/china-scratch/bsub/containerize_job.sh
CONF_FILE=/prj/corp/crd/morpheus/lasvegas/china-scratch/yzao/eagle/eagle.conf
CONF_NAME=eaglet

CMD="bash ./examples/run_qwen2.5_7b_vl_eagle3_online.sh $NUM_GPU"
echo $CMD

OUTPUT_LOGROOT=/prj/corp/crd/morpheus/lasvegas/china-scratch/ziantan/SpecForge
current_time=$(date +"%Y-%m-%d-%H-%M-%S")
OUTPUT_LOG=$OUTPUT_LOGROOT/${NAME}_${current_time}.txt

if [ $1 == 078 ]; then
    bsub -P 17308.00.ai.morpheus -U morph -m morph-lsf78-gpulv -J $NAME -o $OUTPUT_LOG -q normal -R "select[gpu]" -gpu "num=${NUM_GPU}:gmodel=${GPU_TYPE}" $CMD
elif [ $1 == 070 ]; then
    bsub -P 17308.00.ai.morpheus -m morph-lsf70-gpulv -J $NAME -o $OUTPUT_LOG -q normal -R "select[gpu]" -gpu "num=${NUM_GPU}:gmodel=${GPU_TYPE}" $CMD
elif [ $1 == 067 ]; then
    bsub -P 17308.00.ai.morpheus -m morph-lsf67-gpulv -J $NAME -o $OUTPUT_LOG -q normal -R "select[gpu]" -gpu "num=${NUM_GPU}:gmodel=${GPU_TYPE}" $CMD
elif [ $1 == 059 ]; then
    bsub -P 17308.00.ai.morpheus -m morph-lsf59-gpulv -J $NAME -o $OUTPUT_LOG -q normal -R "select[gpu]" -gpu "num=${NUM_GPU}:gmodel=${GPU_TYPE}" $CMD
elif [ $1 == 052 ]; then
    bsub -P 17308.00.ai.morpheus -m morph-lsf52-gpulv -J $NAME -o $OUTPUT_LOG -q normal -R "select[gpu]" -gpu "num=${NUM_GPU}:gmodel=${GPU_TYPE}" $CMD
elif [ $1 == 78 ]; then
    bsub -P 17308.00.ai.morpheus -U morph -m morph-lsf78-gpulv -J $NAME -o $OUTPUT_LOG -q normal -R "select[gpu]" -gpu "num=${NUM_GPU}:gmodel=${GPU_TYPE}" $CONTAINER -f $CONF_FILE -a $CONF_NAME -c "$CMD"
elif [ $1 == 70 ]; then
    bsub -P 17308.00.ai.morpheus -m morph-lsf70-gpulv -J $NAME -o $OUTPUT_LOG -q normal -R "select[gpu]" -gpu "num=${NUM_GPU}:gmodel=${GPU_TYPE}" $CONTAINER -f $CONF_FILE -a $CONF_NAME -c "$CMD"
elif [ $1 == 67 ]; then
    bsub -P 17308.00.ai.morpheus -m morph-lsf67-gpulv -J $NAME -o $OUTPUT_LOG -q normal -R "select[gpu]" -gpu "num=${NUM_GPU}:gmodel=${GPU_TYPE}" $CONTAINER -f $CONF_FILE -a $CONF_NAME -c "$CMD"
elif [ $1 == 59 ]; then
    bsub -P 17308.00.ai.morpheus -m morph-lsf59-gpulv -J $NAME -o $OUTPUT_LOG -q normal -R "select[gpu]" -gpu "num=${NUM_GPU}:gmodel=${GPU_TYPE}" $CONTAINER -f $CONF_FILE -a $CONF_NAME -c "$CMD"
elif [ $1 == 52 ]; then
    bsub -P 17308.00.ai.morpheus -m morph-lsf52-gpulv -J $NAME -o $OUTPUT_LOG -q normal -R "select[gpu]" -gpu "num=${NUM_GPU}:gmodel=${GPU_TYPE}" $CONTAINER -f $CONF_FILE -a $CONF_NAME -c "$CMD"
fi
