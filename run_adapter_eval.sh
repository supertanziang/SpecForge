#!/bin/bash
cd /prj/corp/crd/morpheus/lasvegas/china-scratch/ziantan/SpecForge
TARGET=/prj/corp/crd/morpheus/lasvegas/china-scratch/ziantan/ViSpec/model/Qwen2.5-VL-7B-Instruct
DRAFT=ziantan/outputs/qwen2.5-vl-7b-dflash-stage3-adapter-frozen/epoch_3_step_36000

for ds in sqa textvqa videomme longvideobench; do
  echo "===== eval $ds ====="
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_dflash_spec_video.py \
    --target-model-path $TARGET \
    --draft-path $DRAFT \
    --data-path datasets/eval/${ds}_dflash_video.jsonl \
    --is-vlm --min-pixels 50176 --max-pixels 200704 \
    --max-length 4096 --max-new-tokens 128 --temperature 0.0 \
    --num-samples 30 --modes on,off --trust-remote-code
  echo "===== $ds done ====="
done
echo "ADAPTER EVAL ALL DONE"
