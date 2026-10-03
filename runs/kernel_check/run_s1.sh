#!/bin/bash
# second pair (init seed 1): seed noise of the reference and a second kernel-vs-reference pair
cd /mnt/sandisk/Sparse-Memory-LM/AngryAnt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for impl in torch triton; do
  .venv/bin/python -m smlm.train --model B-1M-sparse --mem_impl $impl --value_lr 2.4e-3 --data wikipedia \
    --tokens 20e6 --eval_every_tokens 2e6 --seed 1 --sample_windows 0 --out_dir runs/kernel_check/${impl}_s1 \
    > runs/kernel_check/${impl}_s1.log 2>&1
  echo "$(date +%T) ${impl}_s1 rc=$?" >> runs/kernel_check/done_s1.txt
done
echo "$(date +%T) ALL DONE" >> runs/kernel_check/done_s1.txt
