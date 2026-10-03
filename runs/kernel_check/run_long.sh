#!/bin/bash
# first 20 M tokens of the real 500 M schedule (warmup 763 steps), reference vs kernels, init seed 0
cd /mnt/sandisk/Sparse-Memory-LM/AngryAnt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for impl in torch triton; do
  .venv/bin/python -m smlm.train --model B-1M-sparse --mem_impl $impl --value_lr 2.4e-3 --data wikipedia \
    --tokens 500e6 --stop_after_tokens 20e6 --eval_every_tokens 2e6 --seed 0 --sample_windows 0 \
    --out_dir runs/kernel_check/${impl}_500msched > runs/kernel_check/${impl}_500msched.log 2>&1
  echo "$(date +%T) ${impl}_500msched rc=$?" >> runs/kernel_check/done_long.txt
done
echo "$(date +%T) ALL DONE" >> runs/kernel_check/done_long.txt
