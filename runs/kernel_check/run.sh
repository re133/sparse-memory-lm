#!/bin/bash
# 20 M-token comparison run: PyTorch reference vs Triton kernels, same init seed, same data and schedule
cd /mnt/sandisk/Sparse-Memory-LM/AngryAnt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for impl in torch triton; do
  .venv/bin/python -m smlm.train --model B-1M-sparse --mem_impl $impl --value_lr 2.4e-3 --data wikipedia \
    --tokens 20e6 --eval_every_tokens 2e6 --seed 0 --sample_windows 0 --out_dir runs/kernel_check/$impl \
    > runs/kernel_check/$impl.log 2>&1
  echo "$(date +%T) $impl rc=$?" >> runs/kernel_check/done.txt
done
echo "$(date +%T) ALL DONE" >> runs/kernel_check/done.txt
