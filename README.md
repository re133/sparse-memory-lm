# Sparse-Memory-LM

*A small, fully documented research project, run on one consumer AMD GPU (RX 9070) plus rented cloud GPUs. The full
lab notebook with every pre-registered criterion, all numbers and all mishaps is [`REPORT.md`](REPORT.md) (German);
[`final.md`](final.md) is a German summary.*

How far does a **product-key memory** — a huge table of learned vectors of which only a few rows are read per
token — get a *small* language model? This project measures it on a 21 M-parameter Llama-style model trained on
Wikipedia, from a 262 k-row table up to a 16.8 M-row table (6.4 B parameters), and checks what such a table costs to
train, to run, and to store outside the GPU.

Every experiment had its success criterion written down before it ran; negative and mixed results are reported as
such.

## Results in short

| Question | Answer | Details |
|---|---|---|
| Does a memory layer help a 21 M model at equal compute per token? (WikiText-103) | A little: −2.0 % PPL after 1 epoch (within seed noise), −4.4 % after 3 epochs | REPORT „Stufe 1“ |
| 3 memory layers sharing a 1 M-row table, 500 M fresh Wikipedia tokens | **−15 % val PPL** (25.67 → 21.84) at +4 % multiply-adds per token; stable over 2 seeds | „Stufe 1b/1c“ |
| …at equal *training time* instead of equal tokens (consumer AMD GPU) | only −3.0 % (21.84 vs 22.51): the memory layers make training slower | „Stufe 1c“ |
| Bigger tables (same 500 M tokens, H200) | 4 M rows: −4.75 % vs 1 M; 16.8 M rows: −8.6 % vs 1 M | „Cloud-Läufe“ |
| How big a plain dense model matches them? | B-1M ≈ 60 M, B-4M ≈ 83 M, B-16M ≈ **114 M** non-embedding parameters (dense models of 50–400 M trained on the same data), at 47–57 M instead of ≈ 105–165 M multiply-adds per token. On WikiText-103 the advantage is smaller (≈ 48 / 65 / 95 M) | „Schritt 1“ |
| Must the table live in GPU memory? | No, not for generating text: from RAM 139–154 tok/s, from an NVMe SSD 114–138 tok/s (vs 212 tok/s in VRAM), bit-identical output, 0.5 GB VRAM. Reading long prompts, however, is 2–70× slower outside VRAM | „Schritt 2“ |
| Does it help a real pretrained LM (Qwen3.5-0.8B, frozen, memory as a gated add-on, 1 M rows, trained on 55 M tokens of Wikipedia articles newer than the model)? | **No advantage over a dense add-on of the same compute.** Both cut held-out PPL by ≈ 22 %. The table memorises its training articles far more (PPL −58 % vs −30 %), but recalls only +2.4 pp more exact facts than the dense control, and the same +2.2 pp on unseen articles. It also costs MMLU −2.7 pp and +7 % PPL on differently formatted Wikipedia text. The dense control showed neither side effect | „Schritt 3“ |

![Validation perplexity, stage 1b](report/s1b_val_ppl.png)

![Equivalent dense size](report/dense_equiv.png)

![Table outside the GPU](report/offload_cache.png)

## What is in the box

- **Model** (`smlm/model.py`, `smlm/pkm.py`): Llama-style decoder, d = 384, 12 layers, 6 heads, SwiGLU, RoPE,
  RMSNorm, GPT-2 BPE. The memory layers replace the FFN of layers 3, 7 and 11.
- **Memory layers** (following Lample et al. 2019 and Meta's *Memory Layers at Scale*, Berges et al. 2024):
  - Product keys: 4 heads, 2 × n sub-keys per head, exact top-32 out of n² keys.
  - BatchNorm on the query, softmax over the top-32.
  - Memory+ "swilu" output path.
  - One value table shared by all memory layers.
- **Training the table** (`smlm/sparse_values.py`): row-sparse gradients and a lazy Adam that only touches the
  rows read in a step. Same quality as dense Adam on the table, 1.15× faster, much less memory for big tables.
- **Triton kernels** (`smlm/kernels.py`; ROCm and CUDA, the PyTorch path stays as reference and fallback):
  - Kernels:
    - Segmented-scan row gradients.
    - Fused product-key top-k selection on order-preserving integer keys.
    - Value bag from fp32 / bf16 / 4-bit tables with the swilu product fused in.
    - Fused lazy Adam.
  - Decode-step graph capture.
  - Effect on an RX 9070: training 1.47× faster (0.62× the speed of the dense baseline), prefill at 1.43× of the
    baseline, decoding equal to the baseline.
- **Table outside the GPU** (`smlm/offload.py`):
  - Table in RAM, or as a 4-bit file on an NVMe drive (mmap + RAM cache of the most-read rows).
  - The rows a step needs are gathered and the same kernel runs on them, so the output is bit-identical to the
    on-GPU table.
- **Cloud runs** (`cloud/`, `scripts/run_cloud*.py`, `scripts/run_dense.py`):
  - One-command setup on a Runpod GPU.
  - Data byte-identical to the local machine, checked by sha256.
  - A test gate and a preflight before paid hours.
  - A budget gate and watchdog.
  - Checkpoints copied to external storage and verified.
- **Pretrained add-on** (`smlm/qwen_memory.py`): memory blocks hooked behind layers of a frozen Hugging Face
  model, gated with a scalar that starts at 0. The test checks that the model is then bit-identical to the
  original.

## Reproduce

```bash
python -m venv --system-site-packages .venv && .venv/bin/pip install -r requirements.txt   # torch with ROCm or CUDA
.venv/bin/python cloud/fetch_data.py          # WikiText-103 + Wikipedia 20231101.en at pinned revisions, sha256-checked
.venv/bin/python -m pytest -q tests           # CPU-only: TRITON_INTERPRET=1 for the kernel tests
# baseline A and B-1M on 500 M Wikipedia tokens (stage 1b/1c)
.venv/bin/python -m smlm.train --model A --out_dir runs/A --data wikipedia --tokens 500e6 --extra_val wikitext103
.venv/bin/python -m smlm.train --model B-1M-sparse --mem_impl triton --value_lr 2.4e-3 --out_dir runs/B-1M \
    --data wikipedia --tokens 500e6 --extra_val wikitext103
```

- **Bigger tables:** B-4M-sparse needs ≈ 29 GB of GPU memory for training, B-16M-sparse ≈ 101 GB. The cloud
  queue for them is described in `CLOUD.md`.
- **Inference outside the GPU:** `scripts/convert_table.py` (checkpoint → bf16 / 4-bit table files) and
  `scripts/bench_offload.py --variant a-bf16|a-q4|b-bf16|b-q4|b-fp32|c` (table in VRAM, in RAM, or as an mmap'ed
  4-bit file); default table location `data/tables/B-16M` or `SMLM_TABLES`.
- **Qwen add-on (step 3):** separate environment with `pip install -r requirements-qwen.txt`. Then:
  - `scripts/prepare_qwen_data.py`: newest Wikipedia articles by creation month (set `SMLM_CONTACT` for the
    Wikimedia API).
  - `scripts/make_fact_cloze.py`: fact test.
  - `scripts/train_qwen_memory.py --kind memory|dense|none`.
  - `scripts/eval_fact_cloze.py`, `scripts/eval_qwen_general.py`, `scripts/qwen_step3_eval.py`.
  - The model is expected in `models/Qwen3.5-0.8B` (or `QWEN_DIR`), the data in `data/qwen_wiki` (or `QWEN_DATA`).
  - The cloud queue for it is `cloud/setup_qwen.sh` + `scripts/run_qwen.py`.

## Honest limits

- **Small scale:** a 21 M model trained on at most 500 M tokens (1.5 B for one equal-time run). Nothing here
  says how the effect behaves at 1 B+ parameters.
- **Seeds:** one seed for B-4M, B-16M and the dense comparison models; the seed noise measured with two seeds
  elsewhere is ≈ 0.4 %.
- **Tokens vs. compute:**
  - At equal tokens the table is a large win.
  - At equal wall-clock time on our hardware it is a small one.
  - The big tables need far more training memory: B-16M took ≈ 101 GB.
- **Data and evaluation:** Wikipedia perplexity only. There are no downstream-task results for the memory models.
- **Offloading:** the RAM / NVMe paths are Python/NumPy. Long-prompt throughput from NVMe is limited by random 4 KB
  reads per 192-byte row; smarter layouts or I/O paths are not explored.
- **Hyperparameters:** taken from the baseline. The learning rate was not tuned per model size.

## Notes for AMD / ROCm users

Everything here runs on ROCm (RX 9070, gfx1201, ROCm 7.2, Triton 3.5) and on CUDA (H100 / H200) with identical
results. Two things mattered on the consumer AMD GPU:
- **Atomics:** fp32 `tl.atomic_add` compiles to the native `global_atomic_add_f32`, but is ≈ 8× slower than plain stores
  (`docs/rocm-issues/repro_atomic_add.py`). Kernels that accumulate rows therefore sort by row and use plain
  read-add-write for rows owned by one program, with atomics only at program borders (17 ms → 2.7 ms).
- **Batch-1 decoding is launch-bound:** ≈ 0.4 ms of CPU time per memory layer. Replaying a HIP graph of the memory
  layer, not faster kernels, fixed it.

Not ROCm issues, for the record:
- An early optimizer kernel corrupted a byte mask; minimal reproducers (`docs/rocm-issues/repro_byte_store.py`) do not
  show any byte-store problem, so this was most likely a bug in our own kernel.
- Qwen3.5 (Gated DeltaNet, PyTorch fallback) crashed twice under ROCm in this setup (lm-eval batches, gradient
  checkpointing); these were not reduced to a minimal case.

## Related work

- Lample et al. 2019 introduced product-key memory layers.
- Berges et al. 2024 (*Memory Layers at Scale*) showed that memory layers scaled to billions of parameters beat
  dense models at equal compute, with a shared table and the Memory+ ("swilu") output path that this project
  follows.
- This repository is a small-scale, open and reproducible counterpart of those results. It adds:
  - an equivalence curve against dense models trained on the same data;
  - measurements of running the table from RAM or an NVMe SSD on consumer hardware;
  - Triton kernels for ROCm and CUDA;
  - a controlled negative result for retrofitting a memory table onto a frozen pretrained model, against a
    dense add-on of equal compute and with a fact-recall test.

## Data and licenses

- **Code:** Apache License 2.0 ([`LICENSE`](LICENSE)).
- **Data, not redistributed (downloaded and tokenised by the scripts):** WikiText-103 (Salesforce, CC BY-SA),
  English Wikipedia (wikimedia/wikipedia 20231101.en and the enwiki dump of 2026-09-01, CC BY-SA 4.0).
- **[`data/qwen_fact_cloze.jsonl`](data/qwen_fact_cloze.jsonl):** contains short excerpts of English Wikipedia
  articles (titles and page ids included for attribution). Licensed CC BY-SA 4.0, © Wikipedia contributors.
- **Qwen3.5-0.8B:** Apache License 2.0 (Qwen Team), used unmodified and not redistributed.
- **Checkpoints:** not in the repository.

## References

- G. Lample et al., *Large Memory Layers with Product Keys*, NeurIPS 2019.
- V.-P. Berges et al., *Memory Layers at Scale*, 2024 (and the `facebookresearch/memory` / lingua reference code).
- Qwen Team, *Qwen3.5* (Qwen3.5-0.8B, Apache 2.0), 2026.

