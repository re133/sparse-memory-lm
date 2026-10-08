# Stage 1: Does a product-key memory layer improve a small language model at equal compute?

*This is the English version of the lab notebook. The German original is [REPORT.de.md](REPORT.de.md); where the
two differ, the German one is the record that was written while the work happened. Quotes of the original German
specifications are kept in German with a translation.*

> **Summary.** Yes, but only a little. At equal compute per token, the memory layer (B) lowers the validation
> perplexity against the baseline (A) by 2.0% after 1 epoch and by 4.4% after 3 epochs.
> By the criteria **fixed before the runs** this means: **1 epoch "not worth it"** (difference within seed noise),
> **3 epochs "unclear"** (the difference is real, but B closes only 20% of the gap to the equally large dense model C,
> 50% was required). The table is healthy (≈ 100% usage) and heavily used; no implementation bug was found. What
> stands out is an almost flat weighting within the top 32. The branch `v2-sharpness` was prepared for that (it ran
> later: v2b gets sharper but gains practically nothing, see `docs/notes/V2_SHARPNESS.md`; corrected 2026-10-05 after
> the Codex review).
> Cost of B despite equal FLOPs: −21% training throughput, −31% prefill throughput, +1.6 GiB VRAM.
>
> **Addendum stages 1b/1c (2026-10-03):** With a 1M table, 3 memory layers and fresh Wikipedia data, B-1M is a
> steady 15% ahead of A at equal tokens (2 seeds per model). A sparse optimizer gives the same quality and is 1.15×
> faster. At **equal training time** the lead shrinks to 3% ("competitive", the "clear advantage" of 5% was
> missed). Details: sections stage 1b and 1c.
>
> **Addendum cloud (2026-10-04, Runpod H200):** Bigger tables bring clearly more at equal tokens.
> B-4M is 4.75% better than B-1M, B-16M another 4.0% (−8.6% against B-1M). The pre-registered criterion "worth it"
> is met. B-1M in the cloud matches the value from home exactly (21.837), so the Triton kernels change nothing. One
> seed per size, cost 25.51 $ (first reported too low as 21 $).
>
> **Addendum steps 1–3 (2026-10-05):**
> - **Step 1 (H100):** Dense comparison models show that B-1M, B-4M and B-16M are as good as dense models with
>   ≈ 60, 83 and 114 M non-embedding parameters, at a third to half of the compute per token.
> - **Step 2 (at home):** B-16M writes with the table in RAM (139–154 tok/s) or on the NVMe (114–138 tok/s)
>   bit-identically to the table in VRAM (212 tok/s). The logits were compared one by one
>   (`report/offload/identical_check.json`); that holds for Triton 3.5, with Triton 3.8 they differ in the last bits
>   (section "Codex review"). Reading long texts outside VRAM, however, is 2 to 70 times slower.
> - **Step 3 (Qwen3.5-0.8B with a table as an add-on):**
>   - Helps only as much as an equally expensive dense add-on block.
>   - Does harm by the criteria: MMLU −2.7 pp, +7% PPL on differently formatted text.
>   - Doesn't plant retrievable knowledge (+2.4 pp fact hits against the control, 10 were required).
> - **Cost:** step 1 ≈ 17 $, step 3 ≈ 22.50 $, including ≈ 7.40 $ of idle time caused by my own mistake (see
>   step 3). The amounts are provisional because Runpod's billing lags behind.

The success criteria were fixed before any run started (commit `bc13f1b`, made more precise in `f79e240` before the
first result).

## Success criteria (fixed before the runs)

Specification (quoted verbatim, German original):

- **lohnt sich** (worth it): Key-Nutzung deutlich über 50 % **und** B schließt mindestens die halbe
  Perplexity-Lücke zwischen A und C (key usage clearly above 50% **and** B closes at least half the perplexity gap
  between A and C)
- **unklar** (unclear): B besser als A, aber knapp → erst Implementierung und Tabellengesundheit prüfen (B better
  than A, but narrowly → check the implementation and table health first)
- **lohnt sich nicht** (not worth it): B trotz gesunder Tabelle auf A-Niveau (B at A's level despite a healthy table)

Operationalisation (this is how it's evaluated, separately for the 1-epoch and the 3-epoch run):

| Quantity | Definition |
|---|---|
| PPL | Token perplexity (GPT-2 BPE) on the **whole** WikiText-103 validation set at the end of training, non-overlapping windows of 1024, model in eval mode. A and B: mean over 2 init seeds, C: 1 seed. |
| Gap closure G | G = (PPL_A − PPL_B) / (PPL_A − PPL_C). "At least half the gap" means G ≥ 0.5. |
| Key usage | Share of the 262,144 entries that are read at least once on the validation set (≈ 247 k tokens, ≈ 31.7 M reads) (definition of "memory usage" after Lample et al. 2019). I set "clearly above 50%" as **≥ 60% for both B seeds** (my interpretation). |
| Table health | Usage as above, plus KL(access weights ‖ uniform) and the share of reads that fall on the most-read 1% of entries. A table with ≥ 60% usage where > 50% of all reads fall on 1% of the entries does **not** count as healthy. |
| Random noise | Seed spread s = max(\|PPL_A,s0 − PPL_A,s1\|, \|PPL_B,s0 − PPL_B,s1\|). A difference A↔B only counts as real if \|PPL_A − PPL_B\| > 2·s. |
| "narrowly" (→ unclear) | B better than A by more than 2·s, but G < 0.5. |
| "at A's level" (→ not worth it) | \|PPL_A − PPL_B\| ≤ 2·s, or B worse than A. |

Cases the specification doesn't cover: if the table is **not** healthy (usage < 60% or strong concentration), the
result is **inconclusive**, regardless of the perplexity. Then the memory layer gets fixed first (query
normalisation, learning rates) before a verdict.

## Setup

**Data.** WikiText-103 (raw, HF `Salesforce/wikitext`), GPT-2 BPE via `tiktoken`: 117.98 M training, 247 k
validation, 283 k test tokens. Training on non-overlapping windows of 1025 whose order per epoch depends only on the
data seed (1234). So all models see exactly the same token stream in the same order. 1 epoch = 3600 steps of
32 × 1024 = 32,768 tokens.

**Models** (Llama style: RMSNorm, RoPE, SwiGLU, no biases, tied input/output embedding, context 1024):

| | Architecture | Params without emb. | Params total | active/token without emb. | MACs/token (forward) |
|---|---|---|---|---|---|
| A | d=384, 12 layers, 6 heads, SwiGLU 1024 | 21.24 M | 40.56 M | 21.24 M | 45.27 M |
| B | like A, FFN of layer 7 (index 6) → PKM | 121.94 M | 141.26 M | 21.33 M | 45.35 M |
| C | d=768, 16 layers, 12 heads, SwiGLU 2304 | 122.71 M | 161.34 M | 122.71 M | 173.90 M |

C is matched to B's parameters **without embeddings** (the way A is specified "without embeddings").
"Active/token" for B = all dense parameters except the value table (including all sub-keys, which are scanned
completely) + the 4 × 32 × 384 values actually read. MACs include the LM head and attention (mean causal context
512).

**Memory layer (B).** 512² = 262,144 entries, 4 heads, top-k 32, key dim 256 (2 × 128), value dim 384, shared value
table (100.7 M parameters). Built after Meta's reference implementation (`facebookresearch/memory`,
`lingua/product_key/memory.py`): own sub-keys per head, exact top-k over the product (top-k per half, then k×k
candidates), softmax over the k scores per head, heads summed, swilu output `W2(m(x) ⊙ silu(W1 x))` ("Memory+"),
initialisation as there. In addition **BatchNorm on the query** (Lample et al. 2019, sect. 4.5: raises usage at 1 M
entries from 25.8% to 80.3%; PEER uses it too). In training BatchNorm sees statistics over the batch (so also later
tokens); all evaluations run in eval mode with fixed statistics, and a unit test checks causality there.
Compute of the replacement: removed FFN 1.18 M MACs/token, memory layer 1.26 M (query 0.39 M, sub-key scores
0.52 M, reading values 0.05 M, swilu 0.30 M).

**Optimisation** (identical for A/B/C): AdamW (β = 0.9/0.95, ε = 1e-8, weight decay 0.1 on matrices), peak LR 6e-4,
linear warmup over 5% of the steps, cosine to 10%, gradient clipping 1.0, bf16 autocast with fp32 weights.
**Exception: memory values** (following the papers): LR 1e-3 absolute (Lample et al.: "higher Adam learning rate of
10⁻³" for the sparsely updated values; Meta: `value_fixed_lr=0.001`), same schedule multiplier, no weight decay, own
clipping (as Meta's `train.py`). So the ratio values/rest is 1.7× instead of 4× in Lample. Micro-batch 8 (A, B) or
4 (C) with gradient accumulation to 32 sequences.

**Seeds.** Data seed 1234 for all. Init seeds 0 and 1 for A and B, 0 for C. GPU kernels (atomics in the
`embedding_bag` backward, flash attention backward) are not bit-exactly deterministic.

**Measurements.** Validation PPL every 4 M (1 epoch) or 8 M tokens (3 epochs) over the whole val set, plus the mean
training loss over the same interval. Tokens/s in training without evaluation time (median over 10-step windows).
Inference: (a) batch-1 decoding with KV cache, 128 prompt + 256 new tokens, greedy; (b) batched forward 16 × 1024
("prefill"). Peak VRAM = `torch.cuda.max_memory_allocated` during training (without evaluation). For B: key usage,
KL and concentration on the val set at every evaluation, usage in training per interval, access histogram over the
whole training and over the val set, and an index sample (the first 65,536 val tokens in text order: indices
[token, head, k] as int32, raw scores, token ids) in `runs/*/B-*/mem_index_sample.npz`.

**Unit tests** (`tests/test_pkm.py`, CPU and GPU, fp64): product-key top-k = brute-force top-k over all n² keys
(scores and index sets exact); the gradient of the value table is ≠ 0 exactly on the rows that were read (both
implementations); both value-read implementations agree in output and gradients; causality in eval mode (with and
without memory); KV-cache decoding = full forward.

## Results

### Trial run: 20 M tokens, 1 seed each (pipeline check, not for the verdict)

| Run | Val PPL | Test PPL | Val PPL (word) | Train tok/s | Decode b=1 tok/s | Prefill tok/s | VRAM train | Key usage val | Top-1% share | KL |
|---|---|---|---|---|---|---|---|---|---|---|
| A-s0 | 138.92 | 140.84 | 272.8 | 91,542 | 213 | 383,125 | 7.89 GiB | – | – | – |
| B-s0 | 135.68 | 137.64 | 265.6 | 73,043 | 197 | 269,315 | 9.47 GiB | 81.8% | 33.0% | 2.03 |
| C-s0 | 107.75 | 110.12 | 204.4 | 34,441 | 165 | 131,093 | 7.91 GiB | – | – | – |

![Val PPL trial run](report/probe_val_ppl.png)

- B is 2.3% below A (−3.2 PPL), C 22% below. Gap closure G = 0.10. With only one seed the noise can't be measured;
  the trial run was only there to check the pipeline.
- **Key usage collapses at the start and then recovers.** After initialisation 99% of the entries are read, after
  2 M tokens only 9% (85% of all reads on 1% of the entries), after that usage rises steadily to 82% at 20 M tokens
  (top-1% share 33%, still falling). The key norms stay even (max/min ≈ 1.3) and almost all 512 sub-keys per half
  get read. So the collapse doesn't come from "hub keys" with a large norm, but from queries that are not very
  diverse early in training (residual stream still low-rank), so only a few combinations of the two halves win.
  For stage 3 this means: the access distribution depends strongly on how far training is, and measurements on
  models stopped early are not representative.

![Table health trial run](report/probe_memory_health.png)

- Git: A-s0 ran on `bc13f1b`, B-s0 and C-s0 on `fef54eb`. In between only `REPORT.md` and the evaluation script
  changed, not the training code. The `dirty: true` in their `run-info.json` comes only from the folder `runs/`,
  which wasn't versioned at the time (fixed afterwards: outputs no longer count).

### 1 epoch: 118 M tokens (A, B 2 seeds each; C 1 seed)

| Run | Val PPL | Test PPL | Val PPL (word) | Train tok/s | Decode b=1 tok/s | Prefill tok/s | VRAM train | Train time | Key usage val | Top-1% share | KL |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A-s0 | 35.26 | 35.24 | 57.4 | 91,705 | 217 | 383,407 | 7.89 GiB | 21 min | – | – | – |
| A-s1 | 34.93 | 34.77 | 56.8 | 91,640 | 218 | 384,460 | 7.89 GiB | 21 min | – | – | – |
| B-s0 | 34.63 | 34.56 | 56.2 | 72,445 | 199 | 264,882 | 9.47 GiB | 27 min | 99.2% | 18.4% | 1.08 |
| B-s1 | 34.12 | 34.05 | 55.3 | 72,241 | 203 | 264,573 | 9.47 GiB | 27 min | 99.7% | 15.5% | 0.91 |
| C-s0 | 25.80 | 26.02 | 40.3 | 34,468 | 165 | 130,976 | 7.91 GiB | 57 min | – | – | – |

**Evaluation against the pre-registered criteria:**

| Quantity | Value |
|---|---|
| PPL A (mean; seeds) | 35.09 (35.26 / 34.93) |
| PPL B (mean; seeds) | 34.38 (34.63 / 34.12) |
| PPL C | 25.80 |
| Seed spread s | 0.51 (from B; A: 0.32) → threshold 2·s = 1.02 |
| PPL_A − PPL_B | +0.72 (**within** 2·s) |
| Gap closure G | **0.08** |
| Key usage (min.) / top-1% share (max.) | 99.2% / 18.4% → **table healthy** |
| **Verdict** | **not worth it** (B at A's level, despite a healthy table) |

Note on the classification: the threshold 2·s and the mapping "B better, but within 2·s → at A's level" are **my**
operationalisation of the specification. Going by the wording ("B better than A, but narrowly"), **"unclear"** would
also be defensible, since B is better in all four pairings. The checks required for "unclear" (implementation,
table health) are done below anyway.

![Val PPL 1 epoch](report/ep1_val_ppl.png)

![Distance to A, 1 epoch](report/ep1_relative_to_A.png)

**What the numbers say, without sugar-coating:**

- B is better than A in all four A/B pairings, evenly by ≈ 2% from about 30 M tokens on (val and test alike). That
  points to a real but small effect. By the pre-registered criterion it isn't enough: 0.72 PPL is below
  2·s = 1.02, and with two seeds per model "B beats A in all pairings" isn't solid either (by chance it would
  happen in 1 of 6 cases).
- **Even if the effect is real, it's far too small.** B closes 8% of the gap to C, 50% was required. The lead
  doesn't grow over the epoch either, it stays at ≈ 2% (see plot). So B's 100 M extra parameters bring nowhere near
  what the same number of dense parameters brings in C (−26.5% PPL), though C needs 3.8× as many MACs per token.
- **C is clearly undertrained at 1 epoch** (≈ 1 token per parameter instead of ≈ 20). With a fully trained C the gap
  A↔C would be even bigger; the comparison is, if anything, favourable for B.
- **Cost of B:** same FLOPs as A, but 21% less training throughput, 31% less prefill throughput, 7% slower batch-1
  decoding and 1.6 GiB more VRAM (value table with gradient and Adam states: 100.7 M × 16 bytes).

**Table health and implementation** (mandatory check before a verdict counts):

![Table health 1 epoch](report/ep1_memory_health.png)

- Usage 99.2 / 99.7%, KL 1.08 / 0.91, the most-read 1% of the entries get 18 / 16% of the reads, only 0.3–0.8% of
  the entries are never read on the val set. In training every entry was read (the rarest ≈ 2,400×). The collapse
  at the start (see trial run) recovers completely after ≈ 20 M tokens.
- **The memory layer really contributes something** (`scripts/diagnose_memory.py`, first 32,768 val tokens,
  `runs/ep1/B-*/diagnostics.json`): setting its output to zero raises the PPL from 31.3 to 35.5 (s0) and from 30.6
  to 35.7 (s1). Reading random entries instead of the ones found raises it to 36.0 and 36.5. So it matters *which*
  entries the search finds. The layer's output norm (2.5–2.8) is above that of the neighbouring dense FFNs
  (1.2–2.6). The 4 heads almost always read 128 different entries per token.
- The unit tests (exact top-k, gradient only into values that were read, causality, KV cache) are green. I didn't
  find an implementation bug.
- **Something stands out: the weighting within the top 32 is almost flat.** Effectively each head mixes 30.5 of 32
  entries (exp(entropy)), the top-1 weight is 0.066 (uniform: 0.031). The score scale has barely grown since
  initialisation (key norm 0.41 → 0.48, BatchNorm γ ≈ 1.09). So the layer doesn't retrieve a few entries in a
  targeted way, it roughly averages over 128. That fits the finding "replaces an FFN and a bit more, but not much
  more": without the memory the model is missing a whole layer (+14–16% PPL), with it it's only 2% better than A.

**Access pattern (for stage 3, `report/ep1_access_stats.json`):**

![Access distribution 1 epoch](report/ep1_access_distribution.png)

| | B-s0 | B-s1 |
|---|---|---|
| Share of reads on the top 1 / 10 / 20 / 50% (val) | 18 / 55 / 71 / 91% | 16 / 50 / 66 / 89% |
| Val reads covered by the top 20% from **training** | 68% | 63% |
| Rank correlation of read counts training ↔ val (Spearman) | 0.89 | 0.89 |
| Reads already read within the last 1 / 16 / 256 tokens | 0.7 / 11.7 / 52.5% | 0.9 / 11.5 / 50.2% |

The distribution is clearly skewed, but not an extreme Zipf: a cache of the 20% most-read entries (determined in
training; here 52 k entries × 384 × 2 bytes ≈ 40 MB in bf16) catches ≈ 65% of the reads on the val set. Consecutive
tokens read almost disjoint entries (< 1% repetition to the direct predecessor); over a window of 256 tokens,
however, half of them repeat. For offloading to an SSD this means: ≈ 35% of the 128 reads per token would still have
to come randomly from the SSD even with a hot-set cache.

![Train vs. val loss 1 epoch](report/ep1_train_vs_val.png)

At 1 epoch there is, as expected, no memorisation (every window is seen exactly once); train being above val is the
lagging interval mean, see limitations.

**Inference VRAM** (fp32 weights under bf16 autocast, peak `max_memory_allocated`; also applies to 3 epochs):
batch-1 decoding A 1.16 GiB, B 1.54 GiB, C 1.46 GiB; prefill 16 × 1024 A 2.75 GiB, B 3.13 GiB, C 3.11 GiB.
B's value table alone takes 0.38 GiB (fp32), in bf16 it would be 0.19 GiB.

**Provenance:** A-s0 ran on `4f0e5fb`, all other 1-epoch runs on `6859f1f`. Between these commits only the report
changed, not `smlm/` or `scripts/run_suite.py`. B-s1 is marked `dirty`, only because of the analysis script
`scripts/diagnose_memory.py`, which wasn't versioned at the time and isn't used in training.

### 3 epochs: 354 M tokens, own cosine schedule (A, B 2 seeds each; C 1 seed)

A separate run with warmup (540 steps) and cosine over all 10,800 steps; the intermediate values at 118 M tokens are
therefore **not** comparable with the end of the 1-epoch run (there the LR had already decayed). Each epoch has its
own order, fixed by the data seed.

| Run | Val PPL | Test PPL | Val PPL (word) | Train tok/s | Decode b=1 tok/s | Prefill tok/s | VRAM train | Train time | Key usage val | Top-1% share | KL |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A-s0 | 24.81 | 24.99 | 38.5 | 91,659 | 219 | 383,220 | 7.89 GiB | 64 min | – | – | – |
| A-s1 | 24.65 | 24.84 | 38.2 | 91,478 | 217 | 384,725 | 7.89 GiB | 64 min | – | – | – |
| B-s0 | 23.80 | 24.02 | 36.7 | 71,935 | 203 | 264,825 | 9.47 GiB | 82 min | 99.97% | 11.0% | 0.63 |
| B-s1 | 23.47 | 23.76 | 36.1 | 71,640 | 206 | 265,278 | 9.47 GiB | 82 min | 99.99% | 11.0% | 0.59 |
| C-s0 | 19.15 | 19.51 | 28.7 | 34,607 | 164 | 131,352 | 7.91 GiB | 170 min | – | – | – |

**Evaluation against the pre-registered criteria:**

| Quantity | Value |
|---|---|
| PPL A (mean; seeds) | 24.73 (24.81 / 24.65) |
| PPL B (mean; seeds) | 23.63 (23.80 / 23.47) |
| PPL C | 19.15 |
| Seed spread s | 0.33 (from B; A: 0.16) → threshold 2·s = 0.67 |
| PPL_A − PPL_B | +1.10 (**above** 2·s) |
| Gap closure G | **0.20** |
| Key usage (min.) / top-1% share (max.) | 99.97% / 11.0% → **table healthy** |
| **Verdict** | **unclear** (B really better than A, but narrowly: G < 0.5) |

The test set gives the same picture (A 24.92, B 23.89, C 19.51 → −4.1%, G = 0.19).

![Val PPL 3 epochs](report/ep3_val_ppl.png)

![Distance to A, 3 epochs](report/ep3_relative_to_A.png)

**What the numbers say:**

- **With more data B's lead grows, but stays small.** Relative to A: −0.8% at 40 M tokens, −2.0% at 80 M, −2.9% at
  120 M, −4.0% at 176 M, then constant at −4.4% from ≈ 240 M tokens on (both seeds, see plot). The gap closure G
  rises accordingly to ≈ 0.20 and stays there. Both B seeds are now clearly below both A seeds.
- **More than half is missing for "worth it".** Even after 3 epochs C is 22.6% better than A (at 3.8× the
  MACs/token and still undertrained: ≈ 3 tokens per parameter). B gets a fifth of that.
- **Memorisation** (val loss minus train loss at the end, nats): A 0.014 / 0.015, B 0.035 / 0.043, C 0.119. In
  epochs 2 and 3 the training loss visibly drops at the epoch boundaries. B memorises a bit more than A, much less
  than C; B's val loss still keeps falling until the end.

![Train vs. val loss 3 epochs](report/ep3_train_vs_val.png)

**Table health and implementation** (mandatory check for "unclear", `runs/ep3/B-*/diagnostics.json`, first 32,768
val tokens, GPU):

- Usage 99.97 / 99.99%, KL 0.63 / 0.59, the most-read 1% get 11% of the reads; in training every entry was read
  (the rarest 961× and 1,632×, median ≈ 100 k×). Healthier than after 1 epoch.
- **After 3 epochs the model relies much more on the memory:** output set to zero → PPL 21.8 → 31.3 (s0) and
  21.4 → 32.7 (s1), so +44 / +53% (after 1 epoch: +14 / +16%). Random instead of found entries → 34.3 and 37.3.
  Without the memory B is clearly worse than A. The layer's output norm (7.4 / 8.3) is the largest of all FFN
  positions in the middle (neighbours 2.8–5.2).
- **The weighting within the top 32 gets a bit sharper with training, but stays flat:** effectively 28.6 / 28.3 of
  32 entries per head (after 1 epoch 30.5), top-1 weight 0.094 / 0.097. Key norm 0.41 → 0.59 / 0.61, BatchNorm γ
  1.0 → 1.33 / 1.37. So the scale grows, but slowly.
- Unit tests green; no implementation bug found. The table works and is used; the open question is why it brings so
  little beyond an extra dense layer (see discussion).

![Table health 3 epochs](report/ep3_memory_health.png)

**Access pattern (for stage 3, `report/ep3_access_stats.json`):**

| | B-s0 | B-s1 |
|---|---|---|
| Share of reads on the top 1 / 10 / 20 / 50% (val) | 11 / 40 / 56 / 83% | 11 / 39 / 55 / 82% |
| Val reads covered by the top 20% from **training** | 54% | 52% |
| Rank correlation of read counts training ↔ val (Spearman) | 0.90 | 0.90 |
| Reads already read within the last 1 / 16 / 256 tokens | 0.9 / 9.4 / 44.8% | 0.7 / 8.9 / 42.8% |

With longer training the reads spread out **more evenly**: a hot-set cache of 20% of the entries catches only ≈ 53%
of the reads after 3 epochs (after 1 epoch ≈ 65%). For SSD offloading that's bad news: the better the table is used,
the less caching helps. The ranking of "hot" entries is stable between training and val, though (Spearman 0.90).

![Access distribution 3 epochs](report/ep3_access_distribution.png)

**Provenance:** A-s0 ran on `6859f1f` (marked `dirty`, only because of the unversioned `scripts/diagnose_memory.py`),
all other 3-epoch runs on `e44a197`. Training code identical to all 1-epoch runs
(`git diff a27c99d 1a43e0d -- smlm/ scripts/run_suite.py` is empty).

## Discussion

**Answer to this stage's question.** At equal compute per token, a product-key memory layer improves the small model
**measurably, but only a little**: −2.0% PPL after 118 M tokens (can't be separated from seed noise), −4.4% after
354 M tokens (real). By the pre-registered criteria that's not enough for "worth it" in either run: B closes 8% and
20% of the gap to the equally large dense model, 50% was required. I won't sugar-coat it: in this configuration,
100 M extra parameters in the table are worth about a fifth of what the same number of dense parameters does.

**What speaks against it being just a bug:** the tests are green, the table is healthy and broadly used, and after 3
epochs the model relies heavily on it (+44–53% PPL without the memory; random entries are even worse). So the memory
works. It just brings little more than the FFN it replaces.

**Possible reasons** (ordered by my estimate of how likely they are, none of them shown here):

1. **Flat weighting within the top-k.** Each head mixes 28–30 of 32 entries almost equally; the layer hardly
   retrieves in a targeted way. Candidates: the query BatchNorm fixes the query scale, and weight decay pulls the
   keys small. `v2-sharpness` is prepared for that (v2a: no weight decay on the keys; v2b: additionally a learnable
   temperature), see `docs/notes/V2_SHARPNESS.md`.
2. **Data regime.** The lead grows with data (2% → 4.4%), but saturates from ≈ 240 M tokens on, while training
   repeats the same 118 M tokens. The papers show the benefit at hundreds of billions of tokens and mostly on
   fact-heavy QA tasks; perplexity on WikiText-103 measures "retrieving knowledge" only indirectly.
3. **Learning rate of the values.** 1e-3 at a base LR of 6e-4 is a ratio of 1.7× (Lample: 4×).
4. **Only one memory layer.** Meta (Memory+) uses three layers with a shared table.
5. **Small compute core.** With d = 384 and 21 M dense parameters the capacity of the query network is small too.

**Cost.** Same FLOPs, but B trains 21% slower, is 31% slower at prefill and ≈ 6% slower at batch-1 decoding, and
needs 1.6 GiB more VRAM. That comes from the irregular memory access (gather/scatter, Adam over 100 M values), not
from compute, and it's exactly the part the later stages are about. **Important limitation:** the comparison is at
equal tokens and equal FLOPs, not at equal compute time. In the same wall-clock time A would have seen ≈ 27% more
tokens on this hardware. Whether B would still be ahead then wasn't measured; given how much A improves between 1 and
3 epochs, that's not a given.

**For the whole project.** Stage 1 doesn't refute the approach, but it doesn't give a reason to scale it yet either.
Before stage 2 I would run the prepared v2 variants (≈ 2 h). If they don't make the weighting clearly sharper and
don't bring B clearly further, the next levers would be more data (a bigger corpus instead of repetition), several
memory layers and a higher value LR. Important for stage 3: with better training the reads are spread more evenly,
so hot-set caching helps less (20% of the entries → 53% of the reads).

## Limitations of this study

- **C is clearly undertrained at 1 epoch.** 118 M tokens for 122.7 M parameters (without embeddings) is ≈ 1 token
  per parameter; compute-optimal (Chinchilla) would be ≈ 20. The distance A↔C, and with it the gap B is supposed to
  close, is smaller at 1 epoch than with a fully trained C. The 3-epoch run softens this only a little
  (≈ 3 tokens/parameter), and repeats the data instead.
- **Seed noise is a rough estimate.** With 2 seeds, s is a single difference, not a spread. The criterion stays as
  fixed, but it's rather optimistic about how well it separates.
- **The training loss in the curves lags behind.** It's the mean over each eval interval, during which the model
  keeps improving; train > val early in training is an artefact, not a bug. What matters for memorisation in the
  3-epoch run is whether the gap closes or reverses.
- **BatchNorm on the query** sees the batch statistics in training, including later tokens (with 8,192 tokens per
  micro-batch a very weak channel). All reported numbers are measured in eval mode with fixed statistics; causality
  there is checked by a test.
- **Dense Adam on the value table.** The unit test shows that gradients only flow into entries that were read. But
  Adam's momentum keeps moving entries that weren't read for a few more steps. The later goal "update single entries
  in a targeted way" needs a sparse optimizer (e.g. SparseAdam / lazy Adam); that wasn't part of this stage.
- **Throughput** is measured in PyTorch eager mode without custom kernels. Batch-1 decoding is limited by kernel
  launch overhead (≈ 150–210 tok/s for all models) and says nothing about the memory bandwidth the later stages are
  about. B is ≈ 20% slower than A in training despite equal FLOPs (gather/scatter on the 100 M table, Adam over
  100 M values).
- **Perplexity** is computed on GPT-2 BPE tokens and can't be compared directly with the word-level WikiText-103
  values in the literature (the word-PPL column converts the total NLL to the 217,646 words + `<eos>` and is only
  roughly comparable).
- **Small scale, one dataset.** The papers show the benefit of memory layers at trillions of tokens and on
  fact-heavy QA tasks; WikiText-103 with ≤ 354 M tokens is a far smaller regime.

## Artefacts

- `runs/<phase>/<run>/run-info.json`, `metrics.csv`, `train_log.csv`, `stdout.log`: in git.
- `runs/<phase>/<run>/model.pt` (weights, fp32), for B also `mem_access_train.npy` (reads per entry over the whole
  training), `mem_access_val.npz` (reads and summed softmax weights per entry on the val set) and
  `mem_index_sample.npz` (index sample for stage 3): only on disk (`.gitignore`, too big for git).
- Plots and tables: `report/`, made with `scripts/make_report.py probe ep1 ep3`.

## Stage 1b: quick test B-1M (criterion fixed before the run, 2026-10-02)

> Status: **finished on 2026-10-02.** Criterion unchanged since `0832753`; both runs on `5fe3245`, not dirty.
> **Result: criterion met** (B-1M −15.1% val PPL against A) **→ big test.**

**Question:** Does a bigger memory configuration on fresh (non-repeated) data bring a clear advantage? Only if so,
a big test follows.

**Setup (1 seed each, init seed 0, same data seed):**

| | A | B-1M |
|---|---|---|
| Compute core | as stage 1 (d = 384, 12 layers) | like A |
| Memory | – | 3 memory layers instead of the FFNs in layers 3, 7, 11 (index 2, 6, 10; centred, spacing 4 as Meta "Memory+") |
| Table | – | **one shared** value table with 1024² = 1,048,576 entries × 384 (402.7 M parameters); keys, query network, BatchNorm and swilu per layer |
| Search | – | per layer 4 heads, top-32, key dim 256 |
| Value LR | – | **4 × base LR** = 2.4e-3 (Lample's ratio), same schedule, no weight decay |
| Rest | optimisation as stage 1 (AdamW, LR 6e-4, warmup 5%, cosine to 10%, 32,768 tokens/step) | like A |
| Data | **500 M fresh tokens**, every sequence exactly once (15,258 steps) | same token stream |

Parameters: A 40.6 M (21.2 M without embeddings); B-1M 444.9 M (425.6 M without embeddings, of which 402.7 M table),
active per token 23.1 M. MACs/token: A 45.3 M, B-1M 47.1 M (+4%, because the sub-key search runs over 1024 instead of
512 keys per half).

**Data:** English Wikipedia (`wikimedia/wikipedia`, dump 20231101.en, GPT-2 BPE). Articles are shuffled with a
fixed seed; all articles whose title appears in the WikiText-103 validation or test set are excluded. A disjoint set
of Wikipedia articles (≈ 1 M tokens) serves as validation set.

**Data, as built** (`scripts/prepare_wikipedia.py`, `data/wikipedia_en_gpt2/meta.json`): 6,407,814 articles in the
dump; training 505 M tokens from 689,951 randomly chosen articles (500 M of them are used), validation 1.48 M tokens
from 1,917 disjoint articles. Of the 122 WikiText-103 val/test titles, 14 were among the chosen articles and were
removed (expected at 12.5% selection: ≈ 15). Renamed articles can slip through the title match.

**Measured beforehand** (79 steps, micro-batch 4): B-1M 35.7 k tok/s and 11.7 GiB peak VRAM in training (estimated
were 40–55 k tok/s and 10.5–11.5 GiB; abort limits 30 k tok/s and 15 GiB). Expected run time ≈ 4.1 h for B-1M and
≈ 1.6 h for A.

**Criterion (specification):** B-1M has **at least 10% lower val PPL than A**, otherwise no big test.
Operationalisation: token perplexity at the end of training on the held-out Wikipedia validation set (same
preprocessing as training), eval mode; **met if PPL(B-1M) ≤ 0.90 × PPL(A).** Seed noise in stage 1 was ≈ 1–1.4% of
the PPL and so far below the 10% threshold; one seed per model is enough for this decision.

Also reported, but not decisive: WikiText-103 val PPL (different text format, so only comparable A↔B-1M, not with
stage 1), table health (usage, concentration, KL), sharpness of the softmax, throughput, VRAM. If the table is
unhealthy (< 60% usage), that's named as a possible reason; the decision rule stays as specified.

### Result stage 1b

| | Val PPL Wikipedia (decisive) | Val PPL WikiText-103 (reported only) | Train tok/s | VRAM train | Train time | Decode b=1 tok/s | Prefill tok/s |
|---|---|---|---|---|---|---|---|
| A | 25.67 | 78.38 | 91,563 | 7.89 GiB | 91 min | 220 | 385,525 |
| B-1M | **21.80** | 66.28 | 33,383 | 11.70 GiB | 249 min | 175 | 150,398 |
| B-1M / A | **0.849 (−15.1%)** | 0.846 (−15.4%) | 0.36 | | 2.7× | 0.80 | 0.39 |

**Criterion PPL(B-1M) ≤ 0.90 × PPL(A): met (0.849) → big test.** Seed noise in stage 1 was ≈ 1–1.4%; the distance of
15% is far above that, even with only one seed per model.

![Val PPL stage 1b](report/s1b_val_ppl.png)

![Distance to A, stage 1b](report/s1b_relative_to_A.png)

**Course:** B-1M's lead grows over the whole training: −10.0% at 100 M tokens, −12.4% at 200 M, −14.0% at 300 M,
−15.1% at the end, and at the end it isn't quite saturated yet. On the WikiText validation set (different text
format, never trained on) the distance is just as big (−15.4%). Memorisation isn't an issue with fresh data: val
minus train is the same for both models (≈ 0.07 nats, the difference between training and validation articles).

**Table and memory** (`runs/s1b/B-1M-s0/diagnostics.json`, first 246,784 val tokens):

- Usage of the shared table 100%, the most-read 1% get 11.8% of the reads, KL 0.62. On its own, the layer in
  layer 3 reads 91.9% of the entries (top-1% share 26%), layers 7 and 11 ≈ 99% each. Every entry was read in
  training (10% quantile ≈ 44 k, median ≈ 115 k reads).
- **The model depends heavily on the memory:** all three memory layers set to zero → PPL 22.5 → 88.0; random
  instead of found entries → 93.8. The output norms of the memory layers (4.6 / 11.0 / 20.2) grow with depth like
  those of the dense FFNs.
- The weighting within the top 32 is flat as in stage 1 (effectively 28.6 of 32 entries, top-1 weight 0.095; per
  layer 29.1 / 28.6 / 28.0). So the gain doesn't come from sharper retrieval.
- The values have moved a lot (norm ≈ 3.6–3.7 instead of 1.0 at initialisation, value LR 2.4e-3).

![Table health stage 1b](report/s1b_memory_health.png)

**Access pattern (stage 3, `report/s1b_access_stats.json`):** the top 1 / 10 / 20 / 50% of the entries get
12 / 39 / 55 / 82% of the val reads; the 20% hottest in training cover 53% of the val reads (Spearman 0.92).
Repetition within 1 / 16 / 256 tokens: 1.6 / 11.3 / 41.5% (over all three layers). The index sample now has the shape
[token, layer, head, k].

**What the result doesn't say (no sugar-coating):**

- **Equal FLOPs, but not equal time.** B-1M has +4% MACs per token, but needs 2.7× as long on this hardware
  (33.4 k instead of 91.6 k tok/s), 48% more VRAM (11.7 instead of 7.9 GiB), and is 2.6× slower at prefill and
  1.25× slower at batch-1 decoding. In the same compute time A could have seen ≈ 2.7× as many tokens. How good A
  would be then wasn't measured; A's own curve still drops by 18% between 250 M and 500 M tokens (intermediate value
  at a high LR against the final value, so only a rough hint). The question "is the memory worth it per compute
  time?" is open and belongs in the big test or the later stages (custom kernels, offloading).
- **Several things changed at once.** Compared with stage 1 (B: −4.4% at 354 M repeated tokens), table size (4×),
  number of memory layers (3 instead of 1), value LR (2.4× higher), dataset and repetition all differ. This test
  doesn't say which factor brings how much.
- **No dense comparison with the same parameter count.** B-1M has 445 M parameters (A: 41 M). A C-like model is
  missing here; the quick test only answers "clearly better than A at equal compute per token?".
- **One seed per model**, one dataset, one point in time (500 M tokens).

## Extra test: quantising B-1M's value table (no training, 2026-10-02)

Only the value table of `runs/s1b/B-1M-s0` is quantised afterwards (in memory, the checkpoint is only read: SHA-256
identical before and after the test); all other weights stay fp32. Measured on the complete Wikipedia validation set
(1,484,009 tokens), same evaluation code as in training. Script: `scripts/quantize_table_eval.py`, raw data:
`report/quant_B-1M-s0.json`.

**Method** (one fp16 scale per row, i.e. per entry with 384 values; 2.1 MB for all scales): 8/4/3/2 bit like
llama.cpp `Q_0`, but per row: codes from −2^(b−1) to 2^(b−1)−1, the value with the largest magnitude in the row is hit
exactly. Ternary like BitNet b1.58: scale = mean magnitude of the row, values −1/0/+1.

| Bits per value | Table (MB) | Val PPL Wikipedia | Loss against unquantised | rel. error of the table |
|---|---|---|---|---|
| 32 (fp32, reference) | 1,610.6 | 21.801 | – | 0.000 |
| 16 (bf16) | 805.3 | 21.800 | −0.001 (−0.00%) | 0.002 |
| 8 | 404.8 | 21.799 | −0.002 (−0.01%) | 0.007 |
| 4 | 203.4 | 21.828 | +0.027 (+0.12%) | 0.115 |
| 3 | 153.1 | 21.911 | +0.110 (+0.50%) | 0.231 |
| 2 | 102.8 | 22.281 | +0.480 (+2.20%) | 0.467 |
| 1.6 (ternary, 1.58 bits of information) | 82.6 | 25.237 | +3.436 (+15.76%) | 0.512 |

**Discussion:**

- Down to 4 bits quantisation costs practically nothing: +0.12% PPL at an eighth of the fp32 size (203 instead of
  1,611 MB). 8 bits and bf16 can't be told apart from fp32. 3 bits cost 0.5%, 2 bits 2.2%.
- Ternary collapses (+15.8%, PPL 25.24): that removes almost the whole lead over A (25.67). Remarkable, because the
  relative error of the table is similar at 2 bits (0.47) and ternary (0.51); the fourth level and the absmax scale
  of 2 bits apparently keep the large values that matter.
- For stage 3 (SSD): at 4 bits an entry is 194 bytes (384 × 0.5 + 2). Per token and memory layer 128 entries are
  read, so ≈ 25 KB; the whole table fits easily into RAM at 203 MB.
- Limitations: only post-training quantisation (quantisation-aware training could improve 2 bits and ternary), one
  model, one seed, only the table (the rest stays fp32), one method per bit width.

## Stage 1c ("Hampter"): sparse optimizer, second seed, equal compute time (criteria fixed before the start, 2026-10-03)

> Status: **finished on 2026-10-03** (queue 01:01–14:07, all four runs on `0a5260e`, not dirty).
> Criteria and procedure fixed in commit `8cd1fde`, before any of the runs started.
> **Result: optimizer ok – met (+0.16%). Stable – met (both seeds −15% against A).
> Equal compute time – "competitive" (−3.0%), the "clear advantage" (≥ 5%) was missed.**

**Questions:** (1) Does an optimizer that only touches the table rows that were read give the same result as the
dense one? (2) Is B-1M's lead from the quick test stable over two seeds? (3) Does B-1M keep up when A gets the same
**compute time** (instead of the same number of tokens)?

### Steps 1–2: sparse optimizer and measurement (before the go)

- **Optimizer** (`smlm/sparse_values.py`, model `B-1M-sparse`): the value table no longer gets a dense gradient.
  Rows that were read are collected in their own accumulator, and Adam (without weight decay) updates only rows
  that were read. Values **and** Adam state of rows that weren't read stay bit-exactly the same (unit test
  `tests/test_sparse_values.py`, CPU and GPU; plus the same gradients as the dense path, the same clipping norm, and
  a first step identical to `torch.optim.Adam`). All other parameters: AdamW as before. No custom kernels.
- **Difference from the previous B-1M:** there, AdamW's momentum also moves rows that weren't read in a step. With
  32,768 tokens per step almost every row is read, though, so a similar result is expected, but not assumed (that's
  what the criterion "optimizer ok" is for).
- **Speed** (122 steps each, micro-batch 4): 39,525 tok/s against 35,374 tok/s dense = **1.12×**. The required 1.5×
  was missed and reported; decision: run anyway, the limit is dropped, no custom kernel. Peak VRAM 11.66 instead of
  11.70 GiB. Raw data: `report/hampter_measure_B-1M-sparse.json`, `report/hampter_measure_B-1M_dense.json`.
- **Is the GPU waiting for data?** No: GPU utilisation 100% at median and minimum, the data path (memmap → pinned →
  GPU) takes 0.2 ms per step (0.02% of 829 ms). Process CPU ≈ 4.7 cores, RSS 3.2 GB, system RAM 14.6 of 125 GB.
  **So nothing was changed in the data path.**
- **Heat** (10 min of B-1M-sparse under full load): edge 51 °C, hotspot 82 °C, memory 82 °C, ≈ 240 W, constant after
  3 min, no throttling.

### Runs (queue `scripts/run_hampter.py`, unattended, in this order)

| # | Run | Data | Tokens | estimated duration | VRAM (training peak) |
|---|---|---|---|---|---|
| 1 | B-1M-sparse, init seed 0 | as quick test (500 M Wikipedia, data seed 1234) | 500 M | ≈ 3.8 h | ≈ 11.7 GiB |
| 2 | A at the same compute time as run 1, init seed 0 | 1.5 B-token Wikipedia stream (see below) | train time(1) × tok/s(A) ≈ 1.16 B | ≈ 3.8 h | ≈ 7.9 GiB |
| 3 | A, init seed 1 | as quick test | 500 M | ≈ 1.6 h | ≈ 7.9 GiB |
| 4 | B-1M-sparse, init seed 1 | as quick test | 500 M | ≈ 3.8 h | ≈ 11.7 GiB |

Total ≈ 13 h. Everything else as in the quick test: value LR 2.4e-3, micro-batch 4 (B) or 8 (A), evaluation every
10 M tokens, WikiText-103 val as second val set. Micro-batch 8 for B-1M-sparse is ruled out: in the test it filled
99% of the VRAM and triggered a graphics reset of the desktop (2026-10-03, 00:14).

**Checked beforehand (function tests, not part of the evaluation):**

- **Pairing:** B-1M-sparse s0 with the real 500 M schedule starts bit-identically to B-1M s0 (val PPL at step 0
  identical). The training losses of the first 60 steps agree to 5–6 digits. So "optimizer ok" measures the
  optimizer and not a changed initialisation.
- **Abort path:** a forced abort ends with exit code 3 and status `aborted`, `abort_check` is in `run-info.json`.
- **Complete run** up to the inference benchmark: no errors. Found and fixed along the way: the table's gradient
  accumulator (1.5 GiB) stayed allocated through the autograd graph of the last loss until the inference
  benchmark. The inference VRAM values would have been 1.5 GiB too high; now 2.31 / 3.89 GiB as with the dense
  B-1M. It has no effect on training.
- **No VRAM cap for PyTorch:** a cap of 13.5 GiB was tested. It doesn't work on this ROCm stack, because memory
  freed with `expandable_segments` doesn't go back to the driver. The process took the whole VRAM despite the cap,
  and the inference benchmark aborted (in a test run that diverged because of 3 warmup steps and therefore needed
  more memory). So the configuration stays the one that ran without problems in the quick test for 4.4 h and in the
  10-min heat test (peak ≈ 11.7 GiB PyTorch, ≈ 14.1 GiB used in total).

- **Equal compute time:** A's token budget = pure training time of run 1 (without evaluations) × measured throughput
  of A s0 in the quick test (tokens / pure training time = 91,569 tok/s), rounded down to whole steps. A gets its own
  cosine schedule over that length (warmup 5%, decay to 10%). So that A doesn't repeat data, a bigger Wikipedia
  slice was prepared (`data/wikipedia_en_gpt2_1500m`, article band 0.35 instead of 0.125): **same val set**
  (byte-identical), and the first 505 M training tokens are byte-identical to the quick-test data (both checked with
  `cmp`). 1.5 B training tokens from 2,052,458 articles; 34 of the 122 WikiText val/test titles were in the wider band
  and were removed. If run 1 were slower (e.g. GPU used by the desktop), A would get more tokens; that's why the
  throughput history of run 1 is reported too.
- **Temperature:** every 10 s edge, hotspot, memory, power, shader and memory clock, fan, VRAM in
  `runs/hampter/<run>/gpu_thermal.csv`; maxima in the status.

### Criteria (specification verbatim, evaluation below)

- **Optimizer ok:** B-1M-sparse s0 at most 2% worse than the previous B-1M s0.
- **Stable:** both B-1M-sparse seeds at least 10% better than the mean of both A seeds (A s0 from the quick test,
  A s1 new).
- **Equal compute time (against B-1M-sparse s0):** at least on par = competitive; at least 5% better = clear
  advantage.

| Criterion | Operationalisation (val PPL Wikipedia at the end of training, same val set as the quick test) |
|---|---|
| Optimizer ok | PPL(B-1M-sparse s0) ≤ 1.02 × PPL(B-1M s0) = 1.02 × 21.801 = **22.237** |
| Stable | PPL(B-1M-sparse s0) **and** PPL(B-1M-sparse s1) ≤ 0.90 × ½ (PPL(A s0) + PPL(A s1)); PPL(A s0) = 25.665 |
| Equal compute time | Q = PPL(B-1M-sparse s0) / PPL(A equal time). Q ≤ 0.95 → **clear advantage**; 0.95 < Q ≤ 1.00 → **competitive**; Q > 1.00 → **not competitive** |

"Better" means lower PPL, "10% better" as in the quick test PPL ≤ 0.90 × reference. I read "on par" literally
(Q ≤ 1.00); if Q is within the seed noise (from A s0/s1 and B-1M-sparse s0/s1), the report says so, the verdict
stays as fixed. WikiText-103 val PPL, table health, throughput and VRAM are reported, but don't decide.

**Abort rule:** if B-1M-sparse s0 at 100 M tokens is more than 5% behind the previous B-1M s0 at the same point, the
queue stops and reports. Implemented in `smlm/train.py` (`--abort_ref`): at the evaluation at 99.94 M tokens (step
3050, the same point as in the quick test) the run aborts if PPL > 1.05 × 39.293 = **41.258**. The run then ends with
status `aborted` (weights are saved), and the whole queue stops.

### Status

<!-- HAMPTER-STATUS:BEGIN -->

**Status** (generated automatically by `scripts/hampter_status.py`, as of 2026-10-03 14:07)

| Run | Status | Tokens | Val PPL Wikipedia | Val PPL WikiText | Train time | tok/s median (5% quantile) | VRAM train | max. edge / hotspot / memory | max. power | Clock under load |
|---|---|---|---|---|---|---|---|---|---|---|
| B-1M s0 (quick test, dense optimizer, reference) | done | 500 M | 21.801 | 66.28 | 249 min | 33,383 (33,345) | 11.70 GiB | – | – | – |
| A s0 (quick test, reference) | done | 500 M | 25.665 | 78.38 | 91 min | 91,563 (91,488) | 7.89 GiB | – | – | – |
| B-1M-sparse s0 | done | 500 M | 21.837 | 65.51 | 216 min | 38,517 (38,464) | 10.52 GiB | 48 / 79 / 80 °C | 238 W | 3,120 MHz |
| A s0 at equal compute time | done | 1,186 M | 22.508 | 66.44 | 216 min | 91,498 (91,395) | 7.89 GiB | 49 / 83 / 80 °C | 261 W | 2,993 MHz |
| A s1 | done | 500 M | 25.756 | 78.48 | 91 min | 91,641 (91,467) | 7.89 GiB | 49 / 82 / 80 °C | 261 W | 2,991 MHz |
| B-1M-sparse s1 | done | 500 M | 21.752 | 65.09 | 216 min | 38,577 (38,519) | 10.52 GiB | 51 / 81 / 82 °C | 240 W | 3,121 MHz |

**Abort rule** (B-1M-sparse s0 at 99.9 M tokens): PPL 39.339 against 39.293 for the previous B-1M s0 = +0.12% (limit
+5%) → **continue**.

**Budget A at equal compute time:** 12,957 s training time of B-1M-sparse s0 × 91,569 tok/s (A s0 in the quick test)
= 1,186.4 M tokens (36206 steps).

| Criterion (fixed beforehand) | Condition | Measured | Result |
|---|---|---|---|
| Optimizer ok | PPL(B-1M-sparse s0) ≤ 1.02 × 21.801 = 22.237 | 21.837 (+0.16%) | **met** |
| Stable | both B-1M-sparse seeds ≤ 0.90 × mean(A s0, A s1) = 23.140 | s0 0.849×, s1 0.846× (mean A 25.711) | **met** |
| Equal compute time | PPL(B-1M-sparse s0) / PPL(A equal time): ≤ 1.00 competitive, ≤ 0.95 clear advantage | 0.970 (−3.0%); train time B 216 min, A 216 min | **competitive** |

**GPU maxima over all Hampter runs** (measured every 10 s, `gpu_thermal.csv` per run): edge 51 °C, hotspot 83 °C,
memory 82 °C (limits according to the driver 110 / 110 / 108 °C), power 261 W.

<!-- HAMPTER-STATUS:END -->

### Result stage 1c (evaluated by hand)

| | Val PPL Wikipedia | Val PPL WikiText | Tokens | pure train time | Train tok/s | VRAM train | Decode b=1 tok/s | Prefill tok/s |
|---|---|---|---|---|---|---|---|---|
| A s0 / s1 (mean) | 25.711 (25.665 / 25.756) | 78.43 | 500 M | 91 min | 91,600 | 7.89 GiB | 219 | 385,000 |
| **A s0 at equal compute time** | **22.508** | 66.44 | 1,186 M | 216 min | 91,498 | 7.89 GiB | 215 | 385,555 |
| B-1M s0, dense optimizer (quick test) | 21.801 | 66.28 | 500 M | 249 min | 33,383 | 11.70 GiB | 175 | 150,398 |
| **B-1M-sparse s0** | **21.837** | 65.51 | 500 M | 216 min | 38,517 | 10.52 GiB | 182 | 150,541 |
| B-1M-sparse s1 | 21.752 | 65.09 | 500 M | 216 min | 38,577 | 10.52 GiB | 178 | 150,039 |

![Val PPL over training time, stage 1c](report/hampter_val_ppl_time.png)

![Val PPL over tokens, stage 1c](report/hampter_val_ppl_tokens.png)

**1. Optimizer ok – met.** B-1M-sparse s0 ends at 21.837 instead of 21.801 (+0.16%, +2% allowed). The distance stays
at +0.1 to +0.3% over the whole training (100 / 200 / 300 / 400 / 500 M tokens: +0.12 / +0.34 / +0.12 / +0.12 /
+0.16%). That's within the seed noise (B-1M-sparse s0 ↔ s1: 0.39%). The table is just as healthy (usage 100%, the
most-read 1% get 11.7% and 11.4% of the reads, KL 0.62 / 0.60; before 11.8% and 0.62). The softmax is just as flat
(28.6 / 28.5 effective entries of 32). Gain: 1.15× faster (216 instead of 249 min) and 1.2 GiB less VRAM in
training. At inference the optimizer changes nothing.
**But:** the goal was 1.5×. Per token B-1M is still 2.4× slower than A (38.5 k against 91.6 k tok/s).

**2. Stable – met.** Both seeds are clearly below the limit of 0.90 × 25.711 = 23.140: s0 at 0.849×, s1 at 0.846×
(−15.1% and −15.4%). The seeds are 0.39% apart for B-1M-sparse and 0.35% for A. So the lead is about 40 times the
seed noise. On WikiText val (never trained on, different format) the distance is just as big (−16.5% / −17.0%
against mean A).

**3. Equal compute time – "competitive", not "clear advantage".** With the same pure training time (216 min, the
times differ by only 5 s) A sees 2.37× as many tokens and reaches 22.508. B-1M-sparse s0 at 21.837 is 3.0% below
that (Q = 0.970; s1, not part of the criterion: 0.966). So the criterion "at least on par" is met, "at least 5%
better" (Q ≤ 0.95) is missed. The difference of 0.67 PPL is more than three times 2 × the seed spread (0.18). A at
equal time has only one seed, though, so this is a clear hint, not a statistical proof (corrected 2026-10-05 after the
Codex review; before, it said "is real"). On WikiText val the lead is smaller (−1.4% / −2.0%).
At equal tokens B-1M was 15% ahead, at equal time about a fifth of that is left.

**Discussion (no sugar-coating):**

- **The advantage per token is robust, the advantage per compute time is small.** On this hardware and with this
  implementation, the memory table buys 3% perplexity at equal training time. That's measurable and real, but far
  from the 15% at equal tokens.
- **Inference costs more, A at equal time doesn't.** A at equal time is just as cheap to run as A: 155 MB of
  weights, 385 k tok/s prefill, 215 tok/s decoding. B-1M needs 1.7 GB of weights (fp32; with a 4-bit table and fp32
  rest ≈ 0.37 GB, see quantisation), has 2.6× less prefill throughput and is ≈ 17% slower at decoding. Counting
  training and inference together, A at equal time currently looks hardly worse: 3% higher PPL, but clearly cheaper
  to run.
- **The time verdict depends on the implementation.** The compute per token is almost the same (+4% MACs). The 2.4×
  run time comes from the memory lookup (top-k search, `embedding_bag`, Adam step over 1 M rows) on ROCm without
  custom kernels. A faster lookup would shift the result in B's favour; by how much wasn't measured. Conversely, A
  wasn't optimised further either (e.g. `torch.compile`).
- **One data point.** Equal time was compared only at ≈ 3.6 h. Whether the time advantage grows with longer training
  (at equal tokens the lead grew from −10% at 100 M to −15% at 500 M tokens) is open; the intermediate values of the
  two curves can't be compared directly because of the different cosine schedules.
- **Data:** A at equal time saw tokens beyond the 505 M from the same article pool (same preprocessing, same val
  set). A distribution difference is practically ruled out, but the data isn't the same.
- **Temperatures** (every 10 s, 13 h): maxima edge 51 °C, hotspot 83 °C, memory 82 °C, 261 W (limits
  110 / 110 / 108 °C). No throttling: throughput was constant in all runs (5% quantile ≤ 0.3% below the median).

## Optimisation (Triton kernels) and cloud preparation (from 2026-10-03)

**Goal:** B-1M as fast as realistically possible, without changing the results. After that, runs with bigger tables
on a rented GPU (first planned at IONOS H200-S, now Runpod: 1 × H200 SXM 141 GB). Kernels only in Triton (ROCm
**and** CUDA). The PyTorch implementation stays as reference and fallback, switchable by configuration.

**Targets** (against A on the same GPU):

| | Target | before (B-1M-sparse) |
|---|---|---|
| Training | ≥ 0.6× as fast as A per token | 0.42× |
| Decoding (batch 1) | at most 10% slower than A | 17% |
| Prefill (16 × 1024) | at most 1.5× slower than A, measured with a bf16 table; fp32, bf16 and 4 bit reported separately | 2.6× (fp32) |

**Abort rule:** if an optimisation step brings less than a 10% gain, stop and report where the limit is. Exception:
the fused lazy-Adam kernel gets built anyway because of the memory needs of large tables.

**Correctness:** every kernel against the reference (forward and gradients), tables with 262k, 1M and 4M rows.
Scores exactly equal, indices equal except for ties at the top-k boundary, outputs and gradients within fixed
tolerances. Tests on the CPU too (`TRITON_INTERPRET=1`, small sizes). Comparison run over 20 M tokens kernel against
reference: loss curves practically identical. All old tests stay green.

### Baseline measurement (profiler, before any kernel)

`scripts/profile_memory.py` → `report/profile_before.json`. RX 9070, trained checkpoints, real batches.

- **Training:** B-1M-sparse needs 848 ms per step (32,768 tokens), A 357 ms (0.42×). The extra 490 ms split up like
  this:
  - collecting row gradients: **285 ms** (24 calls of 11.9 ms; of that `index_put_` with sorting 6.2 ms, weighting
    2.8 ms, gathering 1.4 ms, weight gradients 1.2 ms)
  - lookup forward: 134 ms (top-k 72, `embedding_bag` 48)
  - rest of the memory backward: 40 ms
  - lazy Adam: 29 ms
  - statistics and clipping: 10 ms
  - minus the 3 FFNs that A has instead: −19 ms
- **Prefill** (16 × 1024): B 109 ms, A 42 ms. Per memory layer 24.4 ms (top-k 14, mixing 8.2), an FFN takes 0.8 ms.
- **Decoding:** B 5.67 ms per token, A 4.67 ms. Per memory layer 0.44 ms (≈ 15 small kernel launches), an FFN
  0.10 ms.

### Criteria for the cloud runs (fixed before building, 2026-10-03)

Runs on a rented H200 (at the time of fixing IONOS, now Runpod; the criteria apply unchanged) with the data,
settings and val set of B-1M-sparse: B-1M (control run on the same hardware and with the same kernels), B-4M
(2048² = 4,194,304 entries) and B-16M (4096² = 16,777,216 entries), init seed 0 each, 500 M tokens.

Specification (verbatim, German original):

- **lohnt sich** (worth it): B-4M mindestens 3 % besser als B-1M (Cloud) UND B-16M nochmal besser als B-4M (B-4M at
  least 3% better than B-1M (cloud) AND B-16M better again than B-4M)
- **unklar** (unclear): Verbesserung, aber unter 3 % (improvement, but below 3%)
- **lohnt sich nicht** (not worth it): B-4M nicht besser als B-1M (Cloud) (B-4M not better than B-1M (cloud))

Operationalisation: val PPL Wikipedia at the end of training, same val set as stages 1b/1c.

| Verdict | Condition |
|---|---|
| worth it | PPL(B-4M) ≤ 0.97 × PPL(B-1M) **and** PPL(B-16M) < PPL(B-4M) |
| unclear | PPL(B-4M) < PPL(B-1M), but not "worth it" |
| not worth it | PPL(B-4M) ≥ PPL(B-1M) |

On the case "unclear": it also covers B-4M ≥ 3% better, but B-16M not better than B-4M. The report would name that
explicitly. B-1M-sparse's seed noise was 0.39%; differences below ≈ 0.8% (2 × spread) count as not solid and are
named as such.

### Step 1: collecting row gradients, fused (kernel 1)

`smlm/kernels.py::bag_backward_rows`, switched on with `mem_impl="triton"` (`--mem_impl triton`).

- **Before:** for each of the 524,288 lookups (4096 tokens × 128), w·grad is materialised (0.8 GB) and added with
  `index_put_` (sorting).
- **Kernel:** the lookups are sorted once by table row (`torch.sort`, 0.25 ms). Each program takes 32 sorted
  positions. A segmented scan in registers sums equal rows. Runs that lie entirely inside the program are written
  normally; only the at most two runs at the program borders atomically. The weight gradient dot(grad, row) comes
  out of the same pass.
- **First attempt:** an atomic add for every run, 17 ms per call, so slower than the reference. Atomics are expensive
  on the RX 9070.
- **One call in training shape** (real indices of a trained model, 254k different rows): reference 14.3 ms, kernel
  2.74 ms + 0.25 ms sorting (≈ 4.8×). Deviation from the reference: relative 3·10⁻⁷ (accumulator) and 2·10⁻⁷ (weight
  gradient), `touched` identical.
- **Tests** (`tests/test_kernels.py`): tables with 262k / 1M / 4M rows (4M with 64 instead of 384 columns on GPUs
  < 40 GB, otherwise the test doesn't fit into 16 GB), tolerance rtol = atol = 1e-5 relative to the scale. Plus the
  whole model (3 memory layers, shared table, 2 micro-batches): all gradients equal. On the CPU via
  `TRITON_INTERPRET=1` with small sizes. All 59 tests green. Fixed along the way: on the CPU `_embedding_bag` doesn't
  return `offset2bag` for fp32; the reference now builds it itself.

| Training step (32,768 tokens) | before | kernel 1 |
|---|---|---|
| forward | 261 ms | 262 ms |
| backward | 545 ms | 339 ms |
| lazy Adam + rest | 42 ms | 42 ms |
| **total** | **848 ms (38.7 k tok/s)** | **643 ms (50.9 k tok/s)** |
| against A (357 ms) | 0.42× | **0.56×** |

Gain 1.32× (abort rule: > 10%, continue). Raw data: `report/profile_k1.json`.

### Step 2: lookup fused (kernel 2)

`smlm/kernels.py::pk_select` / `PKSelect` (selection) and `bag_forward` (weighted mixing).

- **Same partial scores:** s1, s2 still come from the same `einsum` as in the reference and are therefore
  bit-identical.
- **Selection in one kernel**, one program per (token, head):
  - top-32 of each half (int32 keys from an order-preserving bf16 code and the index, in two stages over blocks of
    128).
  - pair sums, rounded to bf16 like `s1 + s2` in PyTorch (RTNE with integer arithmetic, so that GPU and CPU
    interpreter round the same way).
  - top-32 of the pairs and softmax in fp32.
  - Of the 32 × 32 pairs only the 130 with (i+1)(j+1) ≤ 32 can make it at all, all others are dominated by ≥ 32 pairs
    that are at least as large.
- **Backward:** softmax derivative, then the gradient of each selected score into its two partial scores, summed in
  fp32 and rounded once to bf16 (like the reference's autograd). The `einsum` backward stays PyTorch.
- **Mixing:** one program per (token, 128 columns), 32 lookups per tile; table fp32 or bf16.
- **Exactness:**
  - The selected scores are **bit-identical** to the reference. Each selected index provably has exactly its score,
    so the selection is an exact top-k.
  - Indices differ only for equal scores. With bf16 that's common: with random data 59% of the rows have a tie
    somewhere with a different choice. `torch.topk` doesn't fix the order of ties either.
  - Softmax weights: deviation ≤ 3·10⁻⁸.
- **Tests:** selection at 512 / 1024 / 2048 keys per half and at N = 1. Backward against the exactly summed
  derivative: ≤ 1 bf16 ulp. Mixing against `embedding_bag` at 262k / 1M / 4M rows, fp32 and bf16. CPU interpreter
  green. 71 tests green in total.

| One call (training shape N = 4096) | Reference | Kernel |
|---|---|---|
| top-k of both halves + cross top-k + softmax | 2.83 ms | 0.86 ms |
| mixing (`embedding_bag` → kernel), fp32 table | 1.97 ms | 0.84 ms |
| memory layer forward total | 5.57 ms | 2.45 ms |

| | before | kernel 1 | kernel 1 + 2 | A |
|---|---|---|---|---|
| Training step | 848 ms | 643 ms | **584 ms** (forward 187, backward 355) | 358 ms |
| Training tok/s | 38.7 k | 50.9 k | **56.1 k** | 91.5 k |
| against A | 0.42× | 0.56× | **0.61×** ✅ (target ≥ 0.6) | |
| Prefill 16 × 1024 (fp32 table) | 109 ms (2.6×) | | 64.0 ms (1.52×) | 42.2 ms |
| Decoding per token | 5.67 ms | | 5.70 ms (+21%) | 4.69 ms |

- **Training gain of step 2:** 1.10× (643 → 584 ms), just above the 10% limit.
- **Backward** got a bit slower (339 → 355 ms): the dense partial-score gradients come from `scatter_add` in fp32
  plus rounding instead of the reference's top-k backward.
- **Decoding** doesn't change. With one token per step, the ≈ 15 kernel launches per memory layer set the time, not
  the compute.

Raw data: `report/profile_k2.json`, `report/profile_k2_infer.json`. The stage times there measure the PyTorch sub-steps;
for the kernels, `forward_total` is what counts.

### Step 3: inference lookup on bf16 and 4-bit tables (kernel 4) and decode graph

- **Inference table:** `Transformer.set_memory_inference_table("fp32" | "bf16" | "q4")` makes an inference copy of
  the shared table.
  - 4 bit with exactly the method of the quantisation test: fp16 scale per row, codes −8…7, two per byte. The test
    checks that the dequantisation is bit-identical with `quantize_table_eval.py`.
- **Kernel `bag_infer`:** mixes directly from fp32, bf16 or 4-bit rows. The swilu product `out * bf16(silu(pre))` and
  the bf16 cast before `value_proj` are built in. It's only used without gradients and under bf16 autocast; training
  stays unchanged.
- **Decoding:** measured, it's not the GPU that slows things down there, but the CPU. A memory layer needs 0.39 ms
  just to launch the ≈ 20 PyTorch ops and Triton kernels; the GPU work is ≈ 0.05 ms, an FFN needs 0.07 ms.
  - Remedy: `Transformer.set_memory_decode_graphs(True)` records `_forward` of the memory layer for one token once as
    a CUDA/HIP graph and replays it afterwards.
  - The same kernels run, the logits are bit-identical (test over 5 decoding steps, all three table types). A runs
    without graphs. **So the decoding gain comes from the graphs, not from a Triton kernel**; A would get faster with
    graphs too.
- **Val PPL** on the whole Wikipedia val set, B-1M-sparse s0 (`scripts/eval_kernels.py`,
  `report/eval_kernels.json`):

  | Variant | Val PPL | Deviation | Time |
  |---|---|---|---|
  | Reference | 21.8369 | – | 12.3 s |
  | Kernel, fp32 | 21.8357 | −0.005% | 8.5 s |
  | Kernel, bf16 | 21.8368 | −0.000% | 8.1 s |
  | Kernel, 4 bit | 21.8663 | +0.13% (quantisation test: +0.12%) | 8.3 s |

| Inference (RX 9070) | Prefill 16 × 1024 | against A | Decoding per token | against A |
|---|---|---|---|---|
| A | 42.4 ms | | 4.62 ms | |
| B before (reference) | 109.3 ms | 2.6× | 5.67 ms | +21% |
| B kernel, fp32 table | 62.9 ms | 1.48× | 5.75 ms | +25% |
| **B kernel, bf16 table** | **60.8 ms** | **1.44×** ✅ | 5.77 ms | +25% |
| B kernel, 4 bit | 62.5 ms | 1.47× | 5.69 ms | +23% |
| B kernel + decode graph, fp32 | 62.6 ms | 1.48× | **4.53 ms** | **−2%** ✅ |
| B kernel + decode graph, bf16 | 61.0 ms | 1.44× | 4.57 ms | −1% |
| B kernel + decode graph, 4 bit | 62.6 ms | 1.48× | 4.55 ms | −2% |

- **Prefill:** bf16 brings only 2 ms over fp32, 4 bit nothing. The values are no longer just read from memory;
  selection (≈ 3.2 ms per layer at 16k tokens), partial scores and projections are now just as big a share. 4 bit
  mainly saves memory (table 203 instead of 1,611 MB).
- **Prefill tokens:** the benchmark uses random tokens like `train.py`. With real text the reads are denser and the
  lookups rather get faster.

Raw data: `report/profile_k4_infer.json`.

### Step 4: lazy Adam fused (kernel 3, for the memory of large tables)

`smlm/kernels.py::lazy_adam_step`, automatically with `mem_impl="triton"`.

- **Kernel:** the same computation as `LazyRowAdam` (PyTorch's Adam formula, global step for the bias correction, no
  weight decay), but directly in the table: rows that were read get updated and their accumulator zeroed. Rows that
  weren't read aren't even loaded.
- **Memory:** the reference saves and restores the rows that weren't read (values and both moments), that's
  3 × 1.5 KB of temporary memory per unread row. At 16M rows with many unread ones that would be tens of GB, with the
  kernel 0.
- **Bug found:** the first version cleared the `touched` mask in the kernel with masked byte stores. On the RX 9070,
  rows of neighbouring programs were cleared before those had read them (result: 60–133 rows that were read but got
  no update). The test found it. The mask is now cleared after the kernel with a `zero_()`.
  *Addendum 2026-10-05:* minimal reproductions (`docs/rocm-issues/repro_byte_store.py`, 42 variants, plus one
  variant close to the kernel at the time) show **no** error with masked byte stores on the RX 9070. So the cause at
  the time was very likely a bug in my own kernel, not in ROCm or Triton; the old version wasn't kept. No bug report
  was filed.
- **Tests:** 3 steps against `LazyRowAdam` at 262k / 1M / 4M rows, with a clipping factor:
  - rows that weren't read bit-identical
  - rows that were read and both moments within rtol = atol = 1e-5
  - accumulator and mask empty afterwards

  On GPUs < 40 GB with 16 columns (otherwise > 13 GB VRAM). 87 GPU tests green, CPU interpreter 20 green (3 graph
  tests GPU only). Peak VRAM of the whole test run 7.4 GiB.
- **Speed:** table optimizer 29.0 → 22.5 ms per step, training step 584 → 575 ms (57.0 k tok/s, 0.62× A). Gain
  < 10%, as expected; built for the memory.

### Step 5: comparison run kernel against reference (correctness)

Same model (B-1M-sparse), same initialisation, same data, once with `mem_impl="torch"`, once with `"triton"` (all
kernels). Plots: `report/kernel_check_*.png`, numbers: `report/kernel_check_*.json`.

| Comparison | Training loss, deviation median / max. | Val PPL reference → kernel |
|---|---|---|
| **first 22 M tokens of the real 500 M schedule** (warmup 763 steps, as in the cloud), seed 0 | **0.04% / 0.13%** | **184.52 → 184.56 (+0.02%)**; intermediate values −1.0 … +0.5% |
| 20 M tokens with its own short schedule (warmup 31 steps), seed 0 | 0.37% / 0.46% | 181.93 → 178.01 (−2.2%) |
| the same, seed 1 | | 186.73 → 193.73 (+3.8%) |
| for comparison: reference seed 0 → reference seed 1 (short schedule) | | 181.93 → 186.73 (+2.6%) |

![Kernel against reference, real schedule](report/kernel_check_500msched.png)

- **In the regime of the real runs the curves are practically identical.** At steps 10–90 the losses agree to 5–6
  digits. After that the deviations grow slowly, but stay below 0.13%.
- **In the short schedule** the learning rate jumps to the full value after 31 steps, table usage collapses briefly
  (13% at 2 M tokens) and recovers. This phase is chaotic:
  - Both runs are equal up to step 20 and separate from step 30 on.
  - At the end the kernel is 2.2% better once and 3.8% worse once.
  - That's the same order of magnitude as two seeds of the reference (2.6%). No systematic difference is visible.
- **Why the runs drift apart at all (new finding):**
  - As in the reference, the partial scores are bf16 (autocast). With an 8-bit mantissa very many candidates have
    exactly the same score.
  - On real activations of the trained B-1M, **65% of the (token, head) rows have a tie at the 32nd score**. In
    **41%**, the kernel and `torch.topk` choose different, equivalent entries, on average 15 of 32.
  - Equal scores mean equal weights, but the output then mixes different value rows. Which ones, `torch.topk` doesn't
    fix either; the PyTorch reference on CUDA can just as well choose differently than on ROCm (not measured; the
    cloud control run B-1M shows it).
  - **So exactly equal curves can't be guaranteed**, probably not even with the reference on other hardware. What's
    achievable and shown: exactly equal scores, a valid exact top-k selection, equal curves in the real schedule.
  - The complete control is the cloud run B-1M (500 M tokens) against B-1M-sparse s0 from home; the seed noise there
    was 0.39%.
- **Side finding for the model (not changed):** because of the bf16 scores the selection has only a coarse
  resolution. Whether fp32 partial scores (hardly more expensive) improve the memory layer would be a separate
  experiment. It would change the results against all previous B runs and therefore doesn't belong in this
  optimisation.

### Result of the optimisation

Measured with `scripts/profile_memory.py` (RX 9070, trained checkpoints; `report/profile_before.json` →
`report/profile_after.json`).

| | Target | before | after | reached |
|---|---|---|---|---|
| Training: step (32,768 tokens) | | 848 ms (38.7 k tok/s) | 577 ms (56.8 k tok/s) | |
| Training against A (359 ms) | ≥ 0.6× | 0.42× | **0.62×** | ✅ |
| Prefill 16 × 1024 against A (42.7 ms), bf16 table | ≤ 1.5× | 2.6× (fp32) | **1.43×** (61.1 ms) | ✅ |
| Prefill, fp32 table / 4 bit | (reported) | 2.6× | 1.47× / 1.49× | |
| Decoding per token against A (4.60 ms) | ≤ +10% | +21% | **−2%** (4.50 ms, with decode graph) | ✅ |
| Decoding without graph | (reported) | +21% | +23% | |
| Val PPL B-1M-sparse s0 (fp32 / bf16 / 4 bit) | unchanged | 21.837 | 21.836 / 21.837 / 21.866 | ✅ |

**Gain per step** (abort rule: stop below 10%, except kernel 3):

| Step | Gain |
|---|---|
| Kernel 1 (row gradients) | training 1.32× |
| Kernel 2 (lookup) | training 1.10×, prefill 109 → 63 ms (1.73×) |
| Kernel 4 (bf16 / 4 bit) | prefill 63 → 61 ms (1.03×) |
| Decode graph | decoding 1.26× |
| Kernel 3 (lazy Adam) | training 1.02×; built for the memory |

Then I stopped: all targets are reached, and none of the remaining items promises another 10%.

**Where the limit is (honestly):**

- **Training:** of the 577 ms per step, 359 ms are the compute core that A has too. The memory layers still cost
  ≈ 220 ms:
  - backward ≈ 130 ms, of which kernel 1 ≈ 3 ms per call, the rest are partial-score gradients, projections and
    BatchNorm.
  - forward ≈ 60 ms.
  - lazy Adam 22 ms, statistics and clipping 10 ms.

  Another big step would need a fused backward for selection and partial scores (today `scatter_add` + `einsum`
  backward in PyTorch) or fused projections. None of them alone can be expected to bring more than 10%.
- **Prefill:** reading the values is no longer a bottleneck: bf16 brings only 2 ms, 4 bit nothing. The selection
  costs ≈ 3.2 ms per layer at 16k tokens; it's bound to a bitonic sort in Triton.
- **Decoding:** the gain comes **from the CUDA/HIP graphs, not from a kernel.** Without a graph B is at +23%, because
  a memory layer launches ≈ 20 ops (0.39 ms CPU). A runs without graphs; with graphs A would get faster too, but the
  distance would stay small.
- **Hardware:** all measurements come from the RX 9070. On the H200 the ratios are different: more bandwidth, and the
  kernel configurations aren't tuned for NVIDIA.

### Cloud preparation (Runpod, 1 × H200)

**Change of provider (2026-10-03):** IONOS was planned first (H200-S, 3.00 €/h; state in commit `36eccfe`), now
**Runpod**. Criteria, runs and data stay unchanged; only the setup, the storage location and the stopping at the end
have changed.

**Provider facts** (Runpod docs and runpod.io/pricing, retrieved 2026-10-03):

- **GPU and price:** H200 SXM 141 GB on-demand: Secure Cloud 4.59 $/h, Community Cloud 3.59 $/h (24 vCPU, 276 GB
  RAM). Billing by the second; the price in the console is what counts. An H100 with 80/94 GB isn't enough for B-16M.
- **Stopping:** a stop frees the GPU and ends the compute cost. `/workspace` (volume disk) stays and costs
  0.20 $/GB/month while stopped (150 GB ≈ 1 $/day); the container disk is wiped. **Terminate** deletes everything.
- **Stop from inside the pod:** every pod gets `RUNPOD_POD_ID` and a pod-scoped `RUNPOD_API_KEY`. With it the pod can
  stop itself (`POST https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop`, alternatively `runpodctl pod stop`). No
  key needs to be created.
- **Restart:** after a stop the GPU may be taken; the pod can then start with 0 GPUs on request. That's enough to
  fetch the checkpoints.
- **Driver and SSH:** driver and SSH come with the template "Runpod PyTorch". `scp`/`rsync` need a public IP ("SSH
  over exposed TCP"), hence Secure Cloud.
- **Balance:** if it drops to 0 $, pods get stopped, and pods without a network volume get **deleted along with their
  data**. Top up enough beforehand (≥ 40 $).

**Built** (all in git, instructions `docs/notes/CLOUD.md`):

- **`cloud/setup.sh`:** one command in the pod.
  - Sequence: packages, GPU check, deploy key + clone to `/workspace`, Python environment (PyTorch CUDA wheels with
    Triton), data, all tests, trial run of all three configurations, then the queue in tmux.
  - Everything that must survive a stop lives on `/workspace`; packages and the SSH key are set up again after every
    start.
  - Finished steps are skipped on a restart.
  - On an error: log to GitHub, message, stop the pod.
- **`cloud/fetch_data.py`:**
  - Downloads WikiText-103 and Wikipedia 20231101.en at **pinned Hugging Face revisions** and checks every raw file by
    SHA-256. All 45 raw files at home match the LFS checksums of these revisions.
  - Rebuilds the token files and aborts if one isn't **byte-identical** to the one at home
    (`cloud/data_sha256.txt`).
- **`scripts/run_cloud.py`:** queue B-1M (control) → B-4M → B-16M.
  - 500 M tokens each, settings of B-1M-sparse, `--mem_impl triton`.
  - GPU temperature, power and clock every 10 s (`smlm/gpu_monitor.py`, via `nvidia-smi`).
  - After every run: status in REPORT.md, `git pull --rebase` + push, phone message (ntfy, optional).
  - At the end: checksum list of the checkpoints, then **stop via the Runpod API** (`cloud/stop_pod.sh`).
  - Protection: a run without a log change for 30 min gets ended; a limit of 12 h for everything.
- **GitHub:** private repo `re133/sparse-memory-lm`, deploy key with write access for this repo only. The starter
  kit `~/smlm-cloud-kit/` lives on the PC (`setup.sh`, `stop_pod.sh`, `deploy_key`, `cloud.env`).
- **Not used:** the Runpod plugin for Claude Code. Its installation was blocked by the permission check, and it isn't
  needed either. The pod is created in the web console; only the pod-scoped key is used.

**Memory needs on the H200** (141 GB ≈ 131 GiB):

- The table needs 4 fp32 copies × 384 × 4 B = 6 KiB per row: values, accumulator, Adam m and v.
- "Rest" was measured with B-1M: 10.5 GiB peak minus 6.0 GiB table.

| | Entries | Table + optimizer | + rest | Peak (estimate) | Checkpoint |
|---|---|---|---|---|---|
| B-1M | 1,048,576 | 6.0 GiB | 4.5 GiB | ≈ 10.5 GiB (measured) | 1.7 GB |
| B-4M | 4,194,304 | 24.0 GiB | ≈ 4.8 GiB | ≈ 29 GiB | ≈ 6.5 GB |
| B-16M | 16,777,216 | 96.0 GiB | ≈ 5.5 GiB | ≈ 102 GiB | ≈ 26 GB |

B-16M only fits with the fused lazy Adam (kernel 3). The reference would additionally need up to 3 × 1.5 KiB per row
for the copies of the unread rows, at 16M rows with half of them unread ≈ 36 GiB, and that doesn't fit anymore.

**Run time and cost** (H200 SXM, Secure Cloud 4.59 $/h). Extrapolated from the RX 9070 (575 ms per step, H200 ≈ 7.5×
bandwidth, ≈ 5× compute; the small model doesn't saturate the H200), **uncertain up to a factor of 2**:

| | Duration | Cost (Secure) |
|---|---|---|
| Setup (Python, 11 GB of data + tokenising, tests, trial run) | 40–55 min | 3.10–4.20 $ |
| B-1M | 25–45 min | 1.90–3.40 $ |
| B-4M | 30–55 min | 2.30–4.20 $ |
| B-16M (Adam over 16M rows, top-k over 4096 keys) | 40–75 min | 3.10–5.70 $ |
| Fetching checkpoints (pod restarted, cheaper with 0 GPUs) | 20–60 min | 0–4.60 $ |
| **Total** | **≈ 2.5–4.5 h** | **≈ 11–22 $** (Community Cloud ≈ 9–17 $; + ≈ 1 $/day as long as the stopped pod isn't deleted) |

**Checked before the start (here, without an NVIDIA GPU):**

- **Data:** `cloud/fetch_data.py` in a fresh copy of the repo. All 45 raw files pass the SHA-256 check; all 7 token
  files and meta.json come out **byte-identical** (126 s). The download through the pinned Hugging Face URL is tested
  with one file, including the checksum.
- **Queue:** trial run with `SMLM_CLOUD_DRYRUN=1` (B-1M, 2 M tokens, Triton): run, GPU log every 10 s, checksum list
  and end all work. Outside a pod the stop script finds no pod key, the queue reports "stop FAILED" and sends the
  warning to stop by hand; so the error path is checked too (at the time with the IONOS script, the Runpod script
  checks the same thing).
- **GitHub:** private repo created, pushed. Cloning with the deploy key tested.
- **Tests:** 87 GPU tests green on ROCm, CPU interpreter green.
- **Added afterwards, only tested on the CPU** (the GPU wasn't released anymore after the measurement window):
  - Kernel 1 no longer sets the `touched` mask itself; that's the same pattern as the kernel-3 bug, now a PyTorch
    scatter.
  - New tests: selection with 4096 keys per half (B-16M) and offsets above 2³¹ elements (only GPUs ≥ 60 GB).
  - A trial run of all three configurations in `setup.sh`.
  - `git pull --rebase` before every push.

  CPU suite 42 green, CPU interpreter 21 green. On a GPU they run first in the cloud (setup step 6/7) or in the next
  time window at home.

**Not tested** (impossible without the machine). The setup checks each of these points before computing anything,
and stops the pod on an error:

- **Kernels on CUDA/H200:** the tests run there first, plus the trial run of all three configurations with 1 M tokens
  each. Only if both work does the queue start.
- **Runpod specifics:** template, `/workspace` volume and stopping with the pod key. `setup.sh` checks API access
  beforehand (`stop_pod.sh --check`) and warns if it doesn't work. That a pod key may stop its own pod is what the
  Runpod docs say ("Schedule a stop"), but it isn't tried.

**Ready for the cloud: yes.** Procedure in `docs/notes/CLOUD.md`: top up the account, create a pod, upload the kit,
start `setup.sh`.

**Execution (2026-10-03/04, Runpod pod `7yajarg09lrzdn`, 1 × H200 SXM, EUR-IS-4):**

- **First attempt:** 88 of 89 GPU tests green. The new large-table test for lazy Adam failed; the bug was in the
  test. It compared against a differently summed gradient, and with |g| ≈ 0 the sign flips in Adam. Fixed in
  `409dd49`. The pod stopped itself after the error as intended. Second attempt: GPU 89/89 and CPU interpreter 21/21
  green.
- **Trial run on the H200** (peak VRAM in training): B-1M 10.4 GiB, B-4M 28.5 GiB, B-16M 100.9 GiB, as estimated.
- **A single run is CPU-bound on the pod:** one Python thread at ≈ 90%, GPU ≈ 24% utilised, ≈ 81 k tok/s.
  - So from 22:01 UTC on, **parallel execution** (approved by the user): `scripts/run_cloud_parallel.py` takes over
    the running B-1M and starts B-16M right away on the same GPU.
  - B-4M starts only after B-1M and only once B-16M is finished or ≥ max(15 GB, 30.5 GiB) are free before that.
  - A crashed run is reported, not restarted; there are no intermediate checkpoints.
  - The old queue is frozen (SIGSTOP), not ended: closing its tmux window would have sent B-1M a SIGHUP.
- **Same training code and same settings.** The **speed values of the cloud runs (tok/s, train time) can't be
  compared because of the shared GPU**, neither with each other nor with the RX 9070.
- **Self-stop doesn't work:** the pod-scoped `RUNPOD_API_KEY` gets HTTP 403 from the Runpod REST API. The stop at the
  end goes through the MCP (Claude Code, with the user's account).

<!-- CLOUD-STATUS:BEGIN -->

**Cloud status** (automatic, `scripts/cloud_status.py`, as of 2026-10-04 04:28)

| Run | Status | GPU | Val PPL Wikipedia | Val PPL WikiText | Train time | tok/s | VRAM train | Usage | max. temp. GPU / memory | max. power |
|---|---|---|---|---|---|---|---|---|---|---|
| B-1M-sparse s0 at home (RX 9070, PyTorch reference) | done | AMD Radeon RX 9070 | 21.837 | 65.51 | 216 min | 38,517 | 10.5 GiB | 100.0% | 48 / 80 °C | 238 W |
| B-1M (cloud, control) | done | NVIDIA H200 | 21.837 | 65.87 | 138 min | 55,226 | 10.4 GiB | 100.0% | 41 / 40 °C | 343 W |
| B-4M (cloud) | done | NVIDIA H200 | 20.799 | 63.07 | 119 min | 77,551 | 28.5 GiB | 99.7% | 46 / 47 °C | 379 W |
| B-16M (cloud) | done | NVIDIA H200 | 19.960 | 59.97 | 155 min | 52,684 | 100.9 GiB | 91.3% | 47 / 48 °C | 404 W |

| Criterion (fixed beforehand) | Measured | Result |
|---|---|---|
| worth it: B-4M ≥ 3% better than B-1M (cloud) and B-16M better than B-4M | B-4M / B-1M = 0.9525 (−4.75%; limit 0.97); B-16M / B-4M = 0.9596 (−4.04%) | **worth it** |

Control: B-1M in the cloud (Triton kernels, H200) against at home (PyTorch reference, RX 9070): 21.837 against
21.837 (−0.00%; seed spread at home 0.39%).

<!-- CLOUD-STATUS:END -->

### Result of the cloud runs (evaluated by hand, 2026-10-04)

**Verdict by the pre-registered criteria: "worth it".** B-4M is 4.75% better than B-1M (cloud), ≥ 3% was required,
and B-16M is another 4.0% better than B-4M.

| | Entries | Val PPL Wikipedia | against B-1M | Val PPL WikiText | Usage (val) | Top-1% share | KL | VRAM train |
|---|---|---|---|---|---|---|---|---|
| B-1M-sparse s0 at home (RX 9070, PyTorch) | 1.05 M | 21.837 | | 65.51 | 100% | 11.7% | 0.62 | 10.5 GiB |
| **B-1M** (H200, Triton, control) | 1.05 M | **21.837** | – | 65.87 | 100% | 11.8% | 0.62 | 10.4 GiB |
| **B-4M** | 4.19 M | **20.799** | **−4.75%** | 63.07 (−4.3%) | 99.7% | 18.9% | 0.99 | 28.5 GiB |
| **B-16M** | 16.8 M | **19.960** | **−8.6%** (against B-4M −4.0%) | 59.97 (−9.0%) | 91.3% | 23.4% | 1.35 | 100.9 GiB |

Course, val PPL at equal tokens:

| Tokens | B-1M | B-4M | B-16M | B-4M / B-1M | B-16M / B-4M |
|---|---|---|---|---|---|
| 100 M | 39.18 | 38.12 | 37.36 | 0.973 | 0.980 |
| 200 M | 29.62 | 28.64 | 27.78 | 0.967 | 0.970 |
| 300 M | 25.46 | 24.43 | 23.65 | 0.959 | 0.968 |
| 400 M | 22.93 | 21.90 | 21.09 | 0.955 | 0.963 |
| 500 M | 21.84 | 20.80 | 19.96 | 0.953 | 0.960 |

**Discussion (no sugar-coating):**

- **Control passed:** B-1M in the cloud (H200, Triton kernels) and at home (RX 9070, PyTorch reference) are 0.00%
  apart (21.8365 against 21.8369). Kernels and hardware don't change the result, not even through the ties in the
  bf16 selection.
- **Clearly bigger than the seed noise:** the distances (4.75% and 4.0%) are about six times two seed spreads of
  B-1M-sparse (2 × 0.39%). But the seed spread comes from B-1M and is only carried over to the big tables (corrected
  2026-10-05 after the Codex review; before, it said "solid despite one seed"). On the never-trained WikiText val set
  they're just as big. It's still one seed per size.
- **The lead is still growing:** at 100 M tokens the 4× table brings 2.7%, at 500 M 4.7%; for B-16M against B-4M the
  values grow from 2.0% to 4.0%. With more data the distance would probably keep rising; that isn't measured.
- **The big tables are used more unevenly:** with B-16M, 8.7% of the 16.8 M entries were never read on the val set,
  and the most-read 1% get 23% of the reads (B-1M: 12%). Per entry, 500 M tokens are little for 16.8 M entries
  (B-16M: ≈ 11 k reads per entry, B-1M: ≈ 180 k).
- **Equal tokens, not equal cost:**
  - B-16M has 6.5 B parameters, 6.44 B of them in the table.
  - Per token B-16M computes 20% more than B-1M, B-4M 7% more (partial scores over 4096 and 2048 instead of 1024 keys
    per half: 56.5 / 50.2 / 47.1 M MACs).
  - But the table needs 26 GB in fp32, ≈ 3.3 GB with 4 bits, and ≈ 100 GiB of GPU memory in training (values,
    accumulator, Adam).
  - A comparison at equal compute time against A wasn't done for the big tables.
- **Speed values not comparable:** B-1M and B-16M shared the GPU for a while, later B-16M and B-4M. Train time, tok/s
  and the inference benchmarks in the `run-info.json` of the cloud runs therefore can't be compared.
- **Cost:** **25.51 $** for everything according to Runpod's billing (`list-billing`, queried on 04.10. after
  deleting the pod: GPU 24.98 $, disk 0.52 $), including the first attempt with the test bug and the disk cost until
  the deletion on 04.10. around 13:30.
  *Correction:* this first said 21.01 $ (GPU 20.89 $, disk 0.12 $). That value was too low. It came from a query
  shortly after the end of the runs, when the billing apparently wasn't complete yet. The pod run time of
  ≈ 5.4 h × 4.59 $/h fits the 24.98 $ of GPU cost.
- **Procedure:**
  - All checkpoints were fetched with `rsync` and checked against `runs/cloud/checkpoints.sha256`: 12 of 12 OK.
  - In the end the pod **did stop itself**: a POST stop with the pod key worked, even though reading returned 403.
    It was then started briefly once more to fetch the rest of the B-4M download, and then stopped via the MCP.
  - GPU maxima: 48 °C, 404 W.

## Step 1: what the table is worth – dense comparison models (plan fixed before the run, 2026-10-04)

**Question:** How big would a normal dense model without a table have to be to be as good as B-1M (21.84), B-4M
(20.80) and B-16M (19.96) at the same 500 M tokens? This is a **measurement without a pass criterion**.

**Models:** Llama style like A, without a memory layer. Width and depth grow together. `head_dim = 64`; the FFN width
is ≈ 8/3 · d, rounded to a multiple of 64 (as in A: 384 → 1024). Presets in `smlm/train.py`.

| Model | d | Layers | Heads | FFN | Params without emb. | Emb. | MACs/token forward | Micro-batch |
|---|---|---|---|---|---|---|---|---|
| A (existing, RX 9070) | 384 | 12 | 6 | 1024 | 21.24 M | 19.32 M | 45.3 M | 8 |
| D-50M | 640 | 10 | 10 | 1728 | 49.58 M | 32.19 M | 88.3 M | 8 |
| D-100M | 768 | 14 | 12 | 2048 | 99.11 M | 38.63 M | 148.7 M | 8 |
| D-200M | 1024 | 16 | 16 | 2752 | 202.41 M | 51.51 M | 270.7 M | 8 |
| D-400M | 1280 | 20 | 20 | 3456 | 396.55 M | 64.39 M | 487.1 M | 4 |
| *for comparison:* B-1M / B-4M / B-16M | 384 | 12 | 6 | 1024 | active 23.1 / 26.2 / 32.5 M | 19.32 M | 47.1 / 50.2 / 56.5 M | 4 |

**Training:** like A and B, without deviation.
- Data:
  - 500 M Wikipedia tokens, every sequence exactly once
  - data seed 1234, init seed 0
  - validation: Wikipedia val set (1.48 M tokens); WikiText-103 val only as a side value
- Optimisation:
  - AdamW (0.9 / 0.95), LR 6e-4, warmup 5%, cosine to 10%
  - weight decay 0.1, clip 1.0
  - 32,768 tokens/step, bf16 autocast
- The micro-batch is only adapted to the memory; gradient accumulation fills up to 32,768 tokens.
  `tests/test_dense.py` checks that this doesn't change the gradients.
- **A** enters as the smallest point with its existing run: seed 0, PPL 25.665, same data, same val set, computed at
  home. That home and cloud give the same result is shown by B-1M: 21.8369 against 21.8365.

**Evaluation (fixed):**
1. **Curve:** val PPL (Wikipedia) against parameters without embeddings, log–log. Points: A, D-50M, D-100M, D-200M,
   D-400M, as far as they ran.
2. **Equivalent size N_eq** for B-1M, B-4M and B-16M:
   - Main value: piecewise linear interpolation of log PPL over log N between the two neighbouring dense points.
   - Comparison value: fit PPL = E + a · N^(−α) over all dense points. Grid over α and E, a per grid point by least
     squares in PPL, the point with the smallest error in log PPL is chosen (corrected 2026-10-05 after the Codex
     review; before, it said "least squares in log PPL"; an exact log fit moves the fitted values by at most 0.3 M).
3. **Uncertainty** (a sensitivity range, not a confidence interval):
   - The PPL of B and that of the two neighbouring points are shifted by ±0.4%. That's the seed spread from stage 1c:
     A s0/s1 0.35%, B-1M s0/s1 0.39%.
   - The extreme cases give a range for N_eq.
   - If the fit value deviates more, the range is extended up to it.
4. **Bracketing:**
   - If B is better than the largest dense model that ran, only "> N_max" is reported. A fit extrapolation appears at
     most as a marked hint.
   - If B is worse than A, the result is "< 21 M".
5. **Compute per token:**
   - Forward MACs (`macs_per_token`, context 1024) for all models; training ≈ 3×.
   - For B also the table values read per token: 3 layers × 4 heads × 32 rows × 384 values = 147,456 values, so
     295 KB in bf16 or 74 KB in 4 bits.
   - Plus the size of the table.

**Execution (Runpod, 1 × H100 SXM 80 GB, Secure Cloud, 3.49 $/h; `cloud/setup_dense.sh`, `scripts/run_dense.py`):**
- **Data:** the token files come from the Hetzner storage box. They're checked by sha256 against
  `cloud/data_sha256.txt`, so byte-identical to the ones at home.
- **Tests:** `tests/test_dense.py` and `tests/test_stage1b.py`, on the GPU.
- **Trial run:** each size runs alone for 3 M tokens. Measured are tok/s, peak VRAM, evaluation, saving and inference.
- **Budget guard (cap 18 $ for the whole pod):**
  - Projection = pod run time + 0.1 h + 1.15 × Σ (500 M tokens + evaluation tokens / 3) / (tok/s alone) + 0.4 h.
  - Admitted in the order 50M, 100M, 200M, 400M, as long as the projection stays ≤ 18 $ / 3.49 $/h = 5.16 h.
  - Whatever drops out is reported. A bigger model is then not admitted anymore.
- **Parallel operation:** the admitted runs run at the same time, the biggest starts first.
- **Abort:**
  - From 5.16 h − 0.3 h of run time on, whatever is still running gets ended and saved.
  - An independent watchdog stops the pod at 5.16 h − 6 min.
  - Aborted or crashed runs are reported, not restarted (no intermediate checkpoints).
- **After every run:**
  - small files to GitHub
  - run folder including checkpoint via rsync to the storage box, checked there by sha256
  - ntfy message
- **End:**
  - checksum list pushed, everything saved and checked once more.
  - Then Claude stops and deletes the pod.
  - If saving fails, the pod is only stopped and a message goes out.
- **Measurements:** GPU temperature, power and clock every 10 s per run (`gpu_thermal.csv`). Because the runs share
  the card, these files show the whole GPU; tok/s and train time are **not** comparable with single runs.

**Limitations (known beforehand):**
- **Token budget:** 500 M tokens is little for 200–400 M parameters (Chinchilla-optimal would be ≈ 20 tokens per
  parameter). N_eq applies **only to this token budget**.
- **Learning rate:** the LR schedule was chosen for A and isn't tuned per size. Bigger dense models would probably be
  a bit better with an adapted LR; that rather makes the table look **too good**.
- **Seeds:** one seed per dense size.
- **Instability:** if a big model gets unstable at LR 6e-4 (NaN aborts; a loss explosion without NaN is visible in
  the curves), that's reported, not repeated.

## Step 2: B-16M at home – the table doesn't have to be in VRAM (measured 2026-10-04)

**Question:** Can B-16M (table 16.8 M rows × 384) be run on a normal PC when the table isn't in the expensive VRAM?
Measurement without a pass criterion.

**Machine:**
- GPU: RX 9070 (16 GB)
- CPU: Ryzen 9 5900XT
- RAM: 125 GB
- NVMe: Samsung 990 PRO (`/home`)
- The desktop was running with ≈ 0.6–1 GB of VRAM. Nothing else was running.

**Preparation:**
- `scripts/convert_table.py` splits the cloud checkpoint into the small rest (`rest.pt`, 71 M parameters) and the
  table as flat files: bf16 12.9 GB, 4 bit 3.2 GB plus 32 MB of scales.
- The 4-bit quantisation on the CPU is bit-identical with `quantize_q4` on the GPU (checked on 1 M rows).

**Variants** (`smlm/offload.py`, `scripts/bench_offload.py`, one process per variant):
- **a) Table in VRAM:** bf16 or 4 bit. The kernel reads directly; decode graphs are possible.
- **b) Table in RAM:** bf16, fp32 or 4 bit.
  - Per memory layer and call, the indices of the rows read go to the CPU.
  - The CPU gathers the rows into a pinned buffer, from there they go to the GPU.
  - There the same kernel as in a) runs on the compact table.
  - That's three round trips per token. Decode graphs aren't possible because of that.
- **c) 4-bit file on the NVMe:**
  - Access via mmap; readahead is off (`MADV_RANDOM`).
  - Missing rows are requested together per call with `MADV_WILLNEED`.
  - RAM cache: a fixed set of the x% rows read most in training, optionally plus a FIFO part for rows missed
    recently.
  - **Cold:** the file is dropped from the page cache beforehand without root, with `posix_fadvise(DONTNEED)`;
    `fincore` confirms 0 bytes.
  - **"Limited":** the process runs in a systemd scope with `MemoryMax=4G`. So Linux can't keep the 3.2 GB file
    completely in the page cache. Measured, the scope holds 1.4–2.0 GB of the file, `memory.current` stays at 4.0 GB.

**Measurements:**
- **Writing (batch 1):** greedy, 128-token prompt + 256 new tokens, three different prompts; the first one is cold
  for c).
- **Reading a prompt:** 4 × 1024 tokens of val text, median over 3 batches after a warm-up batch.
- **Val PPL:** on the whole Wikipedia val set (as at the end of training, batch 1) and on the first 64 windows (for
  the bit comparison).

**Results** (`report/offload/*.json`, `report/offload_summary.json`, plot `report/offload_cache.png`):

| Variant | Writing tok/s (ms/token, 1st prompt) | Reading tok/s | Cache hits writing / reading | NVMe while reading | VRAM used (total, with desktop; GiB) | RAM peak (RSS; GiB) | Val PPL whole |
|---|---|---|---|---|---|---|---|
| a) bf16 in VRAM, with graphs | 216 (4.6) | 108,000 | – | – | 13.1 | 12.9¹ | 19.9615 |
| a) bf16, without graphs | 171 (5.8) | 108,000 | – | – | | | |
| a) 4 bit in VRAM, with graphs | 212 (4.7) | 107,500 | – | – | 3.6 | 4.0 | 19.9795 |
| a) 4 bit, without graphs | 173 (5.8) | 107,400 | – | – | | | |
| b) bf16 in RAM | 139 (7.2) | 31,200 | – | – | 0.53 | 14.7 | 19.9615 |
| b) fp32 in RAM | 143 (7.0) | 18,900 | – | – | 0.53 | 48.8¹ | **19.9601** |
| b) 4 bit in RAM | 154 (6.5) | 61,700 | – | – | 0.53 | 5.0 | = a) 4 bit² |
| c) NVMe, no cache | 114 (8.7) | 4,900 | 0 / 0% | 57 k reads/s, 235 MiB/s | 0.53 | 5.1³ | = a) 4 bit² |
| c) NVMe, cache 5% (155 MB) | 120 (8.3) | 4,500 | 38 / 28% | 75 k/s | 0.53 | 5.3³ | = a) 4 bit² |
| c) NVMe, cache 10% (310 MB) | 127 (7.8) | 4,400 | 53 / 41% | 86 k/s | 0.53 | 5.5³ | = a) 4 bit² |
| c) cache 10% + FIFO 20% | 127 (7.8) | 4,500 | 73 / 52% | 89 k/s | 0.53 | 6.1³ | = a) 4 bit² |
| c) NVMe, cache 30% (930 MB) | 137 (7.3) | 4,800 | 80 / 72% | 110 k/s, 469 MiB/s | 0.53 | 6.1³ | 19.9795 |
| c) NVMe, cache 50% (1.55 GB) | 138 (7.2) | 6,500 | 92 / 87% | 111 k/s | 0.53 | 6.5³ | = a) 4 bit² |
| c) cache 10%, **RAM limited to 4 GiB** | 124 (8.0) | **1,500** | 53 / 41% | 65 k/s | 0.53 | ≤ 4 (scope)⁴ | = a) 4 bit² |
| c) cache 30%, **RAM limited to 4 GiB** | 133 (7.5) | **2,600** | 80 / 72% | 84 k/s | 0.53 | ≤ 4 (scope)⁴ | = a) 4 bit² |

¹ While loading: the file or checkpoint is read completely once. For b) fp32 the mapped checkpoint pages are counted
too; the table itself takes 25.8 GB.
² On the 64 comparison windows **bit-identical** to a) 4 bit: same NLL sum 205849.488. The whole val set was only
computed for c) with 30% and also gives exactly 19.9795.
³ RSS includes the mapped pages of the 4-bit file. Without a RAM limit, 2–3 GB of the file were in Linux's page cache
at the end.
⁴ Limit of the systemd scope (cgroup `MemoryMax=4G`); what counts is what the kernel charges to this group. That's
not the same as "runs on a machine with 4 GB of RAM": drivers, the desktop and the rest of the page cache are
outside. The process RSS was higher (up to ≈ 5.9 GiB), because it counts mapped file pages (corrected 2026-10-05
after the Codex review; the units in this table are GiB, before it said "GB").

**What this shows:**
- **Quality: the same.**
  - b) and c) compute bit-identically to a) at the same precision: bf16 19.9615 in a) and b); 4 bit 19.9795 in a), b)
    and c).
  - Against fp32 from the cloud (19.9600), bf16 costs +0.007% and 4 bit +0.10%.
  - fp32 from RAM matches the cloud value to 0.001% (19.9601). The rest is the different GPU.
- **Writing word by word works without the table in VRAM:**
  - From RAM: 139–154 tok/s. From the NVMe: 114–138 tok/s, cold and with only 4 GB of RAM 124–133 tok/s.
  - With the table in VRAM: 171–173 tok/s without graphs, 212–216 with graphs.
  - The distance comes mostly from the three CPU round trips per token, not from the NVMe: even without any cache,
    c) loses only 18% against RAM (114 against 139 tok/s).
  - VRAM drops from 13.1 or 3.6 GB to 0.53 GB.
- **Reading long texts is only fast from VRAM:**

  | Where the table is | Reading tok/s |
  |---|---|
  | VRAM | 108,000 |
  | RAM | 19,000–62,000 (depending on bytes per row) |
  | NVMe | 4,400–6,500 |
  | NVMe, RAM limited to 4 GB | 1,500–2,600 |

  While reading, every token needs ≈ 270 different rows. Every missed row costs a 4 KB page, so 21 times more data
  than needed. The NVMe delivers 57,000–111,000 reads/s, and that isn't enough.
- **Cache:** the hit rates are as predicted from the training reads (30% cache → 80% hits when writing, predicted
  79%). A FIFO part raises the hits, but not the speed.

**Limitations:**
- **Scope:** each variant was measured once, with three prompts of 256 tokens and three reading batches.
- **Implementation:** the read paths are written in Python/NumPy. A C/io_uring path, or a row layout that puts rows
  read together onto the same page, could speed up reading from the NVMe a lot; that isn't measured.
- **What "cold" means:** cold only refers to the page cache. The program fills the fixed RAM cache at start; that
  took 0.7–8 s.
- **Memory needs while loading:** the RAM peaks of a) bf16 and b) fp32 include the one-time read.

## Step 3: a table as add-on memory for Qwen3.5-0.8B (prepared; criteria for approval, 2026-10-04)

**None of this is trained.** This lists setup, data, baseline measurements, the proposed criteria and the estimate.
Training starts only after the criteria are approved.

**Base:**
- **Model:** `Qwen/Qwen3.5-0.8B`, revision `2fc06364715b967f1860aea9cf38778875588b17`, released on 02.03.2026.
- **License:** **Apache 2.0** according to the model card and the `LICENSE` in the repo (sha256 `bbedc3fd…e57a`,
  standard Apache 2.0 text).
- **Variant:** the post-trained, multimodal version; only the text part is used (`Qwen3_5ForCausalLM`). There is
  also `Qwen3.5-0.8B-Base`.
- **Text part:** 752.4 M parameters; of them 254 M embedding, tied to the output.
  - 24 blocks: 18 × Gated DeltaNet (linear attention), 6 × gated attention
  - d = 1024, FFN 3584
  - vocabulary 248,320
- **Environment:** its own Python environment `.venv-qwen` with transformers 5.18.0, lm-eval 0.4.13 and accelerate.
  The existing `.venv` stays unchanged.

**Integration** (`smlm/qwen_memory.py`, `tests/test_qwen_memory.py`):
- **Position:** behind blocks 6, 12 and 18 one **additional** block each, via forward hooks; Qwen's module tree stays
  unchanged.
- **Formula:** h ← h + g · M(RMSNorm(h)). g is a scalar per block and starts at 0; the RMSNorm has no parameters.
- **M (Q+T):** memory layer as in B.
  - One shared table with 1024² = 1,048,576 rows × 1024 (1.07 B values).
  - Per block: 4 heads, top-32, query projection 1024 → 4 × 256 with BatchNorm, sub-keys 4 × 2 × 1024 × 128.
  - The swilu projections (Memory+) 1024 × 1024, twice per block, belong to the memory layer as in B. **Please
    confirm** that they may be trained as part of the memory layer. Alternative: without swilu, then really only the
    table, the search and the gates train.
  - Table with row-wise gradients and lazy Adam as in B (Triton kernel).
  - Trainable: ≈ 1.09 B parameters, of them 1.07 B table.
- **Control Q+D:** a SwiGLU block with width 1408 at the same positions.
  - Same MACs per token as a memory block (4.33 M), same gate, same data and steps.
  - Trainable 13 M parameters.
- **Tests** (CPU, small random Qwen configuration; all green):
  - At g = 0 the logits are **bit-identical** to Qwen alone, in train and in eval mode.
  - In the first step only g moves, from the second on the blocks too.
  - Frozen weights stay unchanged.
  - The MACs of the control match.

**Data** (`scripts/prepare_qwen_data.py`, Qwen tokenizer, uint32; checked by sha256 on the storage box):
- **New articles:** from the enwiki dump of 01.09.2026, only the last part files (page ids ≥ 77.5 M). Main namespace,
  no redirects and disambiguations, ≥ 300 characters of plain text (mwparserfromhell).
  - Creation month via page-id thresholds from the creation log of the Wikipedia API, stored in
    `page_id_months.json`.
  - 286,766 articles, created from 01/2025 on.

| Part | Articles | Tokens | Purpose |
|---|---|---|---|
| `train_new` | 79,864 | 55.3 M | training: created 03–08/2026, after Qwen's release |
| `val_new` | 1,500 | 1.02 M | **decisive:** held-out new articles |
| `mem_probe` | 1,000 | 0.66 M | part of the training: how much the table stores |
| `val_known` | 1,917 | 1.54 M | val set of stage 1b (Wikipedia 2023, HF preprocessing) |
| `val_known_same` | 1,500 | 1.62 M | old articles (page ids 4.0–5.4 M, ≈ 2006) from the same 2026 dump, prepared like the new ones |
| `curve_YYYY-MM` | 150 each | 0.08–0.19 M each | cut-off curve 01/2025–08/2026 (never trained) |

**Baseline measurement: Qwen alone, at home** (`runs/qwen/Q-base`, `report/qwen/cutoff_curve.json` and `.png`):
- **Token PPL, window 2048:**

  | Set | PPL |
  |---|---|
  | `val_new` | 12.98 |
  | `mem_probe` | 13.36 |
  | `val_known` | 13.94 |

- **Median PPL per article** (first 1024 tokens, 90% bootstrap interval):

  | Set | Median PPL | Interval |
  |---|---|---|
  | New articles per month, 2025–2026 | 10.9–12.8 | |
  | `val_new` | 11.47 | [10.63; 12.07] |
  | `val_known` | 13.07 | |
  | `val_known_same` | 13.72 | [13.24; 14.53] |

- **Honest conclusion:**
  - **A knowledge cut-off isn't visible.** Articles from after Qwen's release aren't harder for Qwen than those from
    2025, and they're *easier* than old articles prepared the same way.
  - For a 0.8B model the Wikipedia PPL mostly measures language and style (new articles are shorter and more
    uniform), hardly factual knowledge.
  - So a gain on `val_new` isn't automatically "new knowledge". It can also be adaptation to Wikipedia style, and
    that's exactly what the control run Q+D measures too.
  - In addition, the PPL is reported over "knowledge tokens" only: digits and capitalised words that aren't at the
    start of a sentence; that's ≈ 29% of the tokens.
- **Standard benchmarks at home:** not possible. Qwen3.5 crashes reproducibly under ROCm on the RX 9070 ("illegal
  instruction" in lm-eval, "memory access fault" with gradient checkpointing). So the standard benchmarks run for Q,
  Q+T and Q+D on the same cloud GPU.

**Training (proposal):**
- **Scope:** 2 passes over `train_new` (110.5 M tokens), sequence length 2048, 16 sequences per step (32,768 tokens,
  3,373 steps).
- **Optimisation:** as B: LR 6e-4, table 2.4e-3, warmup 5%, cosine to 10%, weight decay 0.1, clip 1.0, bf16. Qwen
  stays frozen in bf16.
- **Loss:** chunked over the 248k vocabulary, recomputed in the backward step.
- **Runs:** 1 seed each for Q+T and Q+D (`scripts/train_qwen_memory.py`).

**Criteria (proposal for approval; Q = Qwen alone, measured on the same GPU):**

*Does it help?* The token PPL on `val_new` decides:

| Verdict | Condition |
|---|---|
| **helps clearly** | PPL(Q+T) ≤ 0.95 × PPL(Q) **and** PPL(Q+T) ≤ 0.98 × PPL(Q+D) |
| **helps a bit** | PPL(Q+T) ≤ 0.98 × PPL(Q), but not "clearly" |
| **doesn't help** | otherwise |

Only reported:
- knowledge-token PPL on `val_new`
- `mem_probe` (stored knowledge)
- `val_known_same`
- the month curve

*Does it harm?* "Doesn't harm" requires all three points, each for Q+T (and Q+D) against Q:
1. **Standard benchmark** (lm-eval 0.4.13, zero-shot, the first 500 examples per task, MMLU per subject; tasks MMLU,
   ARC-Easy, ARC-Challenge, HellaSwag, PIQA, WinoGrande):
   - The mean of the six accuracies drops by at most 1.0 percentage points.
   - No task drops by more than max(2 pp, 2 × standard error).
2. **Known texts:** PPL on `val_known` and `val_known_same` at most +1%.
3. **Chat:** 12 fixed questions (`scripts/eval_qwen_general.py`, 6 German, 6 English), chat template without
   thinking mode, greedy, 200 tokens.
   - Per question two answers, blinded in random order; you judge better / same / worse.
   - "Harms" if Q+T is worse on more than 3 of 12 questions.

**Estimate:**
- **At home** (probe with 30 steps, discarded; sequence 2048, micro-batch 1):

  | Variant | Speed | Memory |
  |---|---|---|
  | Qwen + memory (65 k rows) | 2,990 tok/s | 10.7 GiB |
  | Qwen + dense block | 3,060 tok/s | 9.6 GiB |

  - The planned table needs ≈ 17 GB with Adam states and doesn't fit next to Qwen into 16 GB.
  - Two runs with a small table would take ≈ 20 h at home, with an unstable ROCm.
  - **Not recommended.**
- **H100 SXM (3.49 $/h):**
  - Memory ≈ 45 GB: table with optimizer 17 GB, Qwen, activations at micro-batch 4.
  - Speed uncertain: 25,000–60,000 tok/s, depending on whether the fast DeltaNet kernels (flash-linear-attention)
    run. That gives 0.5–1.2 h per run.
  - With setup, Q measurements, two runs and all evaluations 1.7–3.2 h ≈ 6–11 $ (5.30–9.90 €).
  - After step 1 (≈ 17–18 $) ≈ 16 $ remain; that's enough.
- **Still to build after the approval:** the cloud procedure for step 3 (setup as in step 1, data from the box,
  results and table back to the box) and the small blinding script for the chat answers.

### Step 3: approval and fact test (2026-10-04, before any training)

**Approval:**
- The criteria above are approved in principle; the PPL and "does it harm?" criteria apply unchanged.
- The swilu projections are trained too.
- Training happens in the cloud on an H100.
- **Addition (specification, paraphrased):**
  - The held-out articles only test whether the table helps in general.
  - Whether it plants knowledge is only shown by a fact test on the *training articles*.
  - Plus the same kind of fill-in-the-blank items from the held-out articles as a counter-check.
  - Q, Q+T and Q+D are compared.

**Fact test** (`scripts/make_fact_cloze.py` → `data/qwen_fact_cloze.jsonl`; scoring `scripts/eval_fact_cloze.py`):
- **Scope:** 500 blanks from training articles (`train_new`) and 500 from held-out articles (`val_new`), one blank per
  article, articles chosen at random by a fixed hash.
- **Mix:** 40% names (2–4 capitalised words), 30% dates/years, 30% other numbers (≥ 2 digits; only day numbers after a
  month name or counted amounts before a lower-case word).
- **Blank:** title + empty line + the sentence with the fact, cut off right before the fact.
- **Filter:**
  - The answer isn't in the prompt and not in the title.
  - The sentence start before the blank is ≥ 5 words; the fact isn't at the start of the sentence.
  - The years 2025 and 2026 are excluded, because they can almost always be guessed in new articles.
- **Scoring:** greedy continuation (up to 16 tokens, without chat template). Correct is an exact hit at the start of
  the continuation, followed by no letter and no digit.
- **Evaluation:** accuracy per part and type with a 95% Wilson interval; two models are compared pairwise over the
  same blanks.

**Main criterion "knowledge planted":**
- accuracy(Q+T) − accuracy(Q+D) on the training blanks ≥ **10 percentage points**,
- **and** on the counter-check Q+T isn't worse than Q+D.
- Operationalised, "not worse" means: at most 2 percentage points less. That's within the random noise of a pairwise
  comparison with 500 blanks. **(Please confirm.)**
- Q alone is reported too.

**Go for the start (2026-10-04, ≈ 23:30):**
- The tolerance "not worse = at most 2 percentage points below Q+D" is confirmed.
- The start is approved.
- **Procedure:**
  - A separate H100 pod, only after the step-1 pod is backed up and deleted; two cost caps at the same time could
    together exceed the balance.
  - Cost cap 13 $: balance after step 1 ≈ 16 $ minus a buffer.
  - Baseline Q, measured at home (`report/qwen/facts_Q_home.json`): training blanks 5.0% [3.4; 7.3], counter-check
    3.8% [2.4; 5.9]. What counts for the decision is the measurement on the same cloud GPU.

### Result step 1 (2026-10-05, Runpod H100, pod 4.92 h ≈ 17.20 $)

**Procedure:**
- **Budget guard:** it first left out D-400M as specified (projection 5.44 h > 5.16 h at 18 $). After the user's
  approval the cap was raised to 22 $; D-400M joined from pod hour 0.7 on.
- **Run time:** ≈ 4.9 h in total, exactly as long as the cautious projection. Four runs at once on one GPU weren't
  faster than one after another; the efficiency was ≈ 0.9.
- **Setup:** because of a harmless rsync permission error the setup first fell back to rebuilding the data from
  Hugging Face. I ended that after checking the box data (7/7 sha256 OK); that cost ≈ 10 min. The bug is fixed for
  future runs (`rsync -rt`).
- **Backup:** all four checkpoints are on the storage box and at home, checked by sha256 (4/4). The pod is deleted.

| Model | Params without emb. | Val PPL Wikipedia (decisive) | Val PPL WikiText-103 (side value) | MACs/token forward |
|---|---|---|---|---|
| A | 21.2 M | 25.665 | 78.38 | 45.3 M |
| D-50M | 49.6 M | 22.452 | 65.37 | 88.3 M |
| D-100M | 99.1 M | 20.270 | 59.66 | 148.7 M |
| D-200M | 202.4 M | 18.758 | 51.63 | 270.7 M |
| D-400M | 396.5 M | 17.598 | 47.16 | 487.1 M |

**Fit:** PPL = 13.60 + 7,361 · N^(−0.380), RMSE in log PPL 0.0024. The five dense points lie very smoothly on a
curve.

**Equivalent dense size** (rules as fixed above; `scripts/dense_equiv.py`, `report/dense_equiv.json`):

| | Val PPL | **equivalent dense size** | Sensitivity range (±0.4%, incl. fit) | Fit alone | MACs/token | Params active per token / table |
|---|---|---|---|---|---|---|
| B-1M | 21.837 | **60 M** | 57–63 M | 58 M | 47.1 M | 23.1 M / 0.40 B |
| B-4M | 20.799 | **83 M** | 79–88 M | 83 M | 50.2 M | 26.2 M / 1.61 B |
| B-16M | 19.960 | **114 M** | 106–123 M | 116 M | 56.5 M | 32.5 M / 6.44 B |

All three lie inside the measured range; no extrapolation was needed.

![What the table is worth](report/dense_equiv.png)

**What that means:**
- **B-16M** is as good as a dense model with ≈ 114 M parameters, so a good five times its compute core (21 M). But
  per token it computes only 56.5 M MACs; the equivalent dense model would need ≈ 165 M, so ≈ 2.9× as much.
- **Every quadrupling of the table** brings ≈ 1.4× equivalent size (60 → 83 → 114 M). The gain per doubling stays
  about the same in this range and isn't flattening yet.
- **The price for it is memory:** B-16M's table has 6.44 B parameters, 56 times as many as the equivalent dense model
  (114 M; corrected 2026-10-05 after the Codex review, before it said "thirty times"). In training B-16M needed
  ≈ 101 GB of GPU memory, D-200M 16 GB. For writing, though, the table doesn't have to be in VRAM (step 2).

**Side value WikiText-103, not fixed beforehand as a criterion:**
- On this differently formatted set the advantage is smaller. Interpolated log-log, B-1M corresponds to ≈ 48 M, B-4M
  to ≈ 65 M and B-16M to ≈ 95 M.
- So the table helps more on text like the training material (Wikipedia articles) than on a different preparation of
  the same source.

**Limitations** (as named beforehand):
- One seed per dense size.
- 500 M tokens is little for 200–400 M parameters; N_eq applies only to this token budget.
- The LR isn't tuned per size. That rather makes the table look too good.
- Speed values not comparable: the runs shared the GPU.

### Result step 3 (2026-10-05, Runpod H100; evaluation `scripts/qwen_step3_eval.py` → `report/qwen/step3_summary.json`)

**Procedure, honestly:**
- **Two failed setups** (≈ 1.40 $):
  - The CPU tests called the GPU-only kernels of flash-linear-attention.
  - After that, fla 0.5.2 with Triton 3.6 refused the backward step on Hopper GPUs (known bug). Fixed with Triton
    3.7.1; checked with the unit tests and the test on the real model with gate 0: bit-identical, training runs.
- **My own mistake, ≈ 7.40 $:** a typo from an untested change to `run_dense.py` made the queue crash at the start.
  The pod idled for ≈ 2.1 h, because my watcher only looked at log lines.
  - Fixed: before the start all scripts are compiled; 90 s after the start it's checked whether the queue is alive;
    the watcher reports a dead queue.
- **Cap raised:** with approval from 13 $ to 16.50 $ for the actual run, after the speed probes projected
  13.50–15.70 $.
- **Q+T's standard benchmark failed at first:** lm-eval switches autocast off around the model call, and the fp32
  add-on blocks then didn't match Qwen. Fixed; Q was repeated with the same setting and gave exactly the same values.
- **Pod running:** 3.92 h ≈ 13.70 $.
- **Step 3 in total:** ≈ 22.50 $ (of which ≈ 7.40 $ idle time through my mistake).
- **Backup:** everything on the storage box and at home, checked by sha256. The pod is deleted.

**Speed and memory (H100):**

| Run | Training | Memory | Duration (110.5 M tokens) |
|---|---|---|---|
| Q+T | 32,600 tok/s | 33 GB | 66 min |
| Q+D | 35,800 tok/s | 17 GB | 64 min |

Q's PPL is identical on the H100 (with fla kernels) and at home (PyTorch path) (12.977).

**PPL** (token PPL, window 2048):

| Set | Q (Qwen alone) | Q+T (table) | Q+D (dense control) |
|---|---|---|---|
| `val_new`: held-out new articles (decisive) | 12.977 | 10.093 (−22.2%) | **10.014 (−22.8%)** |
| `val_new`, knowledge tokens only | 16.32 | 12.57 | 12.73 |
| `val_known`: val set 2023, HF preprocessing | 13.935 | **14.904 (+7.0%)** | 13.730 (−1.5%) |
| `val_known_same`: old articles, same preprocessing | 14.282 | 14.029 (−1.8%) | 12.930 (−9.5%) |
| `mem_probe`: trained articles | 13.362 | **5.638 (−58%)** | 9.382 (−30%) |

**Standard benchmark** (zero-shot, accuracy in %, ± standard error from lm-eval):

| Task | Q | Q+T | Q+D |
|---|---|---|---|
| MMLU | 49.7 ± 0.4 | **47.0 (−2.7; limit −2.0)** | 49.6 (−0.1) |
| ARC-Easy | 63.2 ± 2.2 | 65.8 (+2.6) | 65.0 (+1.8) |
| ARC-Challenge | 31.0 ± 2.1 | 38.0 (+7.0) | 37.0 (+6.0) |
| HellaSwag | 42.4 ± 2.2 | 41.6 (−0.8) | 42.0 (−0.4) |
| PIQA | 70.2 ± 2.0 | 71.2 (+1.0) | 69.2 (−1.0) |
| WinoGrande | 56.2 ± 2.2 | 58.8 (+2.6) | 57.0 (+0.8) |
| Mean | 52.1 | 53.7 (+1.6) | 53.3 (+1.2) |

The limit per task is max(2 pp, 2 × standard error of the difference). The standard error of the difference is
√(se_Q² + se_X²); that's how I implemented "2 × standard error".

**Fact test** (exact hit on the first try; 500 blanks per part):

| | Q | Q+T | Q+D | Q+T − Q+D (95% bootstrap, paired) |
|---|---|---|---|---|
| Training blanks | 4.8% | 10.2% | 7.8% | **+2.4 pp [0.4; 4.6]** |
| Counter-check (never seen) | 4.0% | 10.2% | 8.0% | +2.2 pp [0.2; 4.2] |

**Verdicts by the pre-registered criteria:**

| Criterion | Result |
|---|---|
| **Does it help?** (PPL `val_new`) | **"helps a bit".** Q+T is 22% better than Q, but **not better than the dense control** (Q+T/Q+D = 1.008). For "helps clearly" it would have had to be ≤ 0.98. |
| **Does it harm? Q+T** | **"harms"**, and the chat part can't change that anymore. MMLU drops by 2.7 pp (limit 2.0), and the PPL on `val_known` rises by 7.0% (limit 1%). The mean of the six tasks rises, though (+1.6 pp). |
| **Does it harm? Q+D** | "doesn't harm" by the standard benchmark and known texts; the blinded chat comparison is still pending. |
| **Knowledge planted?** | **Not reached.** Q+T gets only 2.4 pp more training facts right than Q+D (≥ 10 required). On the counter-check the lead is just as big (+2.2 pp); so the table makes completion a bit better in general, not specifically for the facts it saw. |

**What that means:**
- **No advantage over an equally expensive add-on block:** as an add-on to a finished, frozen Qwen, the table in this
  setup brings no advantage over a small dense block with the same compute. Both adapt Qwen to new Wikipedia texts
  equally well.
- **No sign of planted facts:** the add-on with table adapts very strongly to the training articles as text (PPL on
  trained articles −58%, dense control −30%). But in the fact test it gets equally better on training and
  counter-check articles. So targeted retrieval of trained facts isn't shown. Whether what was learned sits in the
  table or in keys, projections and gates I haven't tested (corrected 2026-10-05 after the Codex review; before it
  said "stored, but not retrievable"; tested since then, see the addendum to step 3 at the end).
- **Side effects of the table:**
  - MMLU (knowledge questions) drops.
  - On the differently prepared 2023 val set Qwen gets worse.
  - Both happen mainly in the 2nd pass; the intermediate values at 40 M tokens were better
    (`runs/qwen_cloud/QT-s0/metrics.csv`).
  - The dense control doesn't show these side effects.

**Limitations:**
- **Scope:** one seed per variant, one table size (1 M rows), one position (behind blocks 6, 12, 18), one LR schedule
  (as B, not tuned for Qwen) and 2 passes.
- **Chat part:** still pending.
- **Fact test:**
  - An exact hit on the first try is strict.
  - Many blanks allow several correct continuations, even for a human.
  - A small effect can disappear under that; a lead of 10 pp would have had to be visible, though.
- **Knowledge cut-off:** none was visible in the PPL (see above). With the 0.8B Qwen, "new knowledge" is hard to
  separate from style.

## AMD Instinct MI350X: tests and speed on AMD's data-centre GPU (2026-10-05)

**Question:** Do the code and the Triton kernels run unchanged and correctly on an AMD data-centre GPU? How fast?

**Setup:**
- **GPU:** Runpod, 1× AMD Instinct **MI350X** (gfx950, 288 GB), Secure Cloud, 5.49 $/h. An MI300X (2.39 $/h) was
  planned; it wasn't available, the MI350X was taken with approval.
- **Run time and cost:** 0.6 h ≈ 3.30 $.
- **Software:** PyTorch 2.13.0+rocm7.1, HIP 7.1, Triton 3.7.1, Python 3.12.
- **Procedure:** `cloud/setup_amd.sh`, `scripts/run_amd.py`, evaluation `scripts/amd_summary.py`.
- **Code:** it was copied up unchanged, without GitHub (the history was being cleaned up in parallel).
- **Results:** in `runs/amd_mi350x/`, backed up on the storage box and checked by sha256 at home (62/62 files equal).

**Tests:** **107 of 107 passed** (10.6 min), including:
- all kernel tests with tables of 262k / 1M / 4M rows
- selection with 4096 keys per half
- tables with more than 2³¹ elements
- decode graphs and lazy Adam
- the table outside the GPU

**Speed:**
- **Measurement setup:** each model alone, 3 M tokens, same arguments as the H100 trial run of step 1.
- **H100 reference values:** from `runs/cloud_dense_preflight`.
- **RX 9070:** from the optimisation (section "Result of the optimisation"); a different kind of measurement, only for
  orientation.

| Model | Training tok/s | H100 | Memory peak | Writing (batch 1) tok/s | H100 | Reading tok/s | H100 |
|---|---|---|---|---|---|---|---|
| A | 164,900 | – | 7.9 GiB | 266 | – | 2,390,000 | – |
| D-100M | 155,900 | 208,300 | 11.8 GiB | 231 | 192 | 1,171,000 | 844,000 |
| D-400M | 71,800 | 83,000 | 15.3 GiB | 155 | 126 | 497,500 | 332,900 |
| B-1M, PyTorch reference | 77,100 | – | 11.6 GiB | 185 | – | 737,700 | – |
| **B-1M, Triton kernels** | **125,900** | – | 10.5 GiB | 210 | – | 1,387,000 | – |
| B-4M, Triton kernels | 119,400 | – | 28.6 GiB | 216 | – | 1,094,000 | – |
| **B-16M, Triton kernels** | **112,800** | – | **101.0 GiB** | 230 | – | 764,100 | – |

- **The kernels help even more on the MI350X than at home:**
  - training 1.63×, reading 1.88×, writing 1.14× against the PyTorch reference.
  - Same result: val PPL after 3 M tokens 1168.46 against 1168.63.
  - B-1M reaches 76% of the training speed of the model without a table; on the RX 9070 it was 62%.
- **B-16M fits completely on one card:** 101 GiB, 113,000 tok/s. A 500 M-token run as in the cloud would take ≈ 75 min
  alone on one MI350X, ≈ 7 $.
- **Against the H100**, same plain PyTorch code, not tuned for either card:
  - Training dense models is slower on the MI350X (0.75× and 0.87×).
  - Writing (1.2×) and reading (1.4–1.5×) are faster.

**Same results on three GPUs:**
- **Comparison run:** B-1M with Triton kernels, the first 20 M tokens of the 500 M schedule, seed 0, on the MI350X.
  The same run exists from the RX 9070 (`runs/kernel_check/triton_500msched`) and the H200 (`runs/cloud/B-1M-s0`).
- **Result:** val PPL at 20 M tokens: MI350X 210.11, RX 9070 208.72, H200 210.75. At 22 M: 184.81 against 184.56
  (RX 9070).
- **Context:** differences of up to ≈ 1% in this early phase are the known noise from ties at the top-k boundary and
  non-deterministic sums (cf. optimisation: over 500 M tokens 0.002%).
- **Dense models:** D-100M and D-400M reach the same PPL on MI350X and H100 after 3 M tokens, to 0.06%.

![B-1M on three GPUs](report/amd_crosscheck.png)

**Limitations:**
- One short run each. The speed values are indicative and weren't repeated.
- PyTorch reports the card as "AMD Radeon Graphics" (gfx950).
- On the MI350X the GPU log recorded temperature (≈ 62 °C junction) and clock, but no power. The memory value in the
  log is implausible and not used.

### Assessment of the kernels (2026-10-05)

**What's strong:**
- **Portable:** the same Triton code computes the same on three fundamentally different GPU architectures: Radeon
  RX 9070 (RDNA4, consumer), Instinct MI350X (CDNA4, data centre) and H100/H200 (Hopper).
- **Tests:** on the MI350X all 107 tests pass without changes.
- **Speed** against my own PyTorch reference:

  | | RX 9070 | MI350X |
  |---|---|---|
  | Training | 1.47× | 1.63× |
  | Reading | 1.79× | 1.88× |

- **Memory:** the lazy-Adam kernel works in place. Only with it does B-16M fit into 101 GB.

**What the kernels aren't:**
- **No comparison with other optimised implementations**, e.g. Meta's code for "Memory Layers at Scale".
- **The table still costs time:** with a table the model trains at 62% (RX 9070) or 76% (MI350X) of the speed without
  one.
- **Not tuned per GPU**, and only measured on small models.

More detail: `docs/kernels.md`.

## To try it: benchmark script, demo and Hugging Face (2026-10-05)

**New:**
- `scripts/kernel_speedup.py`: kernels against the PyTorch reference on random tokens, no data needed, ~35 s.
- `scripts/demo_generate.py`: B-16M writes text, table (4 bit) on the NVMe, in RAM or in VRAM.
- README section "Try it". The weights (4-bit table + rest, 3.7 GB) are meant to go to Hugging Face.

**Checked from a fresh clone on the RX 9070:**

| PyTorch | Tests | Benchmark (training / reading / writing) |
|---|---|---|
| CachyOS package (2.14.0, HIP 7.2, Triton 3.5.1) | 106 ok, 2 skipped, 41 s | 1.47× / 1.70× / 1.26× |
| official wheel `rocm7.2` (2.14.1, Triton 3.8.0) | 106 ok, 2 skipped, twice | 1.53× / 1.75× / 1.27× |
| official wheel `rocm7.1` (2.13.0, Triton 3.7.1) | **abort** in `test_decode_graph_bit_identical[fp32]` | 1.50× / 1.77× / 1.25× |

Skipped are the Qwen test (no transformers) and one test that needs > 60 GB of GPU memory.

- **Benchmark numbers:** the factors are close to `docs/kernels.md`. But that was measured differently (trained
  model, bf16 table for reading), so this isn't a 1:1 repetition.
- **Abort with `rocm7.1`:**
  - Message: `HSA_STATUS_ERROR_INVALID_PACKET_FORMAT` ("The AQL packet is malformed").
  - Only happens when the test runs after the others. Alone, or only with `tests/test_kernels.py`, it passes.
  - A `torch.cuda.synchronize()` before freeing the graphs changed nothing, so it was removed again.
  - Cause not narrowed down. On the MI350X the same wheel ran without errors.
  - So the README recommends `rocm7.2` for Radeon.
- **`expandable_segments:True`:** with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reading a prompt after
  training crashed in the pure PyTorch path (`HSA_STATUS_ERROR_EXCEPTION`). The script now ignores the variable.
  Cause not narrowed down either.

**Demo (CachyOS PyTorch):**

| Table | Speed | GPU memory (peak) |
|---|---|---|
| NVMe | 144 tok/s | 0.42 GB |
| VRAM | 207 tok/s | 3.75 GB |

- 200 tokens, page cache warm. Same text in all three modes.
- The text is fluent Wikipedia English, the facts are made up ("Sir Isaac Newton was born on 14 November 1803 …").
  The README says so too.

**Hugging Face folder:**
- Files: `rest.pt`, `values_q4.bin`, `scales_q4.bin`, `hot_rows.npy` (hard links to the table files, sha256 checked),
  plus a new `meta.json` without local paths and a model card.
- Prepared for the upload (as of 2026-10-05).

## Codex review (2026-10-05)

A second model (OpenAI Codex) read the old state `AngryAnt` (`8cf0226`) and reported 22 findings. I checked every
code location it named.

**What changes in the results: nothing.**
- After every code change these were bit-identical to before: B-16M decode logits (table in VRAM and on the NVMe),
  B-1M training logits, loss and gradients, and the eval logits with fp32, bf16 and 4-bit tables.
- `report/qwen/step3_summary.json` comes out unchanged.
- The stricter fact scoring changes none of the 4 × 1,000 stored answers.

| No. | Finding | Right? | Done |
|---|---|---|---|
| 1 | Qwen queue reports "done" despite a failed evaluation | yes | "done" and `COMPLETE` only when all 11 steps ran, otherwise "INCOMPLETE" with a list; `run_dense` the same |
| 2 | `bag_infer` reads past the end when heads × knn isn't a multiple of 64 | yes, 4 × 32 = 128 not affected | last block masked (also `bag_forward`); tests with 96, 48, 40 lookups and 3 heads, the old version crashes on them |
| 3 | several new tokens after a filled KV cache without a causal mask | yes, only chunked prefill | mask with offset, test against the full forward |
| 4 | a restart takes empty or half files as done | yes | check (not empty, valid JSON, status), atomic writing (`smlm/atomic.py`); dense and cloud queue: "done" without `model.pt` is reported instead of skipped or retrained. A run identity from configuration and data hash is still missing |
| 5 | budget projection after a restart with 0 h of evaluation | yes | step times in `step_minutes.json` |
| 6 | evaluation reads a hard-coded `Q/general_ac.json` | yes | without this file `Q/general.json` |
| 7 | step push without the result files | yes | the result file is pushed too |
| 8 | a frozen table still gets trained, the table alone can't be trained | yes | both fixed, tests |
| 9 | optimizer state is taken over wrongly when loading | yes | `load_state_dict` now aborts with an error (training is never resumed) |
| 10 | BatchNorm in training isn't prefix-causal | yes | as a limitation in the README; all reported PPL in eval mode with running statistics |
| 11 | statistical certainty overclaimed | yes | "real" and "solid" reworded, ranges named as sensitivity ranges |
| 12 | "stored, but not retrievable" explains more than was measured | yes | reworded: targeted retrieval not shown, location of what was learned not tested |
| 13 | parity overstated in summaries | yes | README made more precise (decoding: table model with, model A without graphs; kernels "up to rounding"); `final.md` is private |
| 14 | MI350X addendum can't be backed up | no | the rewrite has `runs/amd_mi350x` and the section above; Codex checked the old folder |
| 15 | "12" counts as correct for "12.5" | yes | rule made stricter, regression test |
| 16 | unsupported sizes only fail inside the kernel | yes | `mem_impl="triton"` checks in the constructor and reports clearly |
| 17 | batch divisibility in Qwen training, `zero_grad` without effect | yes | checked, or documented |
| 18 | offload bit-identity only shown through the NLL sum | yes | logits compared directly (`scripts/check_offload_identical.py`): bit-identical with Triton 3.5; with Triton 3.8 they differ in the last bits, same tokens |
| 19 | "thirty times" instead of 56 times, GB instead of GiB, 4 GB scope | yes | corrected, footnote on the scope |
| 20 | v2 status, image link, "refuted" | yes | corrected |
| 21 | Qwen runs without commit and versions | yes | `run-info.json` now contains both; dating by page ids named in the README |
| 22 | the fit isn't an exact log fit | yes | description corrected; an exact log fit moves the fitted values by at most 0.3 M (58.5 / 83.4 / 115.7 M) |

**New while checking 18:** with the official `rocm7.2` wheel (Triton 3.8), `bag_infer` sums in a different order on
the small staging table (RAM/NVMe) than on the full table (VRAM).
- Both are equally close to an exact fp64 sum, with a deviation of ≈ 1e-8.
- The logits differ by up to 0.44 because of that (a few bf16 steps). The greedily generated tokens were the same in
  all checks (`report/offload/identical_check_triton38.json`).
- With Triton 3.5 everything is bit-identical.

**Not done, those would be new experiments:** more seeds, table ablation on the Qwen add-on (done since, see the
addendum to step 3), model A with graphs, causal normalisation instead of BatchNorm.

**Checked:**
- 127 tests on the RX 9070, from a fresh clone with the `rocm7.2` wheel.
- The Qwen tests (8) in `.venv-qwen`.
- The kernel tests for the affected kernels also in the CPU interpreter.
- The queue logic without a pod: `tests/test_cloud_queue.py` and the fake mode of `run_dense.py`.

## Step 4: n-gram table (Engram) against product keys (criteria fixed before the run, 2026-10-07)

**Question:** Since the end of August, Qwen3.8-Flash-Next has a large n-gram table that can live on the SSD (the
Engram principle, Cheng et al. 2026). There the rows are picked by the last 2–3 tokens, here by the hidden state
(product keys). Which way of picking rows brings more for a table of the same size?

**Setup, fixed and not tuned on short trial runs:**
- **Model E-1M** (`smlm/engram.py`, preset `E-1M`):
  - Model A plus two Engram modules, added to the residual stream before attention, the FFN stays.
  - Position: layers 2 and 6 (0-based 1 and 5), early and middle as in the paper (layers 2 and 15 of ~30). Chosen by
    that analogy, not tuned.
  - Per module n-grams of order 2 and 3 with 8 hash heads each. Every head has its own table with 524,287 rows (a
    prime) of 24 values, giving 16 rows per token and module, 384 values when concatenated.
  - Token compression NFKC + lower-casing on the GPT-2 tokens: 39,393 instead of 50,304 ids (−22%).
  - Context-aware gate on the hidden state, causal depthwise convolution (kernel 4, dilation 3, initialised to
    zero).
  - Tables together **402,652,416 parameters**, B-1M has 402,653,184. Dense part 21.84 M (A: 21.24 M).
- **Training:** exactly the arguments of B-1M-sparse (`runs/hampter`): Wikipedia, 500 M tokens, data seed 1234,
  batch 32 × 1024, LR 6e-4, eval every 10 M tokens, WikiText-103 as a side value. Only `--model` and the table LR
  **3e-3** (paper: 5 × LR) are changed.
- **Optimizer of the tables:** row-sparse with lazy Adam, like the product-key table. **Deviation from the paper**
  (plain Adam there): here only a small share of the rows is read per step, so lazy or plain Adam can matter more
  than with product keys. That's why there's a control run with plain Adam.
- **Runs** (at home, RX 9070, `scripts/run_engram.py`, output `runs/engram/`):
  - E-1M seed 0 and seed 1: main comparison.
  - E-1M-dense seed 0: control, plain Adam on all rows, one seed.
- **Not in this step:** both tables together (preset BE-1M) don't fit on the 16 GB card, with ~12 GiB for the table
  and optimizer state alone. Comes later on a bigger GPU.

**Criteria:**
- **Main verdict:** val PPL Wikipedia after 500 M tokens, mean of 2 seeds, E-1M against B-1M-sparse (21.837 /
  21.752, mean 21.794). The measured seed spread is ~0.4%.
  - **"E better":** E ≤ 0.99 × B.
  - **"on par":** E within ±1% of B.
  - **"B better":** E ≥ 1.01 × B.
- **Against A** (25.665 / 25.756, mean 25.711): improvement in % and equivalent dense size with the same
  interpolation as for B (B-1M: 60 M).
- **Optimizer control:** if E-1M-dense-s0 ≤ 0.99 × E-1M-s0, lazy Adam puts the n-gram table at a disadvantage. Then
  the main verdict is also given with the dense value, as a hint with only one seed.
- **Reported, but without a verdict:**
  - WikiText-103 val PPL (B-1M-sparse: 65.51 / 65.09).
  - Training speed on the RX 9070 (B-1M with kernels: ≈ 59,700 tok/s) and peak GPU memory.
  - Table values read per token: E 2 × 16 × 24 = 768, B 3 × 128 × 384 = 147,456 (192 times more). That matters for
    offloading to RAM or SSD, but isn't measured here.
  - Share of the table rows that are read on the val set.

**Caveat, written down beforehand:** Engram is built for much bigger models and tables (paper: 5.7 B table
parameters). At 0.4 B in 16 × 524,287 small rows, many n-grams share a row. If E loses here, that says something
about this scale and this configuration, not about Engram in general. Also not taken over: the paper's
hyper-connections (mHC).

**Trial runs before the approval (technique only, no tuning):**
- A complete run over 4 M tokens including evaluation, checkpoint and inference benchmark for E-1M and E-1M-dense.
- The whole queue with mini runs (`--smoke`).
- Speed: E-1M ≈ 79,000 tok/s (≈ 1.9 h per run), E-1M-dense ≈ 64,000 tok/s (≈ 2.3 h). Memory ≈ 10.5 GiB each.

### Result of step 4 (2026-10-08, RX 9070, `scripts/engram_eval.py` → `report/engram/summary.json`)

The three runs ran overnight one after another (7.10. 20:18 to 8.10. 02:22), alone on the card: nothing else was
using the GPU, and the speed stayed flat over the whole run. No errors, no stalls.

| Run | Val PPL Wikipedia | Val PPL WikiText-103 | Training tok/s | Peak GPU memory (training) |
|---|---:|---:|---:|---:|
| E-1M seed 0 | 23.133 | 71.04 | 78,879 | 10.49 GiB |
| E-1M seed 1 | 23.105 | 70.57 | 78,961 | 10.49 GiB |
| E-1M-dense seed 0 (control, plain Adam) | 22.694 | 68.09 | 64,334 | 10.47 GiB |
| B-1M-sparse seed 0 / 1 (reference) | 21.837 / 21.752 | 65.51 / 65.09 | ≈ 59,700 with kernels | 10.52 GiB |
| A seed 0 / 1 (no table) | 25.665 / 25.756 | 78.38 / 78.48 | 91,563 / 91,641 | 7.89 GiB |

**Verdict by the criteria: "B better".**
- **Main verdict:** E-1M averages 23.119 against 21.794 for B-1M, so E = **1.061 × B**, far outside the ±1% band.
  The two Engram seeds are only 0.12% apart.
- **Optimizer control: lazy Adam does put the n-gram table at a disadvantage.** E-1M-dense-s0 is 0.981 × E-1M-s0, so
  below the 0.99 line. As fixed beforehand, the main verdict with the dense value (one seed, a hint only): 22.694 =
  **1.041 × B**, still "B better".
- **Against A:** E-1M is 10.1% better than A (B-1M: 15.2%). Equivalent dense size with the same interpolation as in
  step 1:

| | Val PPL | Equivalent dense size | Sensitivity range (±0.4%, incl. fit) |
|---|---:|---:|---:|
| A (no table) | 25.71 | 21 M | |
| E-1M (mean of 2 seeds) | 23.12 | **41 M** | 39–43 M |
| E-1M-dense (1 seed) | 22.69 | **46 M** | 44–49 M |
| B-1M-sparse (mean of 2 seeds) | 21.79 | **61 M** | 57–64 M |

**Reported without a verdict:**
- **WikiText-103:** the same order, E-1M 70.8 on average, E-1M-dense 68.1, B-1M 65.3.
- **Speed:** E-1M trains at ≈ 78,900 tok/s, ≈ 1.3 times as fast as B-1M with the kernels (≈ 59,700) and 86% of the
  speed of A. The dense control is slower (64,300), because plain Adam touches all 403 M table values in every step.
  (The B-1M runs in `runs/hampter` used the PyTorch path, at ≈ 38,500 tok/s; the fair comparison is the one with
  kernels.) Peak memory is the same for all table models, ≈ 10.5 GiB.
- **Table values read per token:** E 768, B 147,456, i.e. 192 times more for B.
- **Share of the table rows read on the val set:** E-1M 74% (bigram heads 62%, trigram heads 86%), B-1M practically
  100%. With E the rows only depend on the tokens, so the number comes straight from the hash.

**Course at equal tokens, not a criterion:**

| Tokens | E-1M (mean) | E-1M-dense | B-1M (mean) | E / B | E-dense / B |
|---|---:|---:|---:|---:|---:|
| 100 M | 39.95 | 38.89 | 39.25 | 1.018 | 0.991 |
| 200 M | 30.50 | 29.84 | 29.60 | 1.030 | 1.008 |
| 300 M | 26.56 | 26.02 | 25.40 | 1.046 | 1.025 |
| 400 M | 24.15 | 23.70 | 22.87 | 1.056 | 1.037 |
| 500 M | 23.12 | 22.69 | 21.79 | 1.061 | 1.041 |

The n-gram table is quick at the start: the dense control was slightly ahead of B-1M until about 150 M tokens.
At 100 M that lead is under 1%, about as large as the difference between the two Engram seeds at that point, so I
don't read much into it. After that B-1M pulls away steadily, and the gap is still growing at the end. I didn't
fix this comparison beforehand, and the control has one seed.

**What this compares:** two whole architectures with the same table size and the same training, not just two ways
of picking rows.
- **E-1M** keeps the FFN in all 12 layers and adds two modules before the attention of layers 2 and 6.
- **B-1M** replaces the FFN in layers 3, 7 and 11 with memory layers.
- **Reads per token:** E reads 768 table values, B 147,456.
- **Rows:** E has 16 × 524,287 small rows of 24 values, in which many n-grams share a row; B has 1 M rows of 384
  values, shared by the three layers.

So "B better" means that this product-key package beats this Engram package at this size. It doesn't say which single
ingredient makes the difference.

**What I read from it:**
- **Step 7 left a question open:** a large part of what B-1M's table delivers can be predicted from the last two tokens
  (three quarters of the gain in perplexity, 61% in loss). Can a model that learns with n-gram rows from the start make
  up for the context-dependent rest somewhere else? Only partly. Measured against A, E-1M gets 66% of B-1M's gain,
  E-1M-dense 77%. (A different reference from step 7, which measured against a model with its table switched off; the
  two percentages aren't directly comparable.) The rest needs rows picked by the context.
- **Lazy Adam costs the n-gram table about 2%,** for product keys it cost only 0.16%, within the seed noise (stage 1c).
  One possible reason: an n-gram row comes up rarely and only for its own n-gram, and plain Adam keeps moving it with
  the remaining momentum in the steps between. With one seed I'll leave it at that.
- **For offloading the picture is reversed:** E reads 192 times fewer values per token, and which ones is known
  before the layer runs. For a table on the SSD that's a real advantage of Engram, and this run doesn't weigh it.
- **The caveat written down beforehand still holds:** Engram is built for much bigger models and tables. This result
  is about 0.4 B table parameters on a 21 M model and 500 M tokens, not about Engram in general.

## Addendum to step 3: where does Q+T keep what it learned? (table ablation, criteria before measuring, 2026-10-07)

**Question:** Q+T adapts to the training articles much more strongly than Q+D (PPL on trained articles −58% instead of
−30%). Is that in the value table or in keys, query projection, BatchNorm, swilu and gates? Not tested so far (Codex
finding 12).

**Measurement** (at home, RX 9070, `scripts/qwen_table_ablation.py`, no training): the same trained add-on
`runs/qwen_cloud/QT-s0/addons.pt` in three variants.
- **T:** as trained, measured again at home, so that all comparisons come from the same card.
- **Z (main variant):** value table set to zero, everything else unchanged. The add-on blocks then only deliver the
  bias of their output projection.
- **R:** value table drawn again, with the same distribution as at the start of training (normal, σ = 1/√1024).
- **Measured:** PPL on `mem_probe` (a slice of the training articles), `val_new` (new, never trained articles) and
  `val_known` (articles from before Qwen's cut-off), plus the fact test (500 blanks each from training and
  counter-check articles). Reference Q: measured at home (`runs/qwen/Q-base`, `report/qwen/facts_Q_home.json`).

**Criteria:**
- **Share of the table** in the gain per PPL set, on log PPL: s = (ln Z − ln T) / (ln Q − ln T). At s = 1 the whole
  gain over Qwen alone goes away with the table, at s = 0 nothing.
  - **"mostly in the table":** s ≥ 0.5.
  - **"shared":** 0.2 < s < 0.5.
  - **"mostly in the rest":** s ≤ 0.2.
- **Main set** is `mem_probe` (the question about what was memorised), `val_new` and `val_known` are reported the
  same way.
- **Fact test:** hit rate of T, Z and Q with paired bootstrap intervals (95%) for T − Z. Only if the interval excludes
  zero does it say "the fact gain depends on the table".
- **R** is reported without a verdict. R shows whether the model needs the content of the table or just values of
  this size; R worse than Q would suggest that the rest is tuned to the trained table.

**Result** (2026-10-07, RX 9070, 47 min, `report/qwen/table_ablation.json`):

| | Q (gates 0) | T (trained) | Z (table 0) | R (table random) | Share s of the table |
|---|---|---|---|---|---|
| PPL `mem_probe` (training articles) | 13.363 | 5.640 | 12.772 | 12.870 | **0.95** |
| PPL `val_new` (new articles) | 12.977 | 10.095 | 12.388 | 12.485 | **0.82** |
| PPL `val_known` (old articles) | 13.935 | 14.903 | 13.400 | 13.499 | 1.58 |
| PPL `val_known_same` | 14.282 | 14.031 | 13.706 | 13.823 | −1.33 |
| Fact test training | 25/500 (at home) | 51/500 | 23/500 | 25/500 | |
| Fact test counter-check | 19/500 (at home) | 48/500 | 22/500 | 21/500 | |

- **Controls:** Q via gates 0 matches the earlier Q values in every digit. T matches the cloud measurement (5.638 /
  10.093).
- **Verdict `mem_probe`: "mostly in the table"** (s = 0.95). Without the table almost the whole adaptation to the
  training articles is gone (13.36 → 5.64 → 12.77).
- **`val_new`: also "mostly in the table"** (s = 0.82).
- **`val_known`:** the worsening on old articles (+7%) comes entirely from the table. Without the table the add-on is
  even 3.8% better than Qwen alone there (s > 1).
- **`val_known_same`:** T is only 1.8% better than Q, without the table 4% better. Here the table does harm compared
  with the rest of the add-on, so s is negative and can't be read as a share.
- **Fact test: "the fact gain depends on the table".** T − Z = +5.6 pp [3.0; 8.2] on training articles and +5.2 pp
  [3.2; 7.4] on the counter-check. Without the table the add-on falls back to the level of Qwen alone. The gain is the
  same on training and counter-check articles, so the table helps with completing Wikipedia facts in general, not
  specifically with the trained ones.
- **R ≈ Z:** a random table helps as little as none. The model uses the learned content, not just values of this size.
- **The rest of the add-on without the table** brings ≈ 4% evenly on all sets. With the table at zero, the blocks only
  deliver the learned bias of their output projection, so a fixed shift per block, a general adaptation to Wikipedia
  text.

**What that means:** the gain needs the learned table. Without it the add-on loses what was memorised from the
training articles, the adaptation to new articles, the gain in the fact test and also the harm on older text.
Strictly, zeroing the table also cuts off everything the trained keys, queries and swilu projections do, because they
only act through the table. So this shows that the gain lives in the jointly trained table system, not that it sits in
the table values alone. The finding from step 3 stays open: the stored articles don't make the *trained* facts more
retrievable than others. (Sharpened on 2026-10-07 after a review: the first version said "the table is the memory"
and that the question "where is it?" was answered, which goes further than the data.)

## Step 5: reorder the table rows on the SSD (offline simulation, criteria before measuring, 2026-10-07)

**Question:** When B-16M runs with the table on the SSD, every row that isn't in RAM costs a whole 4 KiB page, but a
row is only 192 bytes. A reader comment put it this way: the page cost per missed row is mostly a layout problem. Does
B-16M read clearly fewer pages if rows that are read together sit on the same page?

**Idea:**
- Row r = i · 4096 + j, where i and j are the indices of the two sub-keys. Per head and token the 32 rows come from
  only about 14 different i (32 of the validation sessions, looked at while testing the recorder, today's layout only).
- Rows with the same i already lie in the same 768 KiB block today, but on random pages within it. If the j axis is
  reordered, the same way for every i, so that j's often chosen together become neighbours, rows with the same i can
  share a page.
- The model stays exactly the same: the sub-keys of the second half are permuted in the same way in every head and
  every memory layer, the table rows accordingly. No lookup table, no extra work when reading.
- Mirror image: transpose the table (j outside) and reorder the i axis.

**Setup, fixed before the first real number:**
- **Recorded lookups** (`scripts/record_lookups.py`): which rows B-16M reads, for sessions of 384 tokens (128 prompt +
  256 more, as in the latency measurement), 4-bit table, teacher forcing (the continuation is real text, not the
  model's own tokens). 4,608 sessions from the training text, 1,000 from the validation text, at seeded random
  positions.
- **Simulation** (`scripts/rs_layout.py`, no SSD, no GPU), same setting as the latency measurement:
  - the 30% hottest rows (`hot_rows.npy`) are in RAM, all others come from the SSD;
  - a page read once stays in the page cache until the end of the session, every session starts cold;
  - counted: distinct 4 KiB pages per session, a row across a page boundary costs both pages.
- **Layouts:** A0 today (i outside); AJ: i outside, j axis reordered; AI: j outside, i axis reordered.
- **Learning AJ and AI** from 4,096 training sessions: how often two non-RAM rows with the same i (AJ) or the same j
  (AI) are read in the same session. Two ways to get an order from that: spectral (Fiedler vector) and greedy (always
  append the index with the most co-reads with the last 21 placed ones; a page holds 21.3 rows). Four candidates.
- **Choice:** the 512 other training sessions decide which of the four candidates is used (fewest pages per session).
  The 1,000 validation sessions are evaluated exactly once, with the chosen one.

**Criteria:**
- **Main measure:** pages per session, chosen layout against A0, relative change over the 1,000 validation sessions,
  95% bootstrap interval over the sessions.
  - **"Worth it":** ≥ 25% fewer pages. Then the reordered table gets built, checked to give the same outputs, and
    measured on the SSD (prompt time, per-token latency, reads).
  - **"Small gain":** 10–25% fewer. Only worth pursuing together with faster reading (direct I/O).
  - **"Layout is not the lever":** < 10% fewer. Then the reading itself is the place to start (direct I/O, io_uring).
- **Reported without a verdict:** the same without the RAM cache; prompt pages and new pages per continuation token
  separately; all four candidates on the 512 training sessions.

**Caveats:** It's a simulation. A real page cache can drop pages, and the SSD merges neighbouring pages into one
request, so fewer pages don't automatically mean the same share less time. Teacher forcing on real text instead of
generated text. One model (B-16M), one cache size.

### Result of step 5 (2026-10-07, offline, `report/rs/`)

- **Verdict: "layout is not the lever".** The chosen layout (AI, greedy) reads **1.8% fewer pages** per session than
  today's [95% interval −1.9; −1.8], far below the 10% threshold. The reordered table won't be built.
- 1,000 validation sessions of 384 tokens:

| | Today (A0) | Reordered (AI greedy) | Change |
|---|---:|---:|---:|
| Pages per session, 30% of the rows in RAM | 26,762 | 26,276 | −1.8% |
| of which in the prompt (128 tokens) | 9,648 | 9,495 | |
| New pages per further token | 66.9 | 65.6 | |
| Pages per session, nothing in RAM | 100,515 | 96,767 | −3.7% |

- Choice on the 512 training sessions: AJ spectral −0.4%, AJ greedy −1.7%, AI spectral −0.3%, AI greedy −1.8%.
- **Why so little** (100 of the training sessions): a session reads ~26,500 different rows outside RAM, spread over
  ~3,770 of the 4,096 i blocks with ~7 rows each, and needs ~26,400 pages for them: practically one page per row.
  Which j's are read together depends on the query and the head. Within only 200 sessions, 96% of all 16.8 M possible
  j pairs were read together with the same i at least once. A fixed order has nothing to hold on to.
- **What that means:** with the layouts tested here (one fixed order of an index axis), practically every row that
  misses the RAM costs its own random read. That doesn't rule out every possible layout, but there is little stable
  structure to exploit. (Sharpened on 2026-10-07 after a review: the first version said "whatever the layout".) What
  helps more directly is fewer misses (more rows in RAM, prefetching) or more reads per second. The quick run of the direct-I/O benchmark
  (`scripts/bench_table_io.py`, branch `codex/dio`) gives io_uring 1.4–1.8× the reads per second of mmap; that's
  the next step.
- **Plausibility check:** the latency measurement counted 8,059 device reads for the cold prompt (simulation: 9,648
  pages) and ~31 per further token (simulation: 67). The SSD merges neighbouring pages into one request, and the
  measurement continued with the model's own text, which repeats itself more than real text. So only the order of
  magnitude is comparable.

## Step 6: do the models trained from scratch remember the facts they saw? (fact test, criteria before measuring, 2026-10-07)

**Question:** In step 3 the table add-on for Qwen stored what it learned, but didn't bring the trained facts back
better than others (addendum to step 3). My guess: a frozen Qwen core doesn't know how or where to fetch them. So now
the models that learned with the table from the start: does B-16M bring back facts from articles it saw in training
better than facts from articles it never saw, and by more than dense models of similar quality?

**Setup:**
- **Items** (`scripts/make_fact_cloze_lm.py` → `data/lm_fact_cloze.jsonl`, sha256 `9c02b5b0…2673`), same rules as
  in step 3: names, dates/years, numbers; prompt = title + blank line + the sentence up to the fact; the answer doesn't
  occur in the prompt.
  - **Seen:** 1,382 items from training articles. The answer was a prediction target in a window that was used in
    training. All 500 M-token Wikipedia models used the same windows in the same order (data seed 1234), so "seen" is
    the same for all of them, and the training step in which an item was seen is known.
  - **Unseen:** 1,382 items from the 1,917 validation articles (random articles of the same dump).
  - Same mix on both sides: 600 names, 450 dates/years, 332 numbers. Every article was seen exactly once (one epoch).
- **Scoring** (`scripts/eval_fact_cloze_lm.py`): greedy, first attempt, exact match at the start as in step 3.
  Product-key models with the 4-bit table (B-16M only fits that way; for B-1M the 4-bit table changes val PPL by
  +0.13%).
- **Models:** B-16M (val PPL 19.96) against D-100M (20.27) and D-200M (18.76), which bracket it. Also A, B-1M, B-4M,
  D-50M, D-400M.

**Criteria:**
- **Memory gap** of a model = accuracy on seen items minus accuracy on unseen items, in percentage points.
- **Main verdict:** gap of B-16M minus gap of D-100M, and minus gap of D-200M, each with a 95% bootstrap interval
  (items resampled within each split, the same draws for all models, so the comparison is paired).
  - **"The table remembers more":** both intervals above 0.
  - **"Dense remembers more":** both intervals below 0.
  - **"No clear difference":** otherwise.
- **Side check:** is the gap of B-16M itself above 0 (interval)? If not, B-16M doesn't bring back seen facts better
  than unseen ones at all.
- **Reported without a verdict:** the gaps of all eight models against their val PPL; by type; accuracy on seen items
  by fifth of training (seen early or late: forgetting); accuracy on unseen items (general knowledge and guessing).

**Caveats:** Small models, and every article was seen only once, so low hit rates are expected; what counts is seen
against unseen. The prompt only has the title and the sentence, not the article text before it. One seed per model.
Names come from a simple pattern, so some "names" are other capitalised phrases, the same for all models. Before the
criteria I only tested the scoring on three made-up prompts, none of the items.

### Result of step 6 (2026-10-07, RX 9070, `report/facts_lm/`)

- **Verdict: "no clear difference".** Gap of B-16M minus gap of D-100M: +0.9 pp [−0.5; 2.4]; minus gap of D-200M:
  +1.0 pp [−0.4; 2.4].
- **Side check failed:** the gap of B-16M itself is −0.2 pp [−2.2; 1.7]. B-16M doesn't bring back facts from seen
  articles better than from unseen ones.
- **And neither does any other model:** all eight gaps lie between −1.2 and +0.1 pp, every interval includes 0.

| Model | Val PPL | Seen | Unseen | Gap [95%] |
|---|---:|---:|---:|---:|
| A | 25.67 | 4.9% | 4.9% | 0.0 pp [−1.7; 1.6] |
| B-1M | 21.84 | 5.8% | 5.7% | +0.1 pp [−1.7; 1.8] |
| D-50M | 22.45 | 5.4% | 5.7% | −0.4 pp [−2.1; 1.4] |
| B-4M | 20.80 | 6.5% | 7.2% | −0.7 pp [−2.5; 1.2] |
| D-100M | 20.27 | 5.4% | 6.6% | −1.2 pp [−3.0; 0.6] |
| **B-16M** | **19.96** | **7.0%** | **7.2%** | **−0.2 pp [−2.2; 1.7]** |
| D-200M | 18.76 | 5.9% | 7.2% | −1.2 pp [−3.1; 0.7] |
| D-400M | 17.60 | 7.0% | 7.7% | −0.7 pp [−2.7; 1.2] |

- **By fifth of training** (seen items, ~276 per fifth, so a few points of noise): B-16M 6.0 / 8.6 / 5.8 / 7.1 /
  7.3%, no trend. Only A rises from 2.3 to 7.3%; with one model and this noise I don't read anything into it.
- **By type:** names are hit most often (B-16M 10.0% seen, 11.5% unseen), numbers least (4.2% / 3.6%), no type with a
  clear gap. 172 seen and 165 unseen items are hit by at least one model, 36 and 39 by all eight: mostly general
  knowledge or easy to guess.
- **What that means:** after seeing an article once, none of these models, with or without table, shows a measurable
  advantage on its facts over facts from articles it never saw. For B-16M an advantage of up to 1.7 points is still
  inside the interval. Two limits: "unseen" refers to the article, not the fact, which can also appear in other
  training articles; and the cloze prompt is a different context than the training window, which for a product-key
  model also changes which rows are read. My guess from step 3 (the frozen Qwen core doesn't know how or where to
  fetch) isn't supported: models that learned with the table from the start don't show it either. With one pass
  (here) and two passes (step 3), no targeted recall shows up in these tests. A more sensitive follow-up would score
  the probability of the full answer and use the original training context as the prompt. (Sharpened on 2026-10-07
  after a review: the first version said the models don't recall the facts, which the intervals don't support.) B-16M has the highest hit rate of all models on
  seen items (7.0%, D-400M 6.95%), but just as much on unseen ones: general knowledge, not memory of the article.
- **Open:** from how many repetitions does a fact stick, and does a model with table need fewer than a dense one? That
  needs a controlled test with facts that appear a known number of times (idea, not planned yet).

## Addendum to step 6: a more sensitive fact test (criteria before measuring, 2026-10-07)

**Why:** a review (Codex, 2026-10-07) pointed out that "hit or no hit" is a coarse measure, and that the short cloze
prompt is a different context than the one the model saw in training. For a product-key model a different context
also means other table rows.

**Setup** (`scripts/eval_fact_logprob_lm.py`, same 2 × 1,382 items as step 6, same eight models, 4-bit table):
- **Measure:** log-probability of the whole answer (sum over its tokens, teacher forced), not just greedy hit or
  miss.
- **Two prompts:**
  - *cloze:* as in step 6.
  - *context:* the article's own text before the fact, from the article start, at most 1,000 tokens. For seen items
    that's the text the model read right before the fact in training. Items whose answer already appears in that
    context are left out (copying, not memory): 1,204 seen and 1,189 unseen remain.
- **Recency, new:** the training windows came in random order, so facts seen in the last fifth of training and facts
  seen in the first fifth are equally hard on average. A difference between them can only come from having seen them
  (remembering and forgetting). Unlike seen against unseen, this comparison isn't affected by the two sets of
  articles being of different difficulty.

**Criteria** (context prompt; 95% bootstrap intervals over items, paired across models):
- **Does B-16M remember at all?** Recency of B-16M = mean log-probability of late-seen minus early-seen items. Interval
  above 0: "B-16M carries a measurable memory of recently seen articles". Otherwise: "no measurable memory".
- **Table against dense:** gap (seen minus unseen) of B-16M minus that of D-100M, and minus that of D-200M.
  - "The table remembers more": both intervals above 0.
  - "Dense remembers more": both below 0.
  - "No clear difference": otherwise.
- **Reported without a verdict:** the same for the recency difference B-16M minus D; everything for the cloze prompt;
  all eight models; log-probability per fifth of training; the hit rate along the article's own tokens.

**Before the criteria:** the scoring was only tested on three made-up prompts.

### Result of the addendum (2026-10-07, RX 9070, `report/facts_lm/logprob_summary.json`)

- **Does B-16M remember at all? "No measurable memory".** Recency (late minus early, context prompt) +0.06 nats
  [−1.18; 1.33]. Honestly: with ~220 items per fifth and a large spread per item, this test can only see effects of
  more than about a nat. It is weaker than I expected when I wrote the criteria.
- **Table against dense: "no clear difference".** Gap of B-16M minus D-100M +0.15 nats [−0.01; 0.31], minus D-200M
  +0.10 [−0.04; 0.23].
- Context prompt, mean log-probability of the answer (nats):

| Model | Val PPL | Seen | Unseen | Gap [95%] | Recency [95%] |
|---|---:|---:|---:|---:|---:|
| A | 25.67 | −9.64 | −10.01 | +0.37 [−0.27; 0.99] | −0.08 [−1.44; 1.28] |
| D-50M | 22.45 | −9.10 | −9.46 | +0.36 [−0.26; 0.95] | +0.06 [−1.20; 1.33] |
| B-1M | 21.84 | −8.81 | −9.24 | +0.42 [−0.16; 1.00] | +0.11 [−1.14; 1.38] |
| B-4M | 20.80 | −8.57 | −9.03 | +0.46 [−0.12; 1.02] | +0.30 [−0.99; 1.60] |
| D-100M | 20.27 | −8.72 | −9.11 | +0.39 [−0.21; 0.97] | +0.00 [−1.26; 1.25] |
| **B-16M** | **19.96** | **−8.27** | **−8.80** | **+0.54 [−0.03; 1.08]** | **+0.06 [−1.18; 1.33]** |
| D-200M | 18.76 | −8.38 | −8.82 | +0.44 [−0.14; 0.99] | +0.17 [−1.06; 1.40] |
| D-400M | 17.60 | −8.09 | −8.61 | +0.52 [−0.04; 1.07] | +0.18 [−1.05; 1.40] |

- **What I read from it:** every model, even A, finds the seen answers about 0.4–0.5 nats more likely. Since it's the
  same for all, it can just as well be that the seen articles are a little easier; the test can't separate that. The
  gap grows a bit with model quality, and B-16M has the largest one (+0.54, like D-400M with +0.52), slightly above
  its two dense neighbours, but inside the noise. The cloze prompt gives the same picture (B-16M minus D-100M +0.10
  [−0.05; 0.25], minus D-200M +0.06 [−0.07; 0.20]).
- **Overall with step 6:** once-seen facts leave at most a small trace in all these models, with or without table.
  The way to a clear answer is the controlled test with repeated facts (FACTK in my notes), not a finer measurement of
  the same data.

## Step 7: how lexical is the table? (no training, criteria before measuring, 2026-10-07)

**Question** (from the same review): does the table need context-specific addresses at all, or is most of what it
delivers predictable from the current token or the last two? If so, a cheap n-gram lookup like Engram could replace
much of the expensive search. If not, the content-based addressing of product keys is what counts.

**Setup** (`scripts/lex_bags.py`, model B-1M from the cloud runs, val PPL 21.84):
- The bag (weighted sum of the value rows a memory layer reads, before the swilu gate) is replaced in all three memory
  layers at once. Gate, output projection and everything else still see the real context.
- **Variants:**
  - *real:* unchanged.
  - *zero:* bag = 0, what the model is worth without the table's content.
  - *tok:* mean bag of the current token.
  - *bigram:* mean bag of the last two tokens (hashed into 2^20 buckets; fewer than 4 occurrences fall back to *tok*).
  - *other:* a real bag of the same current token from another context, drawn at random (16 kept per token).
- Means and samples come from 4,096 random training windows (4.2 M tokens). The evaluation runs once on the whole
  validation set (1,449 windows of 1,024 tokens).
- **Kept share** of the table's gain = (PPL_zero − PPL_variant) / (PPL_zero − PPL_real), 95% bootstrap interval over
  the windows.

**Criteria** (for *bigram*):
- **"Largely lexical":** kept share ≥ 75%.
- **"Partly lexical":** 25–75%.
- **"Mostly context":** < 25%.
- **Reported without a verdict:** *tok*, *other*, and the PPL of every variant.

**Before the criteria:** a plumbing test with statistics from only 64 training windows, evaluated on 16 other
training windows (not the validation set), showed that the variants run and differ clearly. I looked at those
numbers; the thresholds above are the ones I had planned before that test.

### Result of step 7 (2026-10-07, RX 9070, `report/lex/eval.json`)

| Variant | Val PPL | Kept share of the gain [95%] |
|---|---:|---:|
| real | 21.84 | 100% |
| zero (no table content) | 84.81 | 0% |
| tok (mean per current token) | 44.21 | 64.5% [64.2; 64.7] |
| **bigram (mean per last two tokens)** | **37.19** | **75.6% [75.3; 75.9]** |
| other (real bag of the same token from another context) | 71.52 | 21.1% [20.5; 21.7] |

- **Verdict by the criteria: "largely lexical"**, just over the line: the bigram mean keeps 75.6% of the gain, the
  whole interval lies above 75%.
- **But that number flatters the lexical part, and I should have seen it before:** the reference "zero" is a model
  whose table suddenly delivers nothing, and at 84.81 it is much worse than model A, which never had a table (25.67).
  Measured against A, the picture turns around: with bigram means B-1M lands at 37.19, clearly *worse* than having no
  table at all. The quarter that depends on context is exactly the part that makes B-1M better than A.
- **"other" shows how context-specific a single bag is:** a real bag of the same token from another sentence keeps
  only 21%. The mean works because it averages the context-specific parts away.
- **Correction (2026-10-09):** the shares above are on the perplexity scale, as the criteria fixed it. On the loss
  scale (nats), the one the model is trained on, the same numbers give tok 48.0%, bigram 60.8%, other 12.6%. My first
  reading, "three quarters of what the table delivers", was too strong; a read-only review by Codex pointed it out.
  The verdict by the criteria stays.
- **What that means:** a large part of what the table delivers can be predicted from the last two tokens (75.6% of the
  gain in perplexity, 60.8% in loss), so an n-gram lookup could supply that part cheaply. The rest depends on the
  context, and in this model it is indispensable. One limit: the rest of B-1M was trained together with the real bags; a
  model trained with n-gram rows from the start can learn to make up for the missing part elsewhere. Whether it does is
  exactly what tonight's Engram run (step 4) measures. (Result: only partly, see "Result of step 4".)

## Step 8: B-16M at home with its table in RAM (criteria fixed before the run, 2026-10-08)

**Question:** On the H200, B-16M needed 101 GB of GPU memory to train, because the table, its gradient accumulator and
Adam's two moments are each as big as the table (6.44 B values). Can a 16 GB gaming card train it if all of that stays
in the PC's RAM (128 GB) and only the rows read in a step travel to the GPU? And does that give the same model?

**How it works** (`--value_device host`: `smlm/host_values.py`, `smlm/host_optim.py`, `smlm/host_rows.c`; written by
Codex as branch `codex/big`, merged and checked on 2026-10-08):
- Table, accumulator and both Adam moments stay fp32 in RAM, 96 GiB in total. The SSD isn't used; nothing of the table
  is written there in training.
- Each memory layer copies the distinct rows it reads to the GPU and the gradient of those rows back, where it is added
  to the accumulator.
- After each step Adam updates only the rows that were read, on the CPU (lazy Adam as before). Since 2026-10-08 that's
  a small C loop over the rows, 2.6 times as fast as the PyTorch version.
- Everything runs one after another, without overlapping steps, so the math is the same as with the table on the GPU.

**Checks before the approval (technique only, no tuning):**
- **Test suite** on the RX 9070: 487 passed, 7 skipped.
- **B-1M, table in RAM against table on the GPU**, 20 training steps each, same seed and data:
  - My rule beforehand was "losses equal to within 1e-4". It failed (differences up to 0.044).
  - But the GPU path doesn't meet it against itself either. Three GPU repeats with the same seed drift apart by up to
    0.024 from step ~10 on: the GPU adds in a different order each time, and the tiny rounding differences grow.
  - The RAM path was 0.019 away, as close as GPU to GPU. New rule, agreed on before the C loops: RAM against GPU no
    further apart than GPU against GPU. With the C loops 2 of 3 runs were within; one was outside because of a single
    gradient spike.
- **Gradient spikes:** single steps with a gradient norm 1.5 to 45 times the usual one turned up at random in all
  variants, GPU and RAM, with and without the C loops, always early in training. Clipping catches them. The Triton bag
  kernels give the same result 40 times in a row on fixed input, so they aren't the cause. In the long runs (every 10th
  step logged) I found no spike after step 1,000. This stays an open question outside this step.
- **Speed of B-16M with the table in RAM:** 5.8 s per step of 32,768 tokens (Adam on the CPU 2.1 s of it), 99.8 GiB
  RAM, 6 GiB GPU memory.
- **Trial run over the real training path** (10 M tokens, stopped after the first evaluation, its perplexity isn't
  used): runs end to end including the evaluations and the final benchmark, at most 100.4 GiB RAM.

**Run** (`scripts/run_big.py`, one run overnight, as a systemd unit with a 112 GB memory cap, so that running out of
RAM can only end the run and not the desktop):
- Exactly the arguments of the H200 run `runs/cloud/B-16M-s0`: Wikipedia, schedule over 500 M tokens, seed 0, data
  seed 1234, batch 32 × 1024, LR 6e-4, table LR 2.4e-3, evaluation every 10 M tokens, WikiText-103 as a side value.
- Plus `--value_device host --value_state fp32` (the same math as on the H200).
- `--stop_after_tokens 99.9e6`: the run stops after the evaluation at step 3,050 (99.9 M tokens). The learning-rate
  schedule stays that of the full 500 M run, so the value can be compared directly with the H200 curve at the same
  step.
- `--no_save`: no 26 GB checkpoint on the SSD.
- Expected duration 3.5 to 6 hours, depending on how many distinct rows each step reads.
- Why only 100 M tokens: the full run would take about a day and a half at home, too long for my PC.

**Criterion:**
- Val PPL Wikipedia in the row of step 3,050 of `metrics.csv`, against the same row of the H200 run (37.360):
  - **"Training in RAM reproduces the H200 run":** within ±1% (36.99 to 37.73).
  - **"Differs":** outside; then I look for the cause.
- Why ±1%: the seed spread at B-1M is 0.4%, B-1M at home and on the H200 ended 0.00% apart, and the GPU path itself
  isn't deterministic.
- **Reported without a verdict:** time per step, RAM and GPU memory peaks, the value at 50 M tokens (H200: 58.26),
  WikiText-103.

**What this can't show:** whether the rest of the way to 500 M tokens would match too (with the same schedule, a match
at 100 M is a strong hint, not a proof); and nothing about tables bigger than B-16M (the next size, 4× as many rows,
doesn't fit into 128 GB).

## Step 9: the same table with a quarter of the reads: narrower rows or fewer rows? (criteria fixed before the runs, 2026-10-09)

**Question:** Per token, B-1M reads 384 table rows of 384 values each, 147,456 values. That is what makes the table
expensive: with the table in RAM (step 8) every row read travels to the GPU and back and gets an Adam update on the CPU;
from the SSD every row read is one access. Can the same 402.65 M table parameters do as well with a quarter of the
values read? Two ways:
- **B-4M-v96, narrower rows:** four times as many rows (2048² = 4.19 M), each only 96 values wide. Still 384 rows per
  token, but 36,864 values. Including gradient and Adam the table needs about 6 GiB, so it fits completely on the 16 GB
  card. The idea comes from a read-only review by Codex (2026-10-08); related: UltraMem, which also uses smaller values.
- **B-1M-k8, fewer rows:** B-1M's table (1 M × 384), but 8 instead of 32 lookups per head: 96 rows per token, also
  36,864 values.

**Runs** (`scripts/run_shape.py`, RX 9070, one after another, as a systemd unit):
- Same arguments as B-1M-sparse in `runs/hampter`: Wikipedia, 500 M tokens, seed 0, data seed 1234, batch 32 × 1024,
  LR 6e-4, table LR 2.4e-3, evaluation every 10 M tokens, WikiText-103 as a side value.
- With the Triton kernels. B-1M ran with the PyTorch path back then; the kernels gave the same B-1M result (21.837 at
  home and in the cloud).
- Reference: B-1M-sparse seed 0: 21.837, seed 1: 21.752, mean **21.794** (seed spread 0.4%).
- One seed per variant. The second run only starts if it can end before 11:00.
- Before the runs: the test suite with new tests for rows narrower than the model (kernels and the whole model, Triton
  against PyTorch), and a short trial of each run.

**What else changes (not tuned, named beforehand):**
- B-4M-v96 has twice the sub-keys (2 × 2,048 per head), so 3.1 M more key parameters and twice the key scoring; the
  projections into and out of the table shrink (−0.66 M). Parameters outside the table: 44.7 M instead of 42.2 M (+6%).
- B-4M-v96's table starts with values of size 1/√96 instead of 1/√384, twice as large. With the same table LR every
  Adam step moves them half as much relative to their size. That's one untuned setting; part of a loss could come from
  it.
- B-1M-k8: the product-key search also keeps only the best 8 sub-keys per half (as in Lample's method), so it compares
  fewer candidates.

**Criteria** (Val PPL Wikipedia at the end of the run):
- Each variant against B-1M, r = PPL / 21.794:
  - **"Better with a quarter of the reads":** r < 0.99 (below 21.58).
  - **"As good":** 0.99 ≤ r ≤ 1.01 (21.58 to 22.01).
  - **"Small cost":** 1.01 < r ≤ 1.03 (up to 22.45).
  - **"Clearly worse":** r > 1.03.
- The two against each other: **"narrower rows beat fewer rows"** if PPL(B-4M-v96) ≤ 0.99 × PPL(B-1M-k8), **"fewer
  rows beat narrower rows"** in the opposite case, otherwise **"no clear difference"**.
- Why 1%: the seed spread of B-1M is 0.4%, and each variant runs with one seed.
- **Reported without a verdict:** tokens per second, peak GPU memory, WikiText-103, dense equivalent
  (`scripts/dense_equiv.py`), the course over training.

**Which reads count where:** in training with the table in RAM the bytes moved count, so values read. From the SSD the
number of accesses counts, so rows read: B-1M-k8 reads 96 rows per token, B-4M-v96 still 384.

**What this can't show:** one seed per variant; only this size; the table LR isn't tuned for 96-wide rows; nothing about
bigger tables (B-16M with 96-wide rows would have 1.6 B table parameters).

## Second addendum to step 6: the fact test in the exact training window (criteria before measuring, 2026-10-09)

**Why:** the addendum's context prompt was meant to be the text the model read right before the fact in training. A
read-only review by Codex (2026-10-08) pointed out that it almost never is: training cuts the data into fixed windows of
1,024 tokens, but the prompt took up to 1,000 tokens from the article start. The window in which the model learned the
fact usually starts somewhere else, on average about 500 tokens before the answer, sometimes in the previous article.
For a product-key model another context means other table rows, so a fact could be stored but only reachable from its
own window. That is the last cheap check before the controlled test with repeated facts.

**Setup** (`scripts/eval_fact_window_lm.py`, same items, same eight models, 4-bit table, same scoring):
- **Prompt *window*:** for seen items the training window in which the answer's first token was a target, from the
  window start up to the answer, exactly what the model had in context when it learned the fact. Checked beforehand for
  all 1,382 seen items: each window is in the batch of its training step.
- For unseen items the same 1,024 grid on the validation split, so both prompts have the same length distribution
  (mean 507 tokens for seen, 498 for unseen).
- The answer must lie inside the same window. Items whose answer already appears in the prompt are left out (copying,
  not memory): 1,182 seen and 1,181 unseen remain.

**Criteria** (the addendum's, only the prompt changes; 95% bootstrap intervals over items, paired across models):
- **Does B-16M remember in its own window?** Recency of B-16M (late-seen minus early-seen). Interval above 0: "B-16M
  carries a measurable memory of recently seen articles in their training window". Otherwise: "no measurable memory,
  even in the training window".
- **Table against dense:** gap (seen minus unseen) of B-16M minus that of D-100M, and minus that of D-200M.
  - "The table remembers more in its own window": both intervals above 0.
  - "Dense remembers more": both below 0.
  - "No clear difference": otherwise.
- **Reported without a verdict:** all eight models next to their values with the article-context prompt, hit rates,
  log-probability per fifth of training.

**What this can't show:** the same limits as the addendum (about 220 items per fifth, the recency test only sees
effects of more than about a nat, seen and unseen articles may differ in difficulty).
