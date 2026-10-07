"""Train one model (A / B / C) and log everything needed for the stage-1 report.

Outputs in --out_dir:
  run-info.json        configuration, hardware, seeds, git commit, duration, final results
  metrics.csv          one row per evaluation (by tokens seen): val loss / ppl, train loss, tok/s, VRAM,
                       memory usage statistics (B)
  train_log.csv        every --log_every steps: train loss, lr, grad norms, tok/s
  model.pt             final weights (not in git)
  B only: mem_access_train.npy (reads per entry over the whole training), mem_access_val.npz
          (reads + summed softmax weight per entry on the validation set), mem_index_sample.npz
          (indices / scores / token ids for the first 64k validation tokens, in text order)
"""
import argparse
import csv
import datetime as dt
import json
import math
import os
import platform
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import DATASETS, TrainStream, data_dir, has_split, load_meta, load_split
from .model import ModelConfig, Transformer
from .optim import build_optimizer, clip_grads, lr_multiplier, set_lr, value_table_memory

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS = {
    "A": dict(),
    "B": dict(mem_layers=[6]),
    "C": dict(d_model=768, n_layers=16, n_heads=12, ffn_hidden=2304),
    # v2 "sharpness" variants of B (everything else identical to B)
    "B-v2a": dict(mem_layers=[6], mem_keys_weight_decay=False),
    "B-v2b": dict(mem_layers=[6], mem_keys_weight_decay=False, mem_score_scale="learned"),
    # stage 1b: 3 memory layers (centred, stride 4) sharing one 1024^2 = 1M-entry value table
    "B-1M": dict(mem_layers=[2, 6, 10], mem_n_keys=1024, mem_share_values=True),
    # Hampter: B-1M with row-sparse table gradients and lazy Adam on the table
    "B-1M-sparse": dict(mem_layers=[2, 6, 10], mem_n_keys=1024, mem_share_values=True, mem_value_grad="row_sparse"),
    # cloud: same as B-1M-sparse with 2048^2 = 4M and 4096^2 = 16M table entries
    "B-4M-sparse": dict(mem_layers=[2, 6, 10], mem_n_keys=2048, mem_share_values=True, mem_value_grad="row_sparse"),
    "B-16M-sparse": dict(mem_layers=[2, 6, 10], mem_n_keys=4096, mem_share_values=True, mem_value_grad="row_sparse"),
    # Engram-style n-gram memory instead of product keys: A plus two modules (early and middle layer, as the paper's
    # layers 2 and 15 of ~30), tables together as big as B-1M's (2 x 16 heads x 524,287 rows x 24 = 402.65M)
    "E-1M": dict(eng_layers=[1, 5]),
    # control: E-1M with dense table gradients and plain Adam on all rows (the paper's optimizer) instead of lazy Adam
    "E-1M-dense": dict(eng_layers=[1, 5], eng_value_grad="dense"),
    # both memories: B-1M-sparse's product-key layers plus E-1M's Engram modules (twice the table parameters;
    # ~12 GiB of table and optimizer state alone, doesn't fit a 16 GB card)
    "BE-1M": dict(mem_layers=[2, 6, 10], mem_n_keys=1024, mem_share_values=True, mem_value_grad="row_sparse",
                  eng_layers=[1, 5]),
    # step 1 "Gegenwert der Tabelle": dense Llama models like A without memory, width and depth grown together,
    # head_dim 64, FFN ~ 8/3 d rounded to a multiple of 64 (as A: 384 -> 1024); name = non-embedding parameters
    "D-50M": dict(d_model=640, n_layers=10, n_heads=10, ffn_hidden=1728),
    "D-100M": dict(d_model=768, n_layers=14, n_heads=12, ffn_hidden=2048),
    "D-200M": dict(d_model=1024, n_layers=16, n_heads=16, ffn_hidden=2752),
    "D-400M": dict(d_model=1280, n_layers=20, n_heads=20, ffn_hidden=3456),
}
# micro-batch (sequences) per forward pass; gradient accumulation fills up --batch_seqs
MICRO_BS = {"A": 8, "B": 8, "C": 4, "B-v2a": 8, "B-v2b": 8, "B-1M": 4, "B-1M-sparse": 4, "B-4M-sparse": 4,
            "B-16M-sparse": 4, "E-1M": 4, "E-1M-dense": 4, "BE-1M": 4, "D-50M": 8, "D-100M": 8, "D-200M": 8, "D-400M": 4}

METRIC_FIELDS = [
    "step", "tokens", "epoch", "lr_mult", "train_loss", "val_loss", "val_ppl", "val_word_ppl",
    "train_tok_s", "train_time_s", "wall_time_s", "peak_vram_gib",
    "mem_val_usage", "mem_val_kl", "mem_val_top1pct_share", "mem_train_usage_interval", "mem_train_usage_cum",
    "mem_val_eff_entries", "mem_val_top1_weight", "mem_score_scale_mean", "mem_key_norm_mean",
    "val2_loss", "val2_ppl",
]
LOG_FIELDS = ["step", "tokens", "lr_mult", "loss", "grad_norm", "value_grad_norm", "tok_s"]


def git_info():
    def run(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return None
    # run outputs (runs/, report/) and the report text are not code and do not make the tree "dirty"
    status = run("status", "--porcelain", "--", ".", ":!runs", ":!report", ":!REPORT.md", ":!REPORT.de.md")
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(status), "dirty_files": status.splitlines() if status else []}


def reference_ppl(metrics_csv, tokens):
    """Val PPL of a reference run at `tokens` (its metrics.csv; log-PPL interpolated between evaluations)."""
    with open(metrics_csv) as f:
        rows = list(csv.DictReader(f))
    t = np.array([float(r["tokens"]) for r in rows])
    loss = np.array([float(r["val_loss"]) for r in rows])
    return math.exp(float(np.interp(tokens, t, loss)))


def hardware_info():
    info = {
        "gpu": torch.cuda.get_device_name(0),
        "gpu_arch": torch.cuda.get_device_properties(0).gcnArchName,
        "gpu_mem_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
        "torch": torch.__version__, "hip": torch.version.hip, "python": platform.python_version(),
        "kernel": platform.release(),
    }
    try:
        with open("/proc/cpuinfo") as f:
            info["cpu"] = next(l.split(":", 1)[1].strip() for l in f if l.startswith("model name"))
        with open("/proc/meminfo") as f:
            info["ram_gib"] = round(int(f.readline().split()[1]) / 2**20, 1)
    except Exception:
        pass
    try:
        out = subprocess.run(["rocm-smi", "--showuse", "--showmemuse"], capture_output=True, text=True, timeout=20).stdout
        info["gpu_state_at_start"] = [l.strip() for l in out.splitlines() if "GPU use" in l or "VRAM%" in l]
    except Exception:
        pass
    return info


class MemoryStats:
    """Lample et al. 2019 usage metrics: usage = fraction of slots with z_i != 0, KL(z || uniform).
    Plus softmax sharpness over the top-k: effective number of mixed entries per head = exp(entropy),
    and the mean weight of the best entry (uniform over k=32 would be 1/32)."""

    def __init__(self, size, device):
        self.size = size
        self.counts = torch.zeros(size, dtype=torch.int64, device=device)
        self.z = torch.zeros(size, dtype=torch.float64, device=device)
        self.eff_sum, self.top1_sum, self.n_rows = 0.0, 0.0, 0

    def add(self, mem):
        idx = mem.last_indices.reshape(-1)
        weights = mem.last_weights.float()                          # (N, heads, knn), as used in forward
        w = weights.reshape(-1)
        self.counts += torch.bincount(idx, minlength=self.size)
        self.z += torch.bincount(idx, weights=w.double(), minlength=self.size)
        ent = -(weights * weights.clamp_min(1e-12).log()).sum(-1)
        self.eff_sum += float(ent.exp().sum())
        self.top1_sum += float(weights.max(-1).values.sum())
        self.n_rows += ent.numel()

    def summary(self):
        p = self.z / self.z.sum()
        nz = p > 0
        kl = math.log(self.size) + float((p[nz] * p[nz].log()).sum())
        c = self.counts.sort(descending=True).values.double()
        top = max(1, self.size // 100)
        return {"usage": float((self.counts > 0).double().mean()), "kl": kl,
                "top1pct_share": float(c[:top].sum() / c.sum()),
                "eff_entries_per_head": self.eff_sum / max(1, self.n_rows),
                "top1_weight": self.top1_sum / max(1, self.n_rows)}


@torch.no_grad()
def evaluate(model, split, seq_len, n_words, batch=8, mem_stats=False, sample_windows=0, dataset=None,
             fused_ce=False, ce_chunk_size=1024):
    """Token-level NLL over the whole split (non-overlapping windows, every token but the first).
    Memory statistics: one MemoryStats per memory layer, plus a combined one over all layers (with a
    shared value table this is the usage of the table itself). Index samples: (T, heads, knn) for one
    memory layer, (T, layers, heads, knn) for several."""
    model.eval()
    mems = model.memory_layers()
    for m in mems:
        m.record = mem_stats or bool(sample_windows)
    stats = [MemoryStats(m.size, "cuda") for m in mems] if mem_stats else []
    combined = MemoryStats(mems[0].size, "cuda") if mem_stats and len(mems) > 1 else None
    sample = {"indices": [], "scores": [], "tokens": []}
    tok = torch.from_numpy(np.asarray(load_split(split, dataset), dtype=np.int64)).cuda()
    n = tok.numel() - 1
    full = n // seq_len
    x_all = tok[:full * seq_len].view(full, seq_len)
    y_all = tok[1:full * seq_len + 1].view(full, seq_len)
    chunks = [(x_all[i:i + batch], y_all[i:i + batch]) for i in range(0, full, batch)]
    if n > full * seq_len:
        chunks.append((tok[full * seq_len:n].unsqueeze(0), tok[full * seq_len + 1:n + 1].unsqueeze(0)))
    nll, count, seen_windows = 0.0, 0, 0
    for x, y in chunks:
        if fused_ce:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y, fused_ce=True, ce_chunk_size=ce_chunk_size)
            nll += float(loss) * y.numel()
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            nll += float(F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1), reduction="sum"))
        count += y.numel()
        for s, m in zip(stats, mems):
            s.add(m)
            if combined is not None:
                combined.add(m)
        if sample_windows and seen_windows < sample_windows and mems:
            k = min(x.shape[0], sample_windows - seen_windows)
            L = x.shape[1]
            per = lambda t, m: t.view(x.shape[0], L, m.heads, m.knn)[:k].reshape(-1, m.heads, m.knn)  # noqa: E731
            idx = torch.stack([per(m.last_indices, m) for m in mems], 1).int().cpu()
            sc = torch.stack([per(m.last_scores, m) for m in mems], 1).half().cpu()
            sample["indices"].append(idx[:, 0] if len(mems) == 1 else idx)
            sample["scores"].append(sc[:, 0] if len(mems) == 1 else sc)
            sample["tokens"].append(x[:k].reshape(-1).int().cpu())
        seen_windows += x.shape[0]
    for m in mems:
        m.record = False
    model.train()
    loss = nll / count
    out = {"loss": loss, "ppl": math.exp(loss), "word_ppl": math.exp(nll / n_words), "n_tokens": count}
    if stats:
        main_stats = combined if combined is not None else stats[0]
        out["mem"] = main_stats.summary()
        out["mem_layers"] = [s.summary() for s in stats]
        out["mem_counts"] = main_stats.counts.cpu().numpy()
        out["mem_z"] = main_stats.z.float().cpu().numpy()
    if sample_windows and sample["indices"]:
        out["sample"] = {k: torch.cat(v).numpy() for k, v in sample.items()}
    return out


@torch.no_grad()
def inference_benchmark(model, seq_len, prompt_len=128, new_tokens=256, prefill_batch=16, dataset=None):
    """Batch-1 greedy decoding with KV cache, and batched full-sequence forward (prefill / scoring)."""
    model.eval()
    val = torch.from_numpy(np.asarray(load_split("validation", dataset)[:prompt_len], dtype=np.int64)).cuda()[None]
    torch.cuda.reset_peak_memory_stats()
    res = {}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for rep in range(2):                                   # first repetition = warmup
            caches = [dict() for _ in range(model.cfg.n_layers)]
            logits = model(val, kv_caches=caches, pos0=0)
            nxt = logits[:, -1:].argmax(-1)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for i in range(new_tokens):
                logits = model(nxt, kv_caches=caches, pos0=prompt_len + i)
                nxt = logits[:, -1:].argmax(-1)
            torch.cuda.synchronize()
            res["decode_b1_tok_s"] = new_tokens / (time.perf_counter() - t0)
        res["decode_peak_vram_gib"] = torch.cuda.max_memory_allocated() / 2**30
        torch.cuda.reset_peak_memory_stats()
        x = torch.randint(0, 50257, (prefill_batch, seq_len), device="cuda")
        model(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        reps = 5
        for _ in range(reps):
            model(x)
        torch.cuda.synchronize()
        res["prefill_tok_s"] = reps * prefill_batch * seq_len / (time.perf_counter() - t0)
        res["prefill_peak_vram_gib"] = torch.cuda.max_memory_allocated() / 2**30
    model.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tokens", type=float, default=None, help="training token budget")
    ap.add_argument("--epochs", type=float, default=None, help="alternative to --tokens")
    ap.add_argument("--seed", type=int, default=0, help="init seed")
    ap.add_argument("--data_seed", type=int, default=1234)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--batch_seqs", type=int, default=32)
    ap.add_argument("--micro_bs", type=int, default=None)
    ap.add_argument("--fused_ce", action="store_true", help="chunk the output projection and CE, including backward")
    ap.add_argument("--ce_chunk_size", type=int, default=1024, help="tokens per output-projection chunk with --fused_ce")
    ap.add_argument("--compile", action="store_true", help="compile dense block parts; memory and KV caches stay eager")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--value_lr", type=float, default=1e-3)
    ap.add_argument("--eng_value_lr", type=float, default=None, help="Engram tables (default: --value_lr)")
    ap.add_argument("--value_state", choices=["fp32", "bf16", "int8"], default="fp32",
                    help="LazyRowAdam moment storage for PKM and Engram tables (requires row_sparse gradients)")
    ap.add_argument("--value_device", choices=["gpu", "host"], default="gpu",
                    help="keep PKM tables and lazy Adam state in host RAM (requires row_sparse PKM)")
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--min_lr_ratio", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval_every_tokens", type=float, default=4e6)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--sample_windows", type=int, default=64, help="B: val windows whose indices are saved")
    ap.add_argument("--mem_query_norm", default="batchnorm")
    ap.add_argument("--mem_score_scale_init", type=float, default=None, help="v2b: initial per-head score scale")
    ap.add_argument("--data", default=None, choices=list(DATASETS), help="training / primary val dataset "
                    "(default: wikitext103, or SMLM_DATA_DIR)")
    ap.add_argument("--extra_val", default=None, choices=list(DATASETS), help="second validation set (val2_*)")
    ap.add_argument("--mem_impl", default=None, choices=["torch", "triton"], help="memory layer: PyTorch "
                    "reference or Triton kernels (default: model preset, i.e. torch)")
    ap.add_argument("--abort_ref", default=None, help="metrics.csv of a reference run: stop with exit code 3 if "
                    "the val PPL at the evaluation nearest --abort_at_tokens is more than --abort_max_rel worse "
                    "than the reference's val PPL at the same token count")
    ap.add_argument("--abort_at_tokens", type=float, default=100e6)
    ap.add_argument("--stop_after_tokens", type=float, default=None, help="end training early (after the evaluation "
                    "at this token count) but keep the schedule of --tokens/--epochs (kernel comparison runs)")
    ap.add_argument("--abort_max_rel", type=float, default=0.05)
    ap.add_argument("--no_save", action="store_true", help="do not write model.pt (benchmark runs)")
    args = ap.parse_args()
    micro_bs = MICRO_BS[args.model] if args.micro_bs is None else args.micro_bs
    if micro_bs <= 0 or args.batch_seqs <= 0 or args.batch_seqs % micro_bs:
        ap.error("--micro_bs must be positive and divide --batch_seqs")
    if args.ce_chunk_size <= 0:
        ap.error("--ce_chunk_size must be positive")

    os.makedirs(args.out_dir, exist_ok=True)
    t_start = time.time()
    meta = load_meta(args.data)
    stream = TrainStream(args.seq_len, args.batch_seqs, args.data_seed, dataset=args.data)
    if args.epochs is not None:
        total_steps = int(round(args.epochs * stream.steps_per_epoch))
    else:
        total_steps = int(args.tokens) // stream.tokens_per_step
    warmup_steps = max(1, int(round(args.warmup_frac * total_steps)))
    eval_every = max(1, int(round(args.eval_every_tokens / stream.tokens_per_step)))
    abort_step = (int(round(args.abort_at_tokens / stream.tokens_per_step / eval_every)) * eval_every
                  if args.abort_ref else None)
    accum = args.batch_seqs // micro_bs

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    mcfg = ModelConfig(max_seq_len=args.seq_len, **MODELS[args.model])
    if mcfg.mem_layers:
        mcfg.mem_query_norm = args.mem_query_norm
        if args.mem_score_scale_init is not None:
            mcfg.mem_score_scale_init = args.mem_score_scale_init
        if args.mem_impl is not None:
            mcfg.mem_impl = args.mem_impl
    model = Transformer(mcfg)
    if args.value_device == "host":
        from .host_values import enable_host_values
        model = enable_host_values(model)
    model = model.cuda()
    if args.compile:
        from .compile import compile_dense
        compile_dense(model)
    loss_options = {"fused_ce": True, "ce_chunk_size": args.ce_chunk_size} if args.fused_ce else {}
    opt = build_optimizer(model, args.lr, args.value_lr, args.weight_decay, eng_value_lr=args.eng_value_lr,
                          value_state=args.value_state)
    mems = model.memory_layers()
    n_val_words = meta["splits"]["validation"]["n_words"]
    n_val2_words = load_meta(args.extra_val)["splits"]["validation"]["n_words"] if args.extra_val else None

    info = {
        "status": "running",
        "model_name": args.model,
        "started": dt.datetime.now().isoformat(timespec="seconds"),
        "git": git_info(),
        "hardware": hardware_info(),
        "seeds": {"init_seed": args.seed, "data_seed": args.data_seed},
        "train_config": {**vars(args), "total_steps": total_steps, "warmup_steps": warmup_steps,
                         "tokens_per_step": stream.tokens_per_step, "micro_bs": micro_bs, "grad_accum": accum,
                         "total_tokens": total_steps * stream.tokens_per_step,
                         "epochs": total_steps / stream.steps_per_epoch, "eval_every_steps": eval_every,
                         "optimizer": "AdamW(fused) betas=(0.9,0.95) eps=1e-8; memory values: lr=value_lr, wd=0, separate clip"
                                      + ("; value table: row-sparse gradients + LazyRowAdam (only rows read in the step "
                                         "are updated, values and Adam state of unread rows unchanged)"
                                         if ((mcfg.mem_layers and mcfg.mem_value_grad == "row_sparse") or
                                             (mcfg.eng_layers and mcfg.eng_value_grad == "row_sparse")) else ""),
                         "schedule": "linear warmup, cosine to min_lr_ratio, same multiplier for all groups",
                         "precision": ("bf16 autocast, fp32 weights / optimizer state" if args.value_state == "fp32"
                                       else "bf16 autocast, fp32 weights / dense optimizer state; "
                                            f"{args.value_state} lazy Adam moments")},
        "model_config": mcfg.to_dict(),
        "params": model.param_counts(),
        "value_table_memory": value_table_memory(model, args.value_state),
        "macs_per_token": model.macs_per_token(),
        "data": {"dataset": args.data or "wikitext103", "data_dir": data_dir(args.data), **meta,
                 **({"extra_val": args.extra_val} if args.extra_val else {})},
    }

    if args.value_device == "host":
        from .host_optim import value_table_memory_by_device
        info["value_table_memory_by_device"] = value_table_memory_by_device(model, args.value_state)

    def write_info():
        with open(os.path.join(args.out_dir, "run-info.json"), "w") as f:
            json.dump(info, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    write_info()
    print(json.dumps({k: info[k] for k in ("model_name", "params", "seeds", "value_table_memory")},
                     default=str), flush=True)
    print(f"steps={total_steps} warmup={warmup_steps} eval_every={eval_every} accum={accum}", flush=True)

    mfile = open(os.path.join(args.out_dir, "metrics.csv"), "w", newline="")
    mcsv = csv.DictWriter(mfile, fieldnames=METRIC_FIELDS)
    mcsv.writeheader()
    lfile = open(os.path.join(args.out_dir, "train_log.csv"), "w", newline="")
    lcsv = csv.DictWriter(lfile, fieldnames=LOG_FIELDS)
    lcsv.writeheader()

    size = mems[0].size if mems else 0
    acc_interval = torch.zeros(size, dtype=torch.int64, device="cuda") if mems else None
    acc_total = torch.zeros(size, dtype=torch.int64, device="cuda") if mems else None
    for m in mems:
        m.record = True

    train_time = 0.0
    interval_loss, interval_steps = torch.zeros((), device="cuda"), 0
    log_loss = torch.zeros((), device="cuda")
    log_gn, log_vgn = torch.zeros((), device="cuda"), torch.zeros((), device="cuda")
    last_log_t, steps_since_t = None, 0     # throughput window; evaluation time is excluded
    tok_s_hist = []

    peak_train_vram = 0.0

    def do_eval(step):
        nonlocal interval_loss, interval_steps, peak_train_vram
        # training peak since the last evaluation (evaluation itself is not counted)
        peak_train_vram = max(peak_train_vram, torch.cuda.max_memory_allocated() / 2**30)
        ev = evaluate(model, "validation", args.seq_len, n_val_words, batch=micro_bs, mem_stats=bool(mems),
                      dataset=args.data, **loss_options)
        ev2 = evaluate(model, "validation", args.seq_len, n_val2_words, batch=micro_bs,
                       dataset=args.extra_val, **loss_options) if args.extra_val else None
        for m in mems:
            m.record = True
        row = {
            "step": step, "tokens": step * stream.tokens_per_step, "epoch": round(step / stream.steps_per_epoch, 4),
            "lr_mult": lr_multiplier(max(0, step - 1), total_steps, warmup_steps, args.min_lr_ratio) if step else 0.0,
            "train_loss": float(interval_loss) / interval_steps if interval_steps else "",
            "val_loss": ev["loss"], "val_ppl": ev["ppl"], "val_word_ppl": ev["word_ppl"],
            "train_tok_s": (np.mean(tok_s_hist[-max(1, eval_every // args.log_every):]) if tok_s_hist else ""),
            "train_time_s": round(train_time, 1), "wall_time_s": round(time.time() - t_start, 1),
            "peak_vram_gib": round(peak_train_vram, 3) if step else "",
            **({"val2_loss": ev2["loss"], "val2_ppl": ev2["ppl"]} if ev2 else {}),
        }
        if mems:
            row.update({"mem_val_usage": ev["mem"]["usage"], "mem_val_kl": ev["mem"]["kl"],
                        "mem_val_top1pct_share": ev["mem"]["top1pct_share"],
                        "mem_val_eff_entries": ev["mem"]["eff_entries_per_head"],
                        "mem_val_top1_weight": ev["mem"]["top1_weight"],
                        "mem_score_scale_mean": float(torch.stack([m.score_scale() for m in mems]).mean()),
                        "mem_key_norm_mean": float(torch.stack(
                            [m.keys.detach().norm(dim=-1).mean() for m in mems]).mean())})
            if step:
                row["mem_train_usage_interval"] = float((acc_interval > 0).double().mean())
                row["mem_train_usage_cum"] = float((acc_total > 0).double().mean())
                acc_interval.zero_()
        mcsv.writerow(row)
        mfile.flush()
        if abort_step is not None and step == abort_step:
            ref = reference_ppl(args.abort_ref, row["tokens"])
            rel = ev["ppl"] / ref - 1
            info["abort_check"] = {"step": step, "tokens": row["tokens"], "val_ppl": ev["ppl"], "ref_val_ppl": ref,
                                   "rel_diff": rel, "max_rel": args.abort_max_rel, "ref": args.abort_ref,
                                   "aborted": rel > args.abort_max_rel}
            print(f"[abort check] tokens {row['tokens']/1e6:.2f}M val_ppl {ev['ppl']:.3f} vs reference {ref:.3f} "
                  f"({100 * rel:+.2f} %, limit {100 * args.abort_max_rel:+.1f} %)", flush=True)
            if rel > args.abort_max_rel:
                info.update({"status": "aborted", "finished": dt.datetime.now().isoformat(timespec="seconds"),
                             "duration_s": round(time.time() - t_start, 1), "train_time_s": round(train_time, 1)})
                write_info()
                torch.save({"model_config": mcfg.to_dict(), "state_dict": model.state_dict()},
                           os.path.join(args.out_dir, "model.pt"))
                raise SystemExit(3)
            write_info()
        interval_loss.zero_()
        interval_steps = 0
        torch.cuda.reset_peak_memory_stats()
        print(f"[eval] step {step} tokens {row['tokens']/1e6:.1f}M val_ppl {ev['ppl']:.3f} "
              f"train_loss {row['train_loss'] if row['train_loss']=='' else round(row['train_loss'],4)} "
              + (f"usage {ev['mem']['usage']:.3f} kl {ev['mem']['kl']:.3f}" if mems else ""), flush=True)
        return ev

    do_eval(0)
    model.train()
    torch.cuda.synchronize()
    last_log_t = time.perf_counter()
    for step in range(total_steps):
        set_lr(opt, lr_multiplier(step, total_steps, warmup_steps, args.min_lr_ratio))
        batch = torch.from_numpy(stream.batch(step).astype(np.int64)).pin_memory().cuda(non_blocking=True)
        for i in range(accum):
            mb = batch[i * micro_bs:(i + 1) * micro_bs]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(mb[:, :-1], mb[:, 1:], **loss_options)
            (loss / accum).backward()
            interval_loss += loss.detach() / accum
            log_loss += loss.detach() / accum
            if mems:
                # reads per table entry; with several layers sharing one table the reads are summed
                for m in mems:
                    cnt = torch.bincount(m.last_indices.reshape(-1), minlength=size)
                    acc_interval += cnt
                    acc_total += cnt
        gn, vgn = clip_grads(model, args.clip)
        log_gn += gn
        if vgn is not None:
            log_vgn += vgn
        opt.step()
        opt.zero_grad(set_to_none=True)
        interval_steps += 1
        steps_since_t += 1
        if (step + 1) % args.log_every == 0 or step + 1 == total_steps:
            n_log = (step % args.log_every) + 1
            vals = torch.stack([log_loss, log_gn, log_vgn]).tolist()   # sync point
            now = time.perf_counter()
            train_time += now - last_log_t
            tok_s = steps_since_t * stream.tokens_per_step / (now - last_log_t)
            if step >= 2 * args.log_every:   # skip warm-up kernels / autotuning
                tok_s_hist.append(tok_s)
            lcsv.writerow({"step": step + 1, "tokens": (step + 1) * stream.tokens_per_step,
                           "lr_mult": lr_multiplier(step, total_steps, warmup_steps, args.min_lr_ratio),
                           "loss": vals[0] / n_log, "grad_norm": vals[1] / n_log,
                           "value_grad_norm": vals[2] / n_log if mems else "", "tok_s": round(tok_s)})
            lfile.flush()
            log_loss.zero_(); log_gn.zero_(); log_vgn.zero_()
            last_log_t, steps_since_t = now, 0
            if (step + 1) % (args.log_every * 10) == 0:
                print(f"step {step+1}/{total_steps} loss {vals[0]/n_log:.4f} tok/s {tok_s:.0f}", flush=True)
            if not math.isfinite(vals[0]):
                info["status"] = "diverged"
                write_info()
                raise SystemExit(f"non-finite loss at step {step+1}")
        if (step + 1) % eval_every == 0 or step + 1 == total_steps:
            torch.cuda.synchronize()
            now = time.perf_counter()
            train_time += now - last_log_t
            do_eval(step + 1)
            torch.cuda.synchronize()
            last_log_t, steps_since_t = time.perf_counter(), 0   # evaluation excluded from throughput
            if args.stop_after_tokens and (step + 1) * stream.tokens_per_step >= args.stop_after_tokens:
                info["stopped_early_at_step"] = step + 1
                break

    final_val = evaluate(model, "validation", args.seq_len, n_val_words, batch=micro_bs, mem_stats=bool(mems),
                         sample_windows=args.sample_windows if mems else 0, dataset=args.data, **loss_options)
    final_test = (evaluate(model, "test", args.seq_len, meta["splits"]["test"]["n_words"], batch=micro_bs,
                           dataset=args.data, **loss_options) if has_split("test", args.data) else None)
    final_val2 = (evaluate(model, "validation", args.seq_len, n_val2_words, batch=micro_bs,
                           dataset=args.extra_val, **loss_options) if args.extra_val else None)
    if not args.no_save:
        torch.save({"model_config": mcfg.to_dict(), "state_dict": model.state_dict()},
                   os.path.join(args.out_dir, "model.pt"))
    if mems:
        np.save(os.path.join(args.out_dir, "mem_access_train.npy"), acc_total.cpu().numpy())
        np.savez_compressed(os.path.join(args.out_dir, "mem_access_val.npz"),
                            counts=final_val["mem_counts"], z=final_val["mem_z"])
    if mems and "sample" in final_val:
        s = final_val["sample"]
        np.savez(os.path.join(args.out_dir, "mem_index_sample.npz"), indices=s["indices"], scores=s["scores"],
                 layer_ids=np.array(mcfg.mem_layers),
                 tokens=s["tokens"], note="indices[t, (layer,) head, j] read for val token t (text order, first "
                 f"{args.sample_windows} windows of {args.seq_len}); layer axis only with several memory "
                 "layers (order = layer_ids); scores = raw key scores before softmax")

    del opt
    for p in model.parameters():
        p.grad = None
        if hasattr(p, "row_store"):
            del p.row_store                     # row-sparse gradient accumulator (training only)
    torch.cuda.empty_cache()
    infer = inference_benchmark(model, args.seq_len, dataset=args.data)
    tok_s_mean = float(np.mean(tok_s_hist)) if tok_s_hist else None
    info.update({
        "status": "done",
        "finished": dt.datetime.now().isoformat(timespec="seconds"),
        "duration_s": round(time.time() - t_start, 1),
        "train_time_s": round(train_time, 1),
        "results": {
            "val_loss": final_val["loss"], "val_ppl": final_val["ppl"], "val_word_ppl": final_val["word_ppl"],
            **({"test_loss": final_test["loss"], "test_ppl": final_test["ppl"],
                "test_word_ppl": final_test["word_ppl"]} if final_test else {}),
            **({"val2_dataset": args.extra_val, "val2_loss": final_val2["loss"], "val2_ppl": final_val2["ppl"],
                "val2_word_ppl": final_val2["word_ppl"]} if final_val2 else {}),
            "train_tok_s_mean": tok_s_mean,
            "train_tok_s_median": float(np.median(tok_s_hist)) if tok_s_hist else None,
            "peak_train_vram_gib": peak_train_vram,
            "weights_gib_fp32": sum(p.numel() for p in model.parameters()) * 4 / 2**30,
            **infer,
            **({"mem_val": final_val["mem"],
                "mem_val_per_layer": final_val["mem_layers"],
                "mem_train_usage_total": float((acc_total > 0).double().mean()),
                "mem_score_scale_per_head": mems[0].score_scale().tolist(),
                "mem_score_scale_per_layer_head": [m.score_scale().tolist() for m in mems],
                "mem_key_norm_mean": float(torch.stack(
                    [m.keys.detach().norm(dim=-1).mean() for m in mems]).mean())} if mems else {}),
        },
    })
    write_info()
    mfile.close()
    lfile.close()
    print(json.dumps(info["results"], indent=1), flush=True)


if __name__ == "__main__":
    main()
