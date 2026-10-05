"""Measure what the Triton kernels bring on your GPU. Under a minute on my RX 9070, no data needed.

  python scripts/kernel_speedup.py                 # B-1M (1M-row table), plus the same model without a table

Builds the model three times (no table, table with the PyTorch reference, table with the Triton kernels) and times on
random tokens:
  train    one optimiser step on 32 x 1024 tokens (micro-batches + gradient accumulation, bf16 autocast)
  prefill  a forward pass over 16 x 1024 tokens
  decode   batch-1 generation with KV cache (the Triton version uses the decode graphs)
Weights are random, so this measures speed only; correctness is what the tests are for. Needs ~11 GB of GPU memory.
PYTORCH_CUDA_ALLOC_CONF is ignored here: with expandable_segments:True the prefill after training died with a
hardware exception on my RX 9070 (ROCm 7.2). Without it everything runs. I haven't looked into why yet.
"""
import argparse
import os
import platform
import sys
import time

os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)       # see above
import torch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from smlm.model import ModelConfig, Transformer  # noqa: E402
from smlm.optim import build_optimizer, clip_grads  # noqa: E402
from smlm.train import MICRO_BS, MODELS  # noqa: E402

SEQ, BATCH = 1024, 32


def timed(fn, reps, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


def measure(name, impl, steps):
    torch.manual_seed(0)
    cfg = ModelConfig(max_seq_len=SEQ, **MODELS[name])
    if cfg.mem_layers:
        cfg.mem_impl = impl
    model = Transformer(cfg).cuda()
    opt = build_optimizer(model, 6e-4, 2.4e-3, 0.1)
    mb = MICRO_BS[name]
    g = torch.Generator(device="cuda").manual_seed(1)
    batch = torch.randint(0, 50257, (BATCH, SEQ + 1), device="cuda", generator=g)
    torch.cuda.reset_peak_memory_stats()

    def train_step():
        for i in range(BATCH // mb):
            x = batch[i * mb:(i + 1) * mb]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x[:, :-1], x[:, 1:])
            (loss / (BATCH // mb)).backward()
        clip_grads(model, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

    model.train()
    print(f"  measuring {name}{f' ({impl})' if cfg.mem_layers else ''} ...", flush=True)
    t_train = timed(train_step, steps, 3)
    peak = torch.cuda.max_memory_allocated() / 2**30
    del opt
    for p in model.parameters():
        p.grad = None
        if hasattr(p, "row_store"):
            del p.row_store
    torch.cuda.empty_cache()
    model.eval()
    if cfg.mem_layers and impl == "triton":
        model.set_memory_decode_graphs(True)
    xp = batch[:16, :SEQ]

    @torch.no_grad()
    def prefill():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(xp)

    t_prefill = timed(prefill, 5, 2)

    @torch.no_grad()
    def decode():
        caches = [dict() for _ in range(cfg.n_layers)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(batch[:1, :128], kv_caches=caches, pos0=0)
            nxt = logits[:, -1:].argmax(-1)
            for i in range(128):
                logits = model(nxt, kv_caches=caches, pos0=128 + i)
                nxt = logits[:, -1:].argmax(-1)

    t_decode = timed(decode, 2, 1) / 129
    del model
    torch.cuda.empty_cache()
    return {"train_ms": 1000 * t_train, "train_tok_s": BATCH * SEQ / t_train, "prefill_tok_s": 16 * SEQ / t_prefill,
            "decode_tok_s": 1 / t_decode, "peak_gib": peak}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="B-1M-sparse", choices=[m for m in MODELS if MODELS[m].get("mem_layers")])
    ap.add_argument("--steps", type=int, default=5)
    args = ap.parse_args()
    import triton
    p = torch.cuda.get_device_properties(0)
    arch = getattr(p, "gcnArchName", "")
    print(f"{p.name}{f' ({arch})' if arch else ''}, {p.total_memory / 2**30:.0f} GB | PyTorch {torch.__version__} "
          f"({'HIP ' + torch.version.hip if torch.version.hip else 'CUDA ' + str(torch.version.cuda)}) | Triton "
          f"{triton.__version__} | Python {platform.python_version()}\n")
    rows = [("no table (A)", measure("A", "torch", args.steps)),
            (f"{args.model}, PyTorch", measure(args.model, "torch", args.steps)),
            (f"{args.model}, Triton kernels", measure(args.model, "triton", args.steps))]
    print(f"{'':28s} {'train step':>11s} {'train tok/s':>12s} {'prefill tok/s':>14s} {'decode tok/s':>13s} {'peak GB':>8s}")
    for name, r in rows:
        print(f"{name:28s} {r['train_ms']:9.0f} ms {r['train_tok_s']:12,.0f} {r['prefill_tok_s']:14,.0f} "
              f"{r['decode_tok_s']:13.0f} {r['peak_gib']:8.1f}")
    a, ref, tri = (r for _, r in rows)
    print(f"\nkernels vs PyTorch: training {ref['train_ms'] / tri['train_ms']:.2f}x, prefill "
          f"{tri['prefill_tok_s'] / ref['prefill_tok_s']:.2f}x, decode {tri['decode_tok_s'] / ref['decode_tok_s']:.2f}x")
    print(f"with the kernels the table model trains at {a['train_ms'] / tri['train_ms']:.0%} of the speed of the model "
          f"without a table")


if __name__ == "__main__":
    main()
