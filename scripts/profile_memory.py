"""Where does the time go? B-1M-sparse vs A: training step, prefill, decode, plus a stage-by-stage breakdown
of one memory layer at training (N=4096), prefill (N=16384) and decode (N=1) shapes. ~2 min GPU.

  python scripts/profile_memory.py [--out report/profile_before.json]

Uses the trained checkpoints (runs/hampter/B-1M-sparse-s0, runs/s1b/A-s0; read only) and real training
batches, so the index distribution (hot rows) is the one of a trained model. Times are medians of CUDA-event
measurements after warm-up.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from smlm.data import TrainStream  # noqa: E402
from smlm.model import ModelConfig, Transformer  # noqa: E402
from smlm.optim import build_optimizer, clip_grads  # noqa: E402
from smlm.sparse_values import LazyRowAdam, _RowSparseBag, row_store  # noqa: E402
from smlm.train import MODELS, MICRO_BS  # noqa: E402

CKPT = {"B-1M-sparse": "runs/hampter/B-1M-sparse-s0/model.pt", "A": "runs/s1b/A-s0/model.pt"}


def bench(fn, reps=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return float(np.median(ts))


def load(name):
    cfg = ModelConfig(max_seq_len=1024, **MODELS[name])
    model = Transformer(cfg).cuda()
    sd = torch.load(os.path.join(ROOT, CKPT[name]), map_location="cuda", weights_only=False)["state_dict"]
    model.load_state_dict(sd)
    return model


def train_step_profile(name, stream, steps=6, warmup=3):
    """One optimizer step exactly as in train.py (8 micro-batches, usage bincounts, clip, step), with
    CUDA events between the phases. Returns median ms per phase."""
    model = load(name).train()
    opt = build_optimizer(model, 6e-4, 2.4e-3, 0.1)
    mems = model.memory_layers()
    size = mems[0].size if mems else 0
    acc_i = torch.zeros(size, dtype=torch.int64, device="cuda") if mems else None
    acc_t = torch.zeros(size, dtype=torch.int64, device="cuda") if mems else None
    for m in mems:
        m.record = True
    micro = MICRO_BS[name]
    accum = 32 // micro
    rows = []
    for step in range(warmup + steps):
        batch = torch.from_numpy(stream.batch(1000 + step).astype(np.int64)).pin_memory().cuda(non_blocking=True)
        ev = {k: torch.cuda.Event(enable_timing=True) for k in ("s", "f", "b", "st", "c", "o1", "o2")}
        t = {"fwd": 0.0, "bwd": 0.0, "stats": 0.0}
        torch.cuda.synchronize()
        ev["s"].record()
        for i in range(accum):
            mb = batch[i * micro:(i + 1) * micro]
            e0, e1, e2, e3 = (torch.cuda.Event(enable_timing=True) for _ in range(4))
            e0.record()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(mb[:, :-1], mb[:, 1:])
            e1.record()
            (loss / accum).backward()
            e2.record()
            for m in mems:
                cnt = torch.bincount(m.last_indices.reshape(-1), minlength=size)
                acc_i += cnt
                acc_t += cnt
            e3.record()
            torch.cuda.synchronize()
            t["fwd"] += e0.elapsed_time(e1)
            t["bwd"] += e1.elapsed_time(e2)
            t["stats"] += e2.elapsed_time(e3)
        ev["c"].record()
        clip_grads(model, 1.0)
        ev["o1"].record()
        opts = opt.opts if hasattr(opt, "opts") else [opt]
        e_split = []
        for o in opts:
            o.step()
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            e_split.append(e)
        opt.zero_grad(set_to_none=True)
        ev["o2"].record()
        torch.cuda.synchronize()
        if step >= warmup:
            r = dict(t)
            r["clip"] = ev["c"].elapsed_time(ev["o1"])
            r["opt_adamw"] = ev["o1"].elapsed_time(e_split[0])
            r["opt_lazy_table"] = e_split[0].elapsed_time(e_split[1]) if len(e_split) > 1 else 0.0
            r["step_total"] = ev["s"].elapsed_time(ev["o2"])
            rows.append(r)
    out = {k: float(np.median([r[k] for r in rows])) for k in rows[0]}
    out["tok_s"] = 32 * 1024 / (out["step_total"] / 1000)
    del opt, model
    torch.cuda.empty_cache()
    return out


def memory_layer_stages(mem, x, grad=True, reps=10):
    """Forward stages of ProductKeyMemory.forward, timed one by one (same code as pkm.py)."""
    N = x.shape[0]
    res = {}
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx, torch.autocast("cuda", dtype=torch.bfloat16):
        def qstage():
            q = mem.query_proj(x)
            return mem.query_norm(q.to(mem.query_norm.weight.dtype)).view(N, mem.heads, mem.k_dim)
        res["query_proj_bn"] = bench(qstage, reps)
        q = qstage()
        half = mem.k_dim // 2
        k1, k2 = mem.keys[:, 0], mem.keys[:, 1]
        res["subkey_scores_2x"] = bench(lambda: (torch.einsum("nhd,hkd->nhk", q[..., :half], k1),
                                                 torch.einsum("nhd,hkd->nhk", q[..., half:], k2)), reps)
        s1 = torch.einsum("nhd,hkd->nhk", q[..., :half], k1)
        s2 = torch.einsum("nhd,hkd->nhk", q[..., half:], k2)
        res["topk_halves_2x"] = bench(lambda: (s1.topk(mem.knn, dim=-1), s2.topk(mem.knn, dim=-1)), reps)
        (a1, i1), (a2, i2) = s1.topk(mem.knn, dim=-1), s2.topk(mem.knn, dim=-1)

        def cart():
            all_s = (a1.unsqueeze(-1) + a2.unsqueeze(-2)).view(N, mem.heads, -1)
            all_i = (i1.unsqueeze(-1) * mem.n_keys + i2.unsqueeze(-2)).view(N, mem.heads, -1)
            sc, best = all_s.topk(mem.knn, dim=-1)
            return sc, all_i.gather(-1, best)
        res["cartesian_topk"] = bench(cart, reps)
        scores, idx = cart()
        res["softmax"] = bench(lambda: F.softmax(scores.float(), dim=-1), reps)
        w = F.softmax(scores.float(), dim=-1)
        res["value_bag_fwd"] = bench(lambda: mem.read_values(idx.view(N, -1), w.view(N, -1)), reps)
        out = mem.read_values(idx.view(N, -1), w.view(N, -1))
        res["swilu"] = bench(lambda: mem.value_proj(out * F.silu(mem.swilu_proj(x)).to(out.dtype)), reps)
        res["forward_total"] = bench(lambda: mem(x), reps)
    return res


def memory_layer_backward(mem, x, reps=10):
    """Backward of one memory layer at training shape; the row-sparse value backward separately."""
    res = {}
    table = mem.values.weight

    def fb():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = mem(x)
        y.float().pow(2).mean().backward()
    res["fwd_bwd_total"] = bench(fb, reps)
    st = row_store(table)
    st.reset()
    # the value-bag backward alone (per-sample-weight grads + accumulation into the row store)
    N = x.shape[0]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        q = mem.query_norm(mem.query_proj(x).float()).view(N, mem.heads, mem.k_dim)
        scores, idx = mem.get_indices(q)
    w = F.softmax(scores.float(), dim=-1).view(N, -1).detach().requires_grad_()
    g = torch.randn(N, table.shape[1], device="cuda")

    def bag_bwd():
        out = _RowSparseBag.apply(w, idx.view(N, -1), table.detach(), st)
        out.backward(g)
    res["value_bag_fwd_bwd"] = bench(bag_bwd, reps)
    res["value_bag_fwd"] = bench(lambda: _RowSparseBag.apply(w.detach(), idx.view(N, -1), table.detach(), st), reps)
    st.reset()
    # profiler view of one layer forward + backward
    from torch.profiler import ProfilerActivity, profile
    for _ in range(2):
        fb()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            fb()
        torch.cuda.synchronize()
    ka = prof.key_averages()
    attr = "device_time_total" if hasattr(ka[0], "device_time_total") else "cuda_time_total"
    top = sorted(ka, key=lambda e: getattr(e, attr), reverse=True)
    res["profiler_top_ops_ms_per_fwd_bwd"] = [(e.key, round(getattr(e, attr) / 1000 / 3, 3)) for e in top[:30]]
    st.reset()
    return res


def lazy_adam_alone(table, reps=10):
    """LazyRowAdam step on the 1M table with ~all rows touched (as in training)."""
    st = row_store(table)
    opt = LazyRowAdam([table], lr=1e-3)

    def step():
        st.acc.normal_()
        st.touched.fill_(True)
        st.touched[::50] = False             # ~2 % unread rows
        opt.step()
    t = bench(step, reps)
    t_fill = bench(lambda: (st.acc.normal_(), st.touched.fill_(True)), reps)
    st.reset()
    return {"lazy_adam_step_ms": t - t_fill}


def ffn_reference(model_a, x, reps=10):
    blk = model_a.layers[2]
    ffn = blk.ffn

    def fb():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = ffn(x)
        y.float().pow(2).mean().backward()

    def f():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ffn(x)
    return {"ffn_fwd_bwd": bench(fb, reps), "ffn_fwd": bench(f, reps)}


@torch.no_grad()
def inference(model, prompt, reps=5, new_tokens=64):
    model.eval()
    res = {}
    x = torch.randint(0, 50257, (16, 1024), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        res["prefill_16x1024_ms"] = bench(lambda: model(x), reps, warmup=2)

        def decode():
            caches = [dict() for _ in range(model.cfg.n_layers)]
            logits = model(prompt, kv_caches=caches, pos0=0)
            nxt = logits[:, -1:].argmax(-1)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for i in range(new_tokens):
                logits = model(nxt, kv_caches=caches, pos0=prompt.shape[1] + i)
                nxt = logits[:, -1:].argmax(-1)
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / new_tokens * 1000
        decode()
        res["decode_ms_per_token"] = float(np.median([decode() for _ in range(3)]))
    res["prefill_tok_s"] = 16 * 1024 / (res["prefill_16x1024_ms"] / 1000)
    res["decode_tok_s"] = 1000 / res["decode_ms_per_token"]
    model.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.manual_seed(0)
    stream = TrainStream(1024, 32, 1234, dataset="wikipedia")
    batch = torch.from_numpy(stream.batch(2000).astype(np.int64)).cuda()
    out = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}

    out["train_step_B"] = train_step_profile("B-1M-sparse", stream)
    out["train_step_A"] = train_step_profile("A", stream)
    print(json.dumps({k: out[k] for k in ("train_step_B", "train_step_A")}, indent=1), flush=True)

    model_b = load("B-1M-sparse")
    mem = model_b.memory_layers()[1]                      # middle memory layer (layer index 6)
    # hidden states entering the memory layer of a real batch (4 sequences = one training micro-batch)
    acts = {}
    hook = mem.register_forward_pre_hook(lambda m, i: acts.__setitem__("x", i[0].detach()))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model_b(batch[:16, :-1])
    hook.remove()
    x16 = acts["x"].reshape(-1, mem.d_in)                  # (16384, d) prefill shape
    x4 = x16[:4096].clone().requires_grad_()               # training micro-batch shape
    mem.train()
    out["mem_layer_train_fwd_stages"] = memory_layer_stages(mem, x4, grad=True)
    out["mem_layer_train_bwd"] = memory_layer_backward(mem, x4)
    mem.eval()
    out["mem_layer_prefill_fwd_stages"] = memory_layer_stages(mem, x16, grad=False)
    out["mem_layer_decode_fwd_stages"] = memory_layer_stages(mem, x16[:1], grad=False, reps=50)
    mem.train()
    out["lazy_adam_alone"] = lazy_adam_alone(mem.values.weight)
    prompt = batch[:1, :128]
    out["inference_B"] = inference(model_b, prompt)
    del model_b
    torch.cuda.empty_cache()
    model_a = load("A")
    out["inference_A"] = inference(model_a, prompt)
    out["ffn_reference"] = ffn_reference(model_a, x4.detach().clone().requires_grad_())
    out["ffn_reference_prefill_fwd"] = ffn_reference(model_a, x16.detach().clone().requires_grad_())["ffn_fwd"]
    blk = model_a.layers[2]
    ffn = blk.ffn
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out["ffn_reference_decode_fwd"] = bench(lambda: ffn(x16[:1]), 50)
    print(json.dumps({k: v for k, v in out.items() if k not in ("train_step_B", "train_step_A")}, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
