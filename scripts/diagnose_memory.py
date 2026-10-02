"""Health / usefulness diagnostics for a trained memory model (B). Runs on CPU or GPU.

  python scripts/diagnose_memory.py runs/ep1/B-s0 [--windows 32] [--device cpu] [--threads 4]

Reports (written to <run>/<--out, default diagnostics.json>):
  * val NLL on the first --windows validation windows under three conditions:
      normal | memory output zeroed | memory reads uniformly random entries (same softmax weights)
    -> how much the model relies on the memory, and on *which* entries it reads
  * softmax weights over the k selected entries (as used in the forward pass, i.e. after a learned
    score scale): entropy -> effective number of entries per head
  * how often the 4 heads pick the same entry for a token
  * size of the memory output relative to the residual stream and to the dense FFN outputs
  * value-row norms vs. how often a row was read in training (init norm ~ 1.0)
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smlm.data import load_split  # noqa: E402
from smlm.model import ModelConfig, Transformer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--windows", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="diagnostics.json", help="file name inside run_dir")
    ap.add_argument("--data", default=None, help="validation dataset (smlm.data.DATASETS); default = stage 1")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    dev = args.device

    ck = torch.load(os.path.join(args.run_dir, "model.pt"), map_location="cpu")
    cfg = ModelConfig(**ck["model_config"])
    model = Transformer(cfg)
    model.load_state_dict(ck["state_dict"])
    model.to(dev).eval()
    mems = model.memory_layers()                          # ablations act on all memory layers at once
    mem = mems[0]
    blk = next(b for b in model.layers if b.is_memory)
    T = cfg.max_seq_len
    tok = torch.from_numpy(np.asarray(load_split("validation", args.data)[: args.windows * T + 1], dtype=np.int64))
    x = tok[:-1].view(args.windows, T).to(dev)
    y = tok[1:].view(args.windows, T).to(dev)

    # hooks: capture memory input/output and dense FFN outputs
    cap = {"ffn_out_norm": []}

    def mem_hook(mod, inp, out):
        cap.setdefault("mem_out_cur", []).append(out.detach().float().norm(dim=-1).mean().item())

    def ffn_hook(mod, inp, out):
        cap["ffn_out_norm"].append(out.detach().float().norm(dim=-1).mean().item())

    hooks = [m.register_forward_hook(mem_hook) for m in mems]
    hooks += [b.ffn.register_forward_hook(ffn_hook) for b in model.layers if not b.is_memory]
    resid = {}
    hooks.append(blk.ffn_norm.register_forward_hook(lambda m, i, o: resid.__setitem__("x", i[0].detach())))

    def nll(mode):
        g = torch.Generator(device="cpu").manual_seed(0)
        for m in mems:
            orig = m.read_values
            if mode == "zero":
                m.read_values = (lambda m: lambda idx, w: torch.zeros(
                    idx.shape[0], m.v_dim, dtype=m.values.weight.dtype, device=idx.device))(m)
            elif mode == "random":
                m.read_values = (lambda m, orig: lambda idx, w: orig(
                    torch.randint(0, m.size, idx.shape, generator=g).to(idx.device), w))(m, orig)
        total, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, args.windows, 4):
                for m in mems:
                    m.record = mode == "normal"
                cap["mem_out_cur"] = []
                logits = model(x[i:i + 4])
                total += F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y[i:i + 4].reshape(-1), reduction="sum").item()
                n += y[i:i + 4].numel()
                if mode == "normal":
                    for li, m in enumerate(mems):
                        c = cap.setdefault(li, {})
                        c.setdefault("scores", []).append(m.last_scores.float().cpu())
                        # the weights actually used in the forward pass (include a learned score scale, v2b)
                        c.setdefault("weights", []).append(m.last_weights.float().cpu())
                        c.setdefault("indices", []).append(m.last_indices.cpu())
                    cap.setdefault("mem_out_norm", []).append(cap["mem_out_cur"])
                    cap.setdefault("resid_norm", []).append(resid["x"].float().norm(dim=-1).mean().item())
        for m in mems:
            if "read_values" in m.__dict__:
                del m.read_values                          # back to the class method
            m.record = False
        return total / n

    res = {"run": args.run_dir, "windows": args.windows, "tokens": args.windows * T}
    for mode in ["normal", "zero", "random"]:
        loss = nll(mode)
        res[f"loss_{mode}"] = loss
        res[f"ppl_{mode}"] = math.exp(loss)
        print(mode, round(loss, 4), round(math.exp(loss), 2), flush=True)

    def sharpness(li_list):
        scores = torch.cat([torch.cat(cap[li]["scores"]) for li in li_list])     # (N*, heads, knn)
        w = torch.cat([torch.cat(cap[li]["weights"]) for li in li_list])         # sums to 1 over knn
        ent = -(w * w.clamp_min(1e-12).log()).sum(-1)
        return {"eff": float(ent.exp().mean()), "top1": float(w.max(-1).values.mean()),
                "spread": float((scores[..., 0] - scores[..., -1]).mean())}

    allsh = sharpness(range(len(mems)))                    # pooled over all memory layers
    res["softmax_eff_entries_per_head_mean"] = allsh["eff"]
    res["softmax_top1_weight_mean"] = allsh["top1"]
    res["score_spread_top1_minus_topk_mean"] = allsh["spread"]
    res["score_scale_per_head"] = [float(v) for v in mem.score_scale().detach().cpu()]
    if len(mems) > 1:
        res["memory_layers"] = cfg.mem_layers
        res["per_layer"] = [{**sharpness([li]), "score_scale": [float(v) for v in m.score_scale().detach().cpu()]}
                            for li, m in enumerate(mems)]
    idx = torch.cat(cap[0]["indices"])
    flat = idx.reshape(idx.shape[0], -1)
    uniq = torch.tensor([row.unique().numel() for row in flat[:4096]], dtype=torch.float)
    res["unique_entries_per_token_of_128"] = float(uniq.mean())            # first memory layer
    mo = np.array(cap["mem_out_norm"])                                     # (batches, memory layers)
    res["mem_out_norm_mean"] = float(mo.mean())
    if len(mems) > 1:
        res["mem_out_norm_per_layer"] = [round(float(v), 4) for v in mo.mean(0)]
    res["resid_norm_at_mem_mean"] = float(np.mean(cap["resid_norm"]))       # input of the first memory layer
    ffn = np.array(cap["ffn_out_norm"]).reshape(-1, cfg.n_layers - len(mems)).mean(0)
    res["dense_ffn_out_norm_per_layer"] = [round(float(v), 4) for v in ffn]

    vals = mem.values.weight.detach().float().cpu()
    norms = vals.norm(dim=-1).numpy()
    reads = np.load(os.path.join(args.run_dir, "mem_access_train.npy"))
    q = np.quantile(reads, [0.1, 0.5, 0.9, 0.99])
    bins = {"never": reads == 0, "<=p10": (reads > 0) & (reads <= q[0]), "p10-p50": (reads > q[0]) & (reads <= q[1]),
            "p50-p90": (reads > q[1]) & (reads <= q[2]), "p90-p99": (reads > q[2]) & (reads <= q[3]), ">p99": reads > q[3]}
    res["value_norm_by_train_reads"] = {k: {"n": int(m.sum()), "mean_norm": float(norms[m].mean()) if m.any() else None,
                                            "mean_reads": float(reads[m].mean()) if m.any() else None}
                                        for k, m in bins.items()}
    res["value_norm_init_expected"] = 1.0
    res["train_reads_quantiles_p10_p50_p90_p99"] = [float(v) for v in q]
    with open(os.path.join(args.run_dir, args.out), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
