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
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    dev = args.device

    ck = torch.load(os.path.join(args.run_dir, "model.pt"), map_location="cpu")
    cfg = ModelConfig(**ck["model_config"])
    model = Transformer(cfg)
    model.load_state_dict(ck["state_dict"])
    model.to(dev).eval()
    mem = model.memory_layers()[0]
    blk = next(b for b in model.layers if b.is_memory)
    T = cfg.max_seq_len
    tok = torch.from_numpy(np.asarray(load_split("validation")[: args.windows * T + 1], dtype=np.int64))
    x = tok[:-1].view(args.windows, T).to(dev)
    y = tok[1:].view(args.windows, T).to(dev)

    # hooks: capture memory input/output and dense FFN outputs
    cap = {"ffn_out_norm": []}

    def mem_hook(mod, inp, out):
        cap["mem_in"] = inp[0].detach()
        cap["mem_out"] = out.detach()

    def ffn_hook(mod, inp, out):
        cap["ffn_out_norm"].append(out.detach().float().norm(dim=-1).mean().item())

    hooks = [mem.register_forward_hook(mem_hook)]
    hooks += [b.ffn.register_forward_hook(ffn_hook) for b in model.layers if not b.is_memory]
    resid = {}
    hooks.append(blk.ffn_norm.register_forward_hook(lambda m, i, o: resid.__setitem__("x", i[0].detach())))

    def nll(mode):
        orig_read = mem.read_values
        if mode == "zero":
            mem.read_values = lambda idx, w: torch.zeros(idx.shape[0], mem.v_dim, dtype=mem.values.weight.dtype, device=idx.device)
        elif mode == "random":
            g = torch.Generator(device="cpu").manual_seed(0)
            mem.read_values = lambda idx, w: orig_read(
                torch.randint(0, mem.size, idx.shape, generator=g).to(idx.device), w)
        total, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, args.windows, 4):
                mem.record = mode == "normal"
                logits = model(x[i:i + 4])
                total += F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y[i:i + 4].reshape(-1), reduction="sum").item()
                n += y[i:i + 4].numel()
                if mode == "normal":
                    cap.setdefault("scores", []).append(mem.last_scores.float().cpu())
                    # the weights actually used in the forward pass (include a learned score scale, v2b)
                    cap.setdefault("weights", []).append(mem.last_weights.float().cpu())
                    cap.setdefault("indices", []).append(mem.last_indices.cpu())
                    cap.setdefault("mem_out_norm", []).append(cap["mem_out"].float().norm(dim=-1).mean().item())
                    cap.setdefault("resid_norm", []).append(resid["x"].float().norm(dim=-1).mean().item())
        mem.read_values = orig_read
        mem.record = False
        return total / n

    res = {"run": args.run_dir, "windows": args.windows, "tokens": args.windows * T}
    for mode in ["normal", "zero", "random"]:
        loss = nll(mode)
        res[f"loss_{mode}"] = loss
        res[f"ppl_{mode}"] = math.exp(loss)
        print(mode, round(loss, 4), round(math.exp(loss), 2), flush=True)

    scores = torch.cat(cap["scores"])                     # (N, heads, knn)
    idx = torch.cat(cap["indices"])
    w = torch.cat(cap["weights"])                         # (N, heads, knn), sums to 1 over knn
    ent = -(w * w.clamp_min(1e-12).log()).sum(-1)          # (N, heads)
    res["softmax_eff_entries_per_head_mean"] = float(ent.exp().mean())
    res["softmax_top1_weight_mean"] = float(w.max(-1).values.mean())
    res["score_spread_top1_minus_topk_mean"] = float((scores[..., 0] - scores[..., -1]).mean())
    res["score_scale_per_head"] = [float(v) for v in mem.score_scale().detach().cpu()]
    flat = idx.reshape(idx.shape[0], -1)
    uniq = torch.tensor([row.unique().numel() for row in flat[:4096]], dtype=torch.float)
    res["unique_entries_per_token_of_128"] = float(uniq.mean())
    res["mem_out_norm_mean"] = float(np.mean(cap["mem_out_norm"]))
    res["resid_norm_at_mem_mean"] = float(np.mean(cap["resid_norm"]))
    ffn = np.array(cap["ffn_out_norm"]).reshape(-1, cfg.n_layers - 1).mean(0)
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
