"""Does B-16M compute exactly the same with its 4-bit table in VRAM, in RAM and on the NVMe? Logits compared bit for
bit, not just the perplexity.

  python scripts/check_offload_identical.py [--tables data/tables/B-16M] [--out report/offload/identical_check.json]

Three prompts (Wikipedia validation text if available, else fixed titles): prompt pass + 64 greedy decoding steps
each; all logits of ram and nvme against vram (decode graphs on), plus the generated token ids.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from demo_generate import load  # noqa: E402


@torch.no_grad()
def greedy(model, ids, steps):
    caches = [dict() for _ in range(model.cfg.n_layers)]
    logits_all, toks = [], []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(torch.tensor([ids], device="cuda"), kv_caches=caches, pos0=0)
        logits_all.append(logits[0].float().cpu())
        for i in range(steps):
            nxt = int(logits[0, -1].argmax())
            toks.append(nxt)
            logits = model(torch.tensor([[nxt]], device="cuda"), kv_caches=caches, pos0=len(ids) + i)
            logits_all.append(logits[0].float().cpu())
    return torch.cat(logits_all), toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default=os.environ.get("SMLM_TABLES", os.path.join(ROOT, "data", "tables", "B-16M")))
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--out", default=os.path.join(ROOT, "report", "offload", "identical_check.json"))
    args = ap.parse_args()
    try:
        from smlm.data import load_split
        val = np.asarray(load_split("validation", "wikipedia"), dtype=np.int64)
        prompts = [val[s:s + 128].tolist() for s in (200_000, 400_000, 600_000)]
        source = "wikipedia validation tokens 200000 / 400000 / 600000, 128 each"
    except Exception:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        prompts = [enc.encode_ordinary(t) for t in ("Isaac Newton\n\n", "Volcano\n\n", "Hamburg\n\n")]
        source = "titles Isaac Newton / Volcano / Hamburg"
    runs = {}
    for kind in ("vram", "ram", "nvme"):
        model = load(args.tables, kind, 0.3)
        runs[kind] = [greedy(model, p, args.steps) for p in prompts]
        del model
        torch.cuda.empty_cache()
    res = {"prompts": source, "decode_steps": args.steps, "gpu": torch.cuda.get_device_name(0),
           "torch": torch.__version__, "reference": "vram (4-bit table on the GPU, decode graphs on)"}
    for kind in ("ram", "nvme"):
        same_logits = all(torch.equal(a[0], b[0]) for a, b in zip(runs["vram"], runs[kind]))
        res[kind] = {"logits_bit_identical": same_logits,
                     "max_abs_logit_diff": max(float((a[0] - b[0]).abs().max()) for a, b in zip(runs["vram"], runs[kind])),
                     "tokens_identical": all(a[1] == b[1] for a, b in zip(runs["vram"], runs[kind]))}
    res["tokens_vram"] = [r[1] for r in runs["vram"]]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("ram", "nvme")}, indent=1))


if __name__ == "__main__":
    main()
