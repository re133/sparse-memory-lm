"""Validation PPL of a trained memory model with the PyTorch reference and with the Triton kernels
(inference table fp32 / bf16 / 4 bit). Checkpoint is only read.

  python scripts/eval_kernels.py --run runs/hampter/B-1M-sparse-s0 --out report/eval_kernels.json
"""
import argparse
import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from smlm.data import load_meta  # noqa: E402
from smlm.model import ModelConfig, Transformer  # noqa: E402
from smlm.train import evaluate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/hampter/B-1M-sparse-s0")
    ap.add_argument("--data", default="wikipedia")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ck = torch.load(os.path.join(ROOT, args.run, "model.pt"), map_location="cuda", weights_only=False)
    words = load_meta(args.data)["splits"]["validation"]["n_words"]
    res = {"run": args.run}
    for impl, kind in [("torch", "fp32"), ("triton", "fp32"), ("triton", "bf16"), ("triton", "q4")]:
        cfg = ModelConfig(**{**ck["model_config"], "mem_impl": impl})
        model = Transformer(cfg).cuda()
        model.load_state_dict(ck["state_dict"])
        if impl == "triton":
            model.set_memory_inference_table(kind)
        torch.cuda.synchronize()
        t0 = time.time()
        ev = evaluate(model, "validation", 1024, words, batch=4, dataset=args.data)
        torch.cuda.synchronize()
        res[f"{impl}_{kind}"] = {"val_ppl": ev["ppl"], "val_loss": ev["loss"], "seconds": time.time() - t0}
        print(impl, kind, res[f"{impl}_{kind}"], flush=True)
        del model
        torch.cuda.empty_cache()
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
