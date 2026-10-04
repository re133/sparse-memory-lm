"""Step 3, on the GPU with the real Qwen3.5-0.8B (setup gate before any paid training):
  1. gate 0 -> logits bit-identical to plain Qwen, eval and train mode, memory (Triton kernels) and dense add-on
  2. 3 optimiser steps run, only the add-ons change, the gates move away from 0
Exit code 0 only if everything holds.
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from smlm.optim import build_optimizer  # noqa: E402
from smlm.qwen_memory import AddOnConfig, attach  # noqa: E402
from train_qwen_memory import forward_loss  # noqa: E402


def main(model_dir=os.environ.get("QWEN_DIR", "/workspace/models/Qwen3.5-0.8B"),
         data_dir=os.environ.get("QWEN_DATA", os.path.join(ROOT, "data", "qwen_wiki"))):
    from transformers import AutoModelForCausalLM
    tok = np.fromfile(os.path.join(data_dir, "val_new.bin"), dtype=np.uint32)[:2 * 513].astype(np.int64)
    x = torch.from_numpy(tok).view(2, 513).cuda()
    ok = True
    for kind in ("memory", "dense"):
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model.eval()
            want_eval = model(x[:, :-1]).logits
            noise = (model(x[:, :-1]).logits - want_eval).abs().max().item()   # run-to-run noise of the base model
            model.train()
            want_train = model(x[:, :-1]).logits
        print(f"{kind}: plain Qwen run-to-run max |diff| = {noise:.3g}" +
              (" (deterministic)" if noise == 0 else " (non-deterministic kernels: compared within this noise)"),
              flush=True)
        frozen = {n: p.detach().clone() for n, p in model.named_parameters()}
        attach(model, AddOnConfig(kind=kind, n_keys=256))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model.eval()
            de = (model(x[:, :-1]).logits - want_eval).abs().max().item()
            model.train()
            dt = (model(x[:, :-1]).logits - want_train).abs().max().item()
        e, t = de <= noise, dt <= max(noise, 0.0)
        print(f"{kind}: gate 0 vs plain max |diff| eval={de:.3g} train={dt:.3g} -> "
              f"{'bit-identical' if de == dt == 0 else 'within run-to-run noise' if e and t else 'DIFFERENT'}", flush=True)
        ok &= e and t
        opt = build_optimizer(model.addons, 6e-4, 2.4e-3, 0.1)
        for _ in range(3):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = forward_loss(model, x[:, :-1], x[:, 1:])
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        gates = [float(a.gate) for a in model.addons]
        same = all(torch.equal(p, frozen[n]) for n, p in model.named_parameters() if not n.startswith("addons."))
        print(f"{kind}: 3 steps loss {float(loss):.4f}, gates {gates}, frozen weights unchanged={same}", flush=True)
        ok &= same and all(g != 0 for g in gates) and bool(torch.isfinite(loss))
        del model, opt
        torch.cuda.empty_cache()
    print("CHECK OK" if ok else "CHECK FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
