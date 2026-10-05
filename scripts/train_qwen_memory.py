"""Step 3: train the add-on (memory table or dense control) on a frozen Qwen3.5-0.8B. Prepared, NOT run before the
criteria in REPORT.md (step 3) are approved.

  .venv-qwen/bin/python scripts/train_qwen_memory.py --kind memory --out_dir runs/qwen/QT-s0 --epochs 2
  .venv-qwen/bin/python scripts/train_qwen_memory.py --kind dense  --out_dir runs/qwen/QD-s0 --epochs 2
  .venv-qwen/bin/python scripts/train_qwen_memory.py --kind none   --out_dir runs/qwen/Q --eval_only

Data: --data_dir from scripts/prepare_qwen_data.py (uint32 token files). Training windows: train_new cut into
non-overlapping windows of seq_len + 1, shuffled per epoch with --data_seed. Every window is seen once per epoch.
Loss: next-token cross-entropy; the logits over the 248k vocabulary are computed in chunks with recomputation in
the backward pass (otherwise 16 x 2048 x 248k logits would need ~30 GB).
Optimisation as B: AdamW (0.9 / 0.95), lr 6e-4 for query / keys / BatchNorm / swilu / gates (weight decay 0.1 on
matrices), table: row-sparse gradients + lazy Adam with value_lr 2.4e-3, no decay; linear warmup 5 %, cosine to
10 %; clip 1.0 (table separately). bf16 autocast; the frozen Qwen weights stay bf16.
Evaluation (every --eval_every_tokens and at the end): token PPL on val_new (decisive), val_known, mem_probe and
the cutoff-curve months (end only).
"""
import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from smlm.optim import build_optimizer, clip_grads, lr_multiplier, set_lr  # noqa: E402
from smlm.qwen_memory import AddOnConfig, addon_state_dict, attach, trainable_parameters  # noqa: E402


def load_tokens(data_dir, name):
    return np.memmap(os.path.join(data_dir, name + ".bin"), dtype=np.uint32, mode="r")


def chunked_ce(hidden, weight, targets, chunk=2048):
    """Mean cross-entropy of hidden @ weight.T against targets, computed in token chunks; each chunk's logits are
    recomputed in the backward pass instead of being stored."""
    h = hidden.reshape(-1, hidden.shape[-1])
    t = targets.reshape(-1)

    def part(hc, tc):
        return F.cross_entropy((hc @ weight.t()).float(), tc, reduction="sum")

    total = 0.0
    for a in range(0, h.shape[0], chunk):
        hc, tc = h[a:a + chunk], t[a:a + chunk]
        total = total + (checkpoint(part, hc, tc, use_reentrant=False) if hc.requires_grad else part(hc, tc))
    return total / t.numel()


CE_CHUNK = 1024


def forward_loss(model, x, y):
    hidden = model.model(input_ids=x).last_hidden_state
    return chunked_ce(hidden, model.lm_head.weight, y, chunk=CE_CHUNK)


def knowledge_token_ids(tok):
    """Token ids that carry facts rather than language: any digit, or a word-initial capital letter
    ("Ġ"/space-prefixed or bare). Sentence-initial capitals are removed later via the previous token."""
    vocab = tok.convert_ids_to_tokens(list(range(len(tok))))
    digit = np.zeros(len(vocab), bool)
    cap = np.zeros(len(vocab), bool)
    end = np.zeros(len(vocab), bool)
    for i, t in enumerate(vocab):
        if t is None:
            continue
        s = tok.convert_tokens_to_string([t])
        digit[i] = any(ch.isdigit() for ch in s)
        st = s.lstrip(" ")
        cap[i] = bool(st) and st[0].isupper() and (s.startswith(" ") or len(st) == len(s))
        end[i] = s.rstrip(" ").endswith((".", "!", "?", ":", "\n")) or s.endswith("\n")
    return digit, cap, end


def knowledge_mask(tokens, ids):
    """mask[j] for target position j (token tokens[j + 1]): digit token, or capitalised word not after a sentence end."""
    digit, cap, end = ids
    t = np.asarray(tokens, dtype=np.int64)
    tgt, prev = t[1:], t[:-1]
    return digit[tgt] | (cap[tgt] & ~end[prev])


@torch.no_grad()
def evaluate(model, tokens, seq_len, max_windows=None, batch=4, mask=None):
    """Token PPL over non-overlapping windows; with `mask` (bool per target position, see knowledge_mask) also the
    PPL over the masked ("knowledge") tokens only."""
    model.eval()
    n = (len(tokens) - 1) // seq_len
    if max_windows:
        n = min(n, max_windows)
    nll, count, knll, kcount = 0.0, 0, 0.0, 0
    for a in range(0, n, batch):
        idx = range(a, min(n, a + batch))
        x = torch.from_numpy(np.stack([tokens[i * seq_len:(i + 1) * seq_len] for i in idx]).astype(np.int64)).cuda()
        y = torch.from_numpy(np.stack([tokens[i * seq_len + 1:(i + 1) * seq_len + 1] for i in idx])
                             .astype(np.int64)).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.model(input_ids=x).last_hidden_state.reshape(-1, model.lm_head.weight.shape[1])
            per = torch.cat([F.cross_entropy((h[c:c + CE_CHUNK] @ model.lm_head.weight.t()).float(),
                                             y.reshape(-1)[c:c + CE_CHUNK], reduction="none")
                             for c in range(0, h.shape[0], CE_CHUNK)])
        nll += float(per.sum())
        count += per.numel()
        if mask is not None:
            m = torch.from_numpy(np.concatenate([mask[i * seq_len:(i + 1) * seq_len] for i in idx])).cuda()
            knll += float(per[m].sum())
            kcount += int(m.sum())
    model.train()
    out = {"loss": nll / count, "ppl": math.exp(nll / count), "tokens": count}
    if mask is not None and kcount:
        out.update({"knowledge_ppl": math.exp(knll / kcount), "knowledge_tokens": kcount})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=os.environ.get("QWEN_DIR", os.path.join(ROOT, "models", "Qwen3.5-0.8B")))
    ap.add_argument("--data_dir", default=os.environ.get("QWEN_DATA", os.path.join(ROOT, "data", "qwen_wiki")))
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--kind", choices=["memory", "dense", "none"], required=True)
    ap.add_argument("--n_keys", type=int, default=1024)
    ap.add_argument("--layers", default="5,11,17")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max_steps", type=int, default=None, help="stop after this many steps (speed probes)")
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--batch_seqs", type=int, default=16)
    ap.add_argument("--micro_bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--value_lr", type=float, default=2.4e-3)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--min_lr_ratio", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_seed", type=int, default=1234)
    ap.add_argument("--eval_every_tokens", type=float, default=20e6)
    ap.add_argument("--eval_windows", type=int, default=None, help="limit evaluation windows (probes)")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--save", type=int, default=1)
    ap.add_argument("--ce_chunk", type=int, default=1024, help="tokens per logits chunk in the loss")
    ap.add_argument("--grad_ckpt", type=int, default=0, help="recompute the frozen Qwen layers in the backward pass")
    args = ap.parse_args()
    global CE_CHUNK
    CE_CHUNK = args.ce_chunk
    os.makedirs(args.out_dir, exist_ok=True)
    from transformers import AutoModelForCausalLM

    torch.manual_seed(args.seed)
    t_start = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.bfloat16).cuda()
    acfg = None
    if args.kind != "none":
        acfg = AddOnConfig(kind=args.kind, n_keys=args.n_keys, layers=[int(i) for i in args.layers.split(",")])
        attach(model, acfg)
    else:
        for p in model.parameters():
            p.requires_grad_(False)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    meta = json.load(open(os.path.join(args.data_dir, "meta.json")))
    evals = {k: load_tokens(args.data_dir, k) for k in ("val_new", "val_known", "val_known_same", "mem_probe")
             if os.path.exists(os.path.join(args.data_dir, k + ".bin"))}
    info = {"status": "running", "args": vars(args), "addon_cfg": acfg.to_dict() if acfg else None,
            "trainable_params": sum(p.numel() for p in trainable_parameters(model)) if acfg else 0,
            "data": {k: {kk: vv for kk, vv in v.items() if kk != "pages"} for k, v in meta["splits"].items()},
            "started": time.strftime("%Y-%m-%d %H:%M:%S"), "gpu": torch.cuda.get_device_name(0)}

    def write_info():
        json.dump(info, open(os.path.join(args.out_dir, "run-info.json"), "w"), indent=1)

    from transformers import AutoTokenizer
    kid = knowledge_token_ids(AutoTokenizer.from_pretrained(args.model_dir))
    masks = {k: knowledge_mask(v, kid) for k, v in evals.items()}

    def eval_all(final=False):
        out = {k: evaluate(model, v, args.seq_len, args.eval_windows, mask=masks[k]) for k, v in evals.items()}
        if final:
            for k in sorted(meta["splits"]):
                if k.startswith("curve_"):
                    out[k] = evaluate(model, load_tokens(args.data_dir, k), args.seq_len, args.eval_windows)
        return out

    write_info()
    if args.eval_only or args.kind == "none":
        info["results"] = eval_all(final=True)
        info["status"] = "done"
        write_info()
        print(json.dumps(info["results"], indent=1))
        return

    train = load_tokens(args.data_dir, "train_new")
    n_win = (len(train) - 1) // args.seq_len
    steps_per_epoch = n_win // args.batch_seqs
    total_steps = int(args.epochs * steps_per_epoch)
    warmup = max(1, int(args.warmup_frac * total_steps))
    accum = args.batch_seqs // args.micro_bs
    tok_per_step = args.batch_seqs * args.seq_len
    eval_every = max(1, int(args.eval_every_tokens // tok_per_step))
    opt = build_optimizer(model.addons, args.lr, args.value_lr, args.weight_decay)
    info.update({"total_steps": total_steps, "tokens_per_step": tok_per_step, "train_windows": n_win})
    rng = np.random.default_rng(args.data_seed)
    order = np.concatenate([rng.permutation(n_win) for _ in range(math.ceil(args.epochs) + 1)])
    mfile = open(os.path.join(args.out_dir, "metrics.csv"), "w", newline="")
    mcsv = None
    lfile = open(os.path.join(args.out_dir, "train_log.csv"), "w", newline="")
    lcsv = csv.writer(lfile)
    lcsv.writerow(["step", "tokens", "loss", "grad_norm", "value_grad_norm", "gates", "tok_s", "peak_vram_gib"])

    def log_eval(step):
        nonlocal mcsv
        ev = eval_all()
        row = {"step": step, "tokens": step * tok_per_step,
               **{f"{k}_ppl": v["ppl"] for k, v in ev.items()},
               **{f"{k}_knowledge_ppl": v.get("knowledge_ppl") for k, v in ev.items()},
               "gates": " ".join(f"{float(a.gate):.4f}" for a in model.addons)}
        if mcsv is None:
            mcsv = csv.DictWriter(mfile, fieldnames=list(row))
            mcsv.writeheader()
        mcsv.writerow(row)
        mfile.flush()
        print(f"[eval] step {step} " + " ".join(f"{k} {v['ppl']:.3f}" for k, v in ev.items()), flush=True)

    log_eval(0)
    model.train()
    t0 = time.perf_counter()
    for step in range(total_steps):
        if args.max_steps and step >= args.max_steps:
            break
        set_lr(opt, lr_multiplier(step, total_steps, warmup, args.min_lr_ratio))
        wins = order[step * args.batch_seqs:(step + 1) * args.batch_seqs]
        batch = torch.from_numpy(np.stack([train[w * args.seq_len:(w + 1) * args.seq_len + 1] for w in wins])
                                 .astype(np.int64)).cuda()
        loss_sum = 0.0
        for i in range(accum):
            mb = batch[i * args.micro_bs:(i + 1) * args.micro_bs]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = forward_loss(model, mb[:, :-1], mb[:, 1:])
            (loss / accum).backward()
            loss_sum += float(loss) / accum
        gn, vgn = clip_grads(model.addons, args.clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if not math.isfinite(loss_sum):
            info["status"] = "diverged"
            write_info()
            raise SystemExit(f"non-finite loss at step {step + 1}")
        if (step + 1) % 10 == 0:
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            t0 = time.perf_counter()
            lcsv.writerow([step + 1, (step + 1) * tok_per_step, loss_sum, float(gn),
                           float(vgn) if vgn is not None else "", " ".join(f"{float(a.gate):.4f}" for a in model.addons),
                           round(10 * tok_per_step / dt), round(torch.cuda.max_memory_allocated() / 2**30, 2)])
            lfile.flush()
        if (step + 1) % eval_every == 0 or step + 1 == total_steps:
            log_eval(step + 1)
    info["results"] = eval_all(final=True)
    info.update({"status": "done", "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "duration_s": round(time.time() - t_start, 1),
                 "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
                 "gates": [float(a.gate) for a in model.addons]})
    if args.save:
        torch.save(addon_state_dict(model), os.path.join(args.out_dir, "addons.pt"))
    write_info()
    print(json.dumps(info["results"], indent=1))


if __name__ == "__main__":
    main()
