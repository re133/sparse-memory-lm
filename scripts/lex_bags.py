"""Step 7: how lexical is the product-key table? Replace the value bag of B-1M by what the current token (or the last
two tokens) predicts on average, and see how much of the table's gain survives. No training.

  python scripts/lex_bags.py fit     -> runs/lex/stats.pt     (bag statistics on training text)
  python scripts/lex_bags.py eval    -> report/lex/eval.json  (val PPL of every variant, run once)

Bag = the weighted sum of the value rows a memory layer reads (before the swilu gate and the output projection), so
the gate and everything after it still see the real context. Variants, applied in all three memory layers at once:
  real     unchanged model (reference)
  zero     bag = 0 (what the model is worth without the table's content)
  tok      bag = mean bag of the current token on training text
  bigram   bag = mean bag of (previous, current) token, hashed into BUCKETS rows; buckets seen < MIN_COUNT times fall
           back to tok
  other    bag = a real bag of the same current token from another (training) context, drawn at random
Kept share of the gain = (PPL_zero - PPL_variant) / (PPL_zero - PPL_real). Model: runs/cloud/B-1M-s0 (fp32 table,
torch lookup path so the bag can be replaced; the Triton inference path fuses bag and gate).
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from smlm.atomic import write_json  # noqa: E402
from smlm.data import load_split  # noqa: E402
from smlm.model import ModelConfig, Transformer  # noqa: E402

RUN = os.path.join(ROOT, "runs", "cloud", "B-1M-s0")
STATS = os.path.join(ROOT, "runs", "lex", "stats.pt")
OUT = os.path.join(ROOT, "report", "lex")
SEQ, BATCH, FIT_WINDOWS, SEED = 1024, 8, 4096, 7           # 4,096 windows = 4.2 M training tokens (~5 min)
VOCAB, BUCKETS, MIN_COUNT, POOL = 50304, 1 << 20, 4, 16
BOOT = 2000


def load_model():
    ck = torch.load(os.path.join(RUN, "model.pt"), map_location="cpu", weights_only=False)
    model = Transformer(ModelConfig(**{**ck["model_config"], "mem_impl": "torch"})).cuda()
    model.load_state_dict(ck["state_dict"])
    return model.eval()


def bigram(prev, cur):
    return (prev * 50021 + cur * 1000003) % BUCKETS


class Hook:
    """Replaces read_values of every memory layer; ctx holds the current and previous token of each position."""

    def __init__(self, model):
        self.mems = model.memory_layers()
        self.mode, self.ctx, self.stats = "real", None, None
        for k, m in enumerate(self.mems):
            orig = m.read_values
            m.read_values = lambda idx, w, k=k, orig=orig: self(k, orig(idx, w))

    def __call__(self, k, bag):
        cur, prev = self.ctx
        if self.mode == "fit":
            self.accumulate(k, bag.float(), cur, prev)
            return bag
        if self.mode == "real":
            return bag
        if self.mode == "zero":
            return torch.zeros_like(bag)
        s = self.stats[k]
        tok_mean = s["tok_sum"][cur] / s["tok_n"][cur].clamp_min(1)[:, None]
        if self.mode == "tok":
            return tok_mean.to(bag.dtype)
        if self.mode == "bigram":
            b = bigram(prev, cur)
            use = s["bi_n"][b] >= MIN_COUNT
            mean = torch.where(use[:, None], s["bi_sum"][b] / s["bi_n"][b].clamp_min(1)[:, None], tok_mean)
            return mean.to(bag.dtype)
        if self.mode == "other":
            have = s["pool_n"][cur].clamp(max=POOL)
            slot = (torch.rand(cur.shape, generator=self.gen, device=cur.device) * have.clamp_min(1)).long()
            pick = s["pool"][cur, slot].float()
            return torch.where((have > 0)[:, None], pick, tok_mean).to(bag.dtype)
        raise ValueError(self.mode)

    def accumulate(self, k, bag, cur, prev):
        s = self.stats[k]
        s["tok_sum"].index_add_(0, cur, bag)
        s["tok_n"].index_add_(0, cur, torch.ones_like(cur, dtype=torch.float32))
        b = bigram(prev, cur)
        s["bi_sum"].index_add_(0, b, bag)
        s["bi_n"].index_add_(0, b, torch.ones_like(b, dtype=torch.float32))
        # pool of real bags per token: the first POOL occurrences (training windows come in random order)
        order = torch.argsort(cur, stable=True)
        c = cur[order]
        first = torch.searchsorted(c, c, right=False)
        slot = s["pool_n"][c] + (torch.arange(len(c), device=c.device) - first)
        keep = slot < POOL
        s["pool"][c[keep], slot[keep]] = bag[order[keep]].to(s["pool"].dtype)
        s["pool_n"].index_add_(0, cur, torch.ones_like(cur))


def windows(split, n=None, seed=None):
    t = load_split(split, "wikipedia")
    nw = (len(t) - 1) // SEQ
    idx = np.arange(nw) if n is None else np.sort(np.random.default_rng(seed).permutation(nw)[:n])
    for a in range(0, len(idx), BATCH):
        w = idx[a:a + BATCH]
        x = np.stack([np.asarray(t[i * SEQ:i * SEQ + SEQ + 1], dtype=np.int64) for i in w])
        prev = np.array([int(t[i * SEQ - 1]) if i > 0 else 50256 for i in w])
        yield torch.from_numpy(x).cuda(), torch.from_numpy(prev).cuda(), w


def set_ctx(hook, x, prev0):
    cur = x[:, :-1]
    prev = torch.cat([prev0[:, None], cur[:, :-1]], 1)
    hook.ctx = (cur.reshape(-1), prev.reshape(-1))


@torch.no_grad()
def fit():
    model = load_model()
    hook = Hook(model)
    d = model.memory_layers()[0].v_dim
    z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device="cuda")  # noqa: E731
    hook.stats = [{"tok_sum": z(VOCAB, d), "tok_n": z(VOCAB), "bi_sum": z(BUCKETS, d), "bi_n": z(BUCKETS),
                   "pool": z(VOCAB, POOL, d, dt=torch.bfloat16), "pool_n": z(VOCAB, dt=torch.long)}
                  for _ in hook.mems]
    hook.mode = "fit"
    t0 = time.time()
    for x, prev0, _ in windows("train", FIT_WINDOWS, SEED):
        set_ctx(hook, x, prev0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(x[:, :-1])
    os.makedirs(os.path.dirname(STATS), exist_ok=True)
    torch.save({"stats": [{k: v.cpu() for k, v in s.items()} for s in hook.stats], "fit_windows": FIT_WINDOWS,
                "seconds": round(time.time() - t0, 1)}, STATS)
    print(f"fit: {FIT_WINDOWS} windows in {time.time() - t0:.0f} s", flush=True)


@torch.no_grad()
def evaluate():
    out_path = os.path.join(OUT, "eval.json")
    assert not os.path.exists(out_path), "the validation set is evaluated once"
    model = load_model()
    hook = Hook(model)
    st = torch.load(STATS, weights_only=False)
    hook.stats = [{k: v.cuda() for k, v in s.items()} for s in st["stats"]]
    res = {"run": RUN, "fit_windows": st["fit_windows"], "buckets": BUCKETS, "min_count": MIN_COUNT, "pool": POOL,
           "variants": {}}
    per_window = {}
    for mode in ("real", "zero", "tok", "bigram", "other"):
        hook.mode = mode
        hook.gen = torch.Generator(device="cuda").manual_seed(0)
        nll, cnt = [], []
        for x, prev0, _ in windows("validation"):
            set_ctx(hook, x, prev0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x[:, :-1])
            l = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1), reduction="none")
            nll += l.view(x.shape[0], -1).sum(1).tolist()
            cnt += [x.shape[1] - 1] * x.shape[0]
        per_window[mode] = np.array(nll)
        n = np.array(cnt)
        res["variants"][mode] = {"val_ppl": round(math.exp(per_window[mode].sum() / n.sum()), 4)}
        print(mode, res["variants"][mode], flush=True)
    rng = np.random.default_rng(0)
    k = rng.integers(0, len(n), (BOOT, len(n)))
    ppl = {m: np.exp(v[k].sum(1) / n[k].sum(1)) for m, v in per_window.items()}
    for mode in ("tok", "bigram", "other"):
        share = (ppl["zero"] - ppl[mode]) / (ppl["zero"] - ppl["real"])
        r = res["variants"]
        r[mode]["kept_share_pct"] = round(100 * (r["zero"]["val_ppl"] - r[mode]["val_ppl"])
                                          / (r["zero"]["val_ppl"] - r["real"]["val_ppl"]), 1)
        r[mode]["kept_share_ci95"] = [round(100 * float(np.percentile(share, q)), 1) for q in (2.5, 97.5)]
    os.makedirs(OUT, exist_ok=True)
    write_json(out_path, res)
    print(json.dumps(res["variants"], indent=1))


if __name__ == "__main__":
    {"fit": fit, "eval": evaluate}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: print(__doc__))()
