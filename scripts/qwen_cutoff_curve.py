"""Step 3: where does Qwen's knowledge end? Per-article perplexity by creation month (scripts/prepare_qwen_data.py).

  .venv-qwen/bin/python scripts/qwen_cutoff_curve.py --out report/qwen/cutoff_curve.json

Every article (split at <|endoftext|>) is scored on its first --max_len tokens; per month: median article PPL
with a bootstrap 90 % interval (robust against a few long, repetitive list articles that dominate a token-weighted
PPL), plus the token-weighted PPL. Reference lines: val_known (2023 text, HF preparation), val_known_same (old
articles of the same 2026 dump, same preparation as the new ones), val_new.
"""
import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def article_nll(model, tokens, eot, max_len):
    cuts = np.flatnonzero(tokens == eot)
    starts = np.r_[0, cuts[:-1] + 1]
    out = []
    for s, e in zip(starts, cuts):
        t = torch.from_numpy(tokens[s:min(e + 1, s + max_len + 1)].astype(np.int64)).cuda()[None]
        if t.shape[1] < 16:
            continue
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.model(input_ids=t[:, :-1]).last_hidden_state
            logits = (h @ model.lm_head.weight.t()).float()
        nll = F.cross_entropy(logits[0], t[0, 1:], reduction="sum").item()
        out.append((nll, t.shape[1] - 1))
    return out


def summary(arts, rng):
    ppl = np.array([math.exp(n / c) for n, c in arts])
    boots = [np.median(rng.choice(ppl, len(ppl))) for _ in range(1000)]
    tot = sum(n for n, _ in arts) / sum(c for _, c in arts)
    return {"n_articles": len(arts), "median_ppl": float(np.median(ppl)),
            "median_ppl_90ci": [float(np.percentile(boots, 5)), float(np.percentile(boots, 95))],
            "token_weighted_ppl": math.exp(tot)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="/home/leon/smlm-models/Qwen3.5-0.8B")
    ap.add_argument("--data_dir", default="/home/leon/smlm-data/qwen_wiki")
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--max_articles", type=int, default=300, help="for the reference sets")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.bfloat16).cuda().eval()
    meta = json.load(open(os.path.join(args.data_dir, "meta.json")))
    eot = meta["eot_id"]
    rng = np.random.default_rng(0)
    res = {}
    names = sorted(k for k in meta["splits"] if k.startswith("curve_")) + \
        [k for k in ("val_known", "val_known_same", "val_new") if k in meta["splits"]]
    for k in names:
        tok = np.fromfile(os.path.join(args.data_dir, k + ".bin"), dtype=np.uint32)
        arts = article_nll(model, tok, eot, args.max_len)
        if not k.startswith("curve_"):
            arts = arts[:args.max_articles]
        res[k] = summary(arts, rng)
        r = res[k]
        print(f"{k:16s} n={r['n_articles']:4d} median PPL {r['median_ppl']:7.2f} "
              f"[{r['median_ppl_90ci'][0]:.2f}, {r['median_ppl_90ci'][1]:.2f}]  token-weighted {r['token_weighted_ppl']:.2f}",
              flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"max_len": args.max_len, "results": res}, open(args.out, "w"), indent=1)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        months = [k[6:] for k in names if k.startswith("curve_")]
        med = [res["curve_" + m]["median_ppl"] for m in months]
        lo = [res["curve_" + m]["median_ppl_90ci"][0] for m in months]
        hi = [res["curve_" + m]["median_ppl_90ci"][1] for m in months]
        fig, ax = plt.subplots(figsize=(9, 3.8))
        ax.fill_between(range(len(months)), lo, hi, alpha=0.25, label="90 % Intervall")
        ax.plot(range(len(months)), med, "o-", label="Median-PPL je Artikel (150 je Monat)")
        for k, st in (("val_known_same", "--"), ("val_known", ":")):
            if k in res:
                ax.axhline(res[k]["median_ppl"], ls=st, color="gray",
                           label={"val_known_same": "alte Artikel, gleiche Aufbereitung",
                                  "val_known": "Val-Set 2023 (HF-Aufbereitung)"}[k])
        ax.axvline(months.index("2026-03") - 0.5 if "2026-03" in months else 0, color="red", lw=0.8,
                   label="Qwen3.5 veröffentlicht (02.03.2026)")
        ax.set_xticks(range(len(months)), months, rotation=60, fontsize=7)
        ax.set(ylabel="PPL (Qwen3.5-0.8B)", title="Kennt Qwen neue Wikipedia-Artikel? PPL nach Anlege-Monat")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.splitext(args.out)[0] + ".png", dpi=120)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
