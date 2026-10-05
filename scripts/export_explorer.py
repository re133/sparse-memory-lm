"""Build the interactive table explorer (docs/explorer/index.html) from the stage-1b B-1M run. CPU only, ~1 min.

  python scripts/export_explorer.py [--run runs/s1b/B-1M-s0]

Needs from the run: model.pt (value table), mem_index_sample.npz (which entries were read for the first 64 x 1024
validation tokens, written by smlm/train.py at the end of training) and mem_access_train.npy (reads per entry over
the whole training). The page is docs/explorer/template.html with the data embedded as JSON; GitHub Pages serves it.
"""
import argparse
import base64
import codecs
import json
import os
import sys

import numpy as np
import tiktoken
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = os.path.join(ROOT, "docs", "explorer")
SHOWCASE = ["Solar cycle 21", "Housemarque", "Roman Baths, Strand Lane"]
SHOW_TOKENS = 110        # tokens per showcase paragraph
TOP_PER_TOKEN = 3        # entries per token and layer shown in the sentence view
N_TOP = 150              # most-read entries in the sample
N_RANDOM = 80
N_CTX = 6
LEFT, RIGHT = 14, 3
EOT = 50256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.path.join(ROOT, "runs", "s1b", "B-1M-s0"))
    args = ap.parse_args()
    run = args.run
    enc = tiktoken.get_encoding("gpt2")
    info = json.load(open(os.path.join(run, "run-info.json")))
    z = np.load(os.path.join(run, "mem_index_sample.npz"))
    tok = z["tokens"].astype(np.int64)                       # (T,)
    idx = z["indices"].astype(np.int64)                      # (T, L, H, K)
    sc = z["scores"].astype(np.float32)
    layer_ids = [int(x) + 1 for x in z["layer_ids"]]         # 1-based layer numbers for display
    T, L, H, K = idx.shape
    w = np.exp(sc - sc.max(-1, keepdims=True))
    w /= w.sum(-1, keepdims=True)                            # softmax weights as used (no score scale in B-1M)
    train_reads = np.load(os.path.join(run, "mem_access_train.npy"))
    n_entries = train_reads.size
    side = int(round(n_entries ** 0.5))

    # ---- flattened read list: entry, token position, layer, weight
    e_flat = idx.reshape(-1)
    t_flat = np.repeat(np.arange(T), L * H * K)
    l_flat = np.tile(np.repeat(np.arange(L), H * K), T)
    w_flat = w.reshape(-1)
    sample_weight = np.bincount(e_flat, weights=w_flat, minlength=n_entries)
    sample_reads = np.bincount(e_flat, minlength=n_entries)

    # ---- showcase paragraphs
    eot_pos = np.where(tok == EOT)[0]
    starts = [0] + list(eot_pos + 1)
    show = []
    selected = set()
    for title in SHOWCASE:
        s = next(s for s in starts if enc.decode(tok[s:s + 30].tolist()).split("\n")[0] == title)
        e = s + SHOW_TOKENS
        stop = np.where(tok[s:e] == EOT)[0]
        if stop.size:
            e = s + int(stop[0])
        pieces, reads = [], []
        dec = codecs.getincrementaldecoder("utf-8")(errors="replace")    # characters split over tokens stay whole
        for t in range(s, e):
            pieces.append(dec.decode(enc.decode_single_token_bytes(int(tok[t]))).replace("�", ""))
            per_layer = []
            for li in range(L):
                agg = {}
                for h in range(H):
                    for k in range(K):
                        ent = int(idx[t, li, h, k])
                        agg[ent] = agg.get(ent, 0.0) + float(w[t, li, h, k])
                top = sorted(agg.items(), key=lambda x: -x[1])[:TOP_PER_TOKEN]
                per_layer.append([[ent, round(wt, 4)] for ent, wt in top])
                selected.update(ent for ent, _ in top)
            reads.append(per_layer)
        show.append({"title": title, "start": int(s), "pieces": pieces, "reads": reads})

    top_ids = [int(i) for i in np.argsort(-sample_weight)[:N_TOP]]
    selected.update(top_ids)
    rng = np.random.default_rng(0)
    pool = np.where(sample_reads >= 20)[0]
    rand_ids = [int(i) for i in rng.choice(pool, N_RANDOM, replace=False)]
    selected.update(rand_ids)
    sel = np.array(sorted(selected))

    # ---- occurrences of the selected entries
    mask = np.isin(e_flat, sel)
    oe, ot, ol, ow = e_flat[mask], t_flat[mask], l_flat[mask], w_flat[mask]
    order = np.lexsort((-ow, oe))
    oe, ot, ol, ow = oe[order], ot[order], ol[order], ow[order]
    bounds = np.searchsorted(oe, sel), np.searchsorted(oe, sel, side="right")

    def text(tokens):
        # partial UTF-8 at a token boundary decodes to U+FFFD; drop those fragments
        return enc.decode([int(x) for x in tokens]).replace("<|endoftext|>", "").replace("�", "")

    def chip(i):
        return text([i]) or "(part of a byte)"

    def context(t):
        a = t
        while a > 0 and t - a < LEFT and tok[a - 1] != EOT:
            a -= 1
        b = t + 1
        while b < T and b - t - 1 < RIGHT and tok[b] != EOT:
            b += 1
        return text(tok[a:t]), text(tok[t:t + 1]), text(tok[t + 1:b])

    # ---- value vectors of the selected entries
    sd = torch.load(os.path.join(run, "model.pt"), map_location="cpu", mmap=True)["state_dict"]
    table = sd["layers.2.ffn.values.weight"]
    vals = table[torch.from_numpy(sel)].float().numpy()
    all_norm_sample = table[torch.from_numpy(rng.choice(n_entries, 20000, replace=False))].float().norm(dim=-1).numpy()

    entries = {}
    for j, ent in enumerate(sel):
        a, b = bounds[0][j], bounds[1][j]
        seen, ctx = set(), []
        for p in range(a, b):
            t = int(ot[p])
            if t in seen:
                continue
            seen.add(t)
            left, cur, nxt = context(t)
            ctx.append([left, cur, nxt, round(float(ow[p]), 3), layer_ids[int(ol[p])]])
            if len(ctx) == N_CTX:
                break
        cur_tok = np.bincount(tok[ot[a:b]], minlength=50257)
        nxt_pos = np.minimum(ot[a:b] + 1, T - 1)
        nxt_tok = np.bincount(tok[nxt_pos], minlength=50257)
        per_layer = np.bincount(ol[a:b], minlength=L)
        v = vals[j]
        scale = float(np.abs(v).max()) or 1.0
        q = np.clip(np.round(v / scale * 127), -127, 127).astype(np.int8)
        entries[int(ent)] = {
            "r": int(train_reads[ent]), "n": int(sample_reads[ent]), "s": round(float(sample_weight[ent]), 2),
            "L": [int(x) for x in per_layer],
            "ctx": ctx,
            "cur": [[chip(i), int(cur_tok[i])] for i in np.argsort(-cur_tok)[:6] if cur_tok[i] > 0],
            "nxt": [[chip(i), int(nxt_tok[i])] for i in np.argsort(-nxt_tok)[:6] if nxt_tok[i] > 0],
            "v": base64.b64encode(q.tobytes()).decode(), "vs": round(scale, 4),
            "vn": round(float(np.linalg.norm(v)), 3),
        }

    # ---- heat map of training reads on the 1024 x 1024 product-key grid
    logc = np.log10(train_reads.astype(np.float64) + 1)
    lo, hi = float(np.percentile(logc, 0.1)), float(logc.max())
    heat = np.clip(np.round((logc - lo) / (hi - lo) * 255), 0, 255).astype(np.uint8)
    res = info["results"]
    meta = {
        "n_entries": int(n_entries), "side": side, "v_dim": int(table.shape[1]), "layers": layer_ids,
        "heads": H, "knn": K, "train_tokens": info["train_config"]["total_tokens"],
        "val_ppl": res["val_ppl"], "usage": res["mem_val"]["usage"], "top1pct": res["mem_val"]["top1pct_share"],
        "eff": res["mem_val"]["eff_entries_per_head"], "sample_tokens": int(T),
        "sample_articles": int((tok == EOT).sum()), "entries_with_text": len(entries),
        "heat_lo": lo, "heat_hi": hi, "train_reads_median": float(np.median(train_reads)),
        "train_reads_min": int(train_reads.min()), "train_reads_max": int(train_reads.max()),
        "value_norm_median": float(np.median(all_norm_sample)),
        "commit": info["git"]["commit"][:7], "run": os.path.relpath(run, ROOT),
    }
    data = {"meta": meta, "heat": base64.b64encode(heat.tobytes()).decode(), "entries": entries,
            "show": show, "top": top_ids, "rand": rand_ids}
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")   # no "</script>"
    page = open(os.path.join(PAGE, "template.html")).read().replace("__DATA__", blob)
    with open(os.path.join(PAGE, "index.html"), "w") as f:
        f.write(page)
    print(json.dumps(meta, indent=1), file=sys.stderr)
    print(f"{os.path.relpath(os.path.join(PAGE, 'index.html'), ROOT)}: {len(page.encode()) / 1e6:.1f} MB, "
          f"{len(entries)} entries with text", file=sys.stderr)


if __name__ == "__main__":
    main()
