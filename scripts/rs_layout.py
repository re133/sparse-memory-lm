"""Row layout of the B-16M table on the SSD: how many 4 KiB pages does a session read with today's layout, and with
a layout learned from which rows are read together? Offline simulation on recorded lookups
(scripts/record_lookups.py), no SSD and no GPU needed. Criteria: REPORT.md, "Step 5".

  python scripts/rs_layout.py fit       -> runs/rs/layouts.npz   (co-reads of 4096 training sessions)
  python scripts/rs_layout.py select    -> report/rs/select.json (512 other training sessions pick the layout)
  python scripts/rs_layout.py eval      -> report/rs/eval.json   (1000 validation sessions, run once)

Row r = i * 4096 + j (i, j = indices into the two sub-key sets) lives at byte pos(r) * 192 of the 4-bit table file;
a row that crosses a page boundary costs both pages. Layouts (all of them can be built exactly by permuting the
sub-keys of every head in the same way together with the table rows, so the model does not change):
  A0  today: pos = i * 4096 + j
  AJ  i outside, j axis reordered: pos = i * 4096 + rank_j[j]
  AI  j outside, i axis reordered: pos = j * 4096 + rank_i[i]
Simulation as in scripts/bench_token_latency.py: the hottest 30 % of the rows (hot_rows.npy) sit in RAM, every other
row comes from the SSD, a page once read stays in the page cache until the end of the session, every session starts
cold. Counted: distinct pages per session, split into the prompt (first 128 tokens, one pass) and the continuation.
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs", "rs")
OUT = os.path.join(ROOT, "report", "rs")
TABLES = os.environ.get("SMLM_TABLES", os.path.join(ROOT, "data", "tables", "B-16M"))
N, ROW_BYTES, PAGE, PROMPT = 4096, 192, 4096, 128
N_FIT, N_SELECT = 4096, 512
CACHE_FRAC = 0.3
SPLIT_SEED = 5


def hot_mask(frac):
    hot = np.zeros(N * N, dtype=bool)
    if frac > 0:
        hot[np.load(os.path.join(TABLES, "hot_rows.npy"), mmap_mode="r")[:int(frac * N * N)]] = True
    return hot


def train_split():
    """Fixed random split of the training sessions into fit and select."""
    perm = np.random.default_rng(SPLIT_SEED).permutation(N_FIT + N_SELECT)
    return np.sort(perm[:N_FIT]), np.sort(perm[N_FIT:])


def positions(layout, rank):
    r = np.arange(N * N, dtype=np.int64)
    i, j = r // N, r % N
    if layout == "A0":
        return r
    if layout == "AJ":
        return i * N + rank[j]
    if layout == "AI":
        return j * N + rank[i]
    raise ValueError(layout)


def session_pages(rows, pos, hot):
    """rows: (T, layers, k) of one session -> (distinct pages in the prompt, new pages per continuation token)."""
    T = rows.shape[0]
    tok = np.repeat(np.arange(T), rows[0].size)
    r = rows.reshape(-1)
    keep = ~hot[r]
    p = pos[r[keep]] * ROW_BYTES
    tok = tok[keep]
    pages = np.concatenate([p // PAGE, (p + ROW_BYTES - 1) // PAGE])
    tok = np.concatenate([tok, tok])
    order = np.argsort(tok, kind="stable")
    _, first = np.unique(pages[order], return_index=True)       # first read of every page, in token order
    first_tok = tok[order][first]
    return int((first_tok < PROMPT).sum()), np.bincount(first_tok[first_tok >= PROMPT] - PROMPT,
                                                        minlength=T - PROMPT)


def simulate(trace, idx, pos, hot):
    res = [session_pages(trace[s], pos, hot) for s in idx]
    prompt = np.array([a for a, _ in res])
    cont = np.stack([b for _, b in res])
    return {"total": prompt + cont.sum(1), "prompt": prompt, "per_token": cont.mean(1)}


def co_reads(trace, idx, hot, outer):
    """C[a, b] = number of sessions in which two non-RAM rows with the same outer index and inner indices a, b were
    both read (outer = i for AJ, j for AI)."""
    acc = np.zeros(N * N, dtype=np.int64)
    keys = []
    for n, s in enumerate(idx):
        r = np.unique(trace[s].reshape(-1))
        r = r[~hot[r]]
        o, inner = (r // N, r % N) if outer == "i" else (r % N, r // N)
        order = np.lexsort((inner, o))
        o, inner = o[order], inner[order]
        start = np.flatnonzero(np.r_[True, o[1:] != o[:-1]])
        size = np.diff(np.r_[start, len(o)])
        g_size = np.repeat(size, size)
        g_start = np.repeat(start, size)
        a = np.repeat(np.arange(len(o)), g_size)
        b = np.repeat(g_start, g_size) + (np.arange(len(a)) - np.repeat(np.cumsum(g_size) - g_size, g_size))
        m = a != b
        keys.append(inner[a[m]] * N + inner[b[m]])
        if sum(len(k) for k in keys) > 2e8 or n == len(idx) - 1:
            acc += np.bincount(np.concatenate(keys), minlength=N * N)
            keys = []
    return acc.reshape(N, N).astype(np.float64)


def order_spectral(C):
    d = C.sum(1) + 1e-9
    L = np.eye(N) - C / np.sqrt(np.outer(d, d))
    _, vec = np.linalg.eigh(L)
    return np.argsort(vec[:, 1] / np.sqrt(d))


def order_greedy(C, window=21):
    """Start with the item with the most co-reads, then always append the item with the most co-reads with the
    last `window` placed items (one page holds 21.3 rows)."""
    placed = [int(C.sum(1).argmax())]
    free = np.ones(N, dtype=bool)
    free[placed[0]] = False
    score = C[:, placed[0]].copy()
    for _ in range(N - 1):
        s = np.where(free, score, -1.0)
        nxt = int(s.argmax()) if s.max() > 0 else int(np.flatnonzero(free)[0])
        placed.append(nxt)
        free[nxt] = False
        score += C[:, nxt]
        if len(placed) > window:
            score -= C[:, placed[-window - 1]]
    return np.array(placed)


def rank_of(order):
    rank = np.empty(N, dtype=np.int64)
    rank[order] = np.arange(N)
    return rank


def boot_ci(new, base, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    k = rng.integers(0, len(new), (n, len(new)))
    rel = new[k].sum(1) / base[k].sum(1) - 1
    return [round(100 * float(np.percentile(rel, q)), 1) for q in (2.5, 97.5)]


def summary(sim, base):
    rel = sim["total"].sum() / base["total"].sum() - 1
    return {"pages_per_session": round(float(sim["total"].mean()), 1),
            "prompt_pages": round(float(sim["prompt"].mean()), 1),
            "pages_per_continuation_token": round(float(sim["per_token"].mean()), 2),
            "change_vs_A0_pct": round(100 * float(rel), 1),
            "ci95_pct": boot_ci(sim["total"], base["total"])}


def main(stage):
    os.makedirs(OUT, exist_ok=True)
    hot = hot_mask(CACHE_FRAC)
    if stage == "fit":
        trace = np.load(os.path.join(RUNS, "train.npy"), mmap_mode="r")
        fit, _ = train_split()
        out = {}
        for layout, outer in (("AJ", "i"), ("AI", "j")):
            t0 = time.time()
            C = co_reads(trace, fit, hot, outer)
            out[f"{layout}_spectral"] = order_spectral(C)
            out[f"{layout}_greedy"] = order_greedy(C)
            print(f"{layout}: {C.sum():.3g} co-reads, {time.time() - t0:.0f} s", flush=True)
        np.savez(os.path.join(RUNS, "layouts.npz"), **out)
        return
    orders = np.load(os.path.join(RUNS, "layouts.npz"))
    if stage == "select":
        trace = np.load(os.path.join(RUNS, "train.npy"), mmap_mode="r")
        _, idx = train_split()
        base = simulate(trace, idx, positions("A0", None), hot)
        res = {"sessions": len(idx), "cache_frac": CACHE_FRAC, "A0": summary(base, base)}
        for name in orders.files:
            res[name] = summary(simulate(trace, idx, positions(name[:2], rank_of(orders[name])), hot), base)
            print(name, res[name], flush=True)
        res["chosen"] = min(orders.files, key=lambda k: res[k]["pages_per_session"])
        json.dump(res, open(os.path.join(OUT, "select.json"), "w"), indent=1)
        print("chosen:", res["chosen"])
        return
    if stage == "eval":
        out_path = os.path.join(OUT, "eval.json")
        assert not os.path.exists(out_path), "the validation sessions are evaluated once"
        chosen = json.load(open(os.path.join(OUT, "select.json")))["chosen"]
        trace = np.load(os.path.join(RUNS, "val.npy"), mmap_mode="r")
        idx = np.arange(trace.shape[0])
        res = {"sessions": len(idx), "chosen": chosen}
        for frac in (CACHE_FRAC, 0.0):
            h = hot if frac == CACHE_FRAC else hot_mask(0.0)
            base = simulate(trace, idx, positions("A0", None), h)
            new = simulate(trace, idx, positions(chosen[:2], rank_of(orders[chosen])), h)
            res[f"cache{frac:g}"] = {"A0": summary(base, base), chosen: summary(new, base)}
            print(frac, res[f"cache{frac:g}"], flush=True)
        json.dump(res, open(out_path, "w"), indent=1)
        return
    raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
