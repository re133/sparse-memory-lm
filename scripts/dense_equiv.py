"""Step 1 evaluation (rules pre-registered in REPORT.md, "Schritt 1"): equivalent dense size of B-1M / B-4M / B-16M.

  python scripts/dense_equiv.py            -> report/dense_equiv.json, report/dense_equiv.png, markdown table on stdout

Dense points: A (runs/s1b/A-s0, seed 0) and D-50M ... D-400M (runs/cloud_dense/D-*-s0), Wikipedia val PPL against
non-embedding parameters N. For each B (runs/cloud/B-*-s0):
  * main value: piecewise linear interpolation of log PPL over log N between the two neighbouring dense points
  * comparison: fit PPL = E + a * N^-alpha over all dense points (grid over alpha and E, a by least squares in PPL
    for each grid point, the point with the smallest squared error in log PPL wins)
  * range: PPL of B and of both neighbours shifted by +-0.4 % (seed spread of stage 1c), extreme cases; widened to the
    fit value if that lies outside
  * bracketing: better than the largest dense model -> "> N_max" (fit extrapolation only as a marked hint);
    worse than A -> "< N_A"
Compute per token: forward MACs (run-info macs_per_token, context 1024); for B also the table values read per token.
"""
import json
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED_SPREAD = 0.004
DENSE = [("A", os.path.join("runs", "s1b", "A-s0"))] + \
    [(f"D-{s}", os.path.join("runs", "cloud_dense", f"D-{s}-s0")) for s in ("50M", "100M", "200M", "400M")]
BS = [(f"B-{s}", os.path.join("runs", "cloud", f"B-{s}-s0")) for s in ("1M", "4M", "16M")]


def load(rel):
    p = os.path.join(ROOT, rel, "run-info.json")
    if not os.path.exists(p):
        return None
    i = json.load(open(p))
    if i.get("status") != "done":
        return None
    return {"ppl": i["results"]["val_ppl"], "ppl2": i["results"].get("val2_ppl"),
            "n": i["params"]["non_embedding"], "active": i["params"].get("active_non_embedding_per_token"),
            "macs": i["macs_per_token"]["total"], "table": i["params"].get("memory_values", 0),
            "tok_s": i["results"].get("train_tok_s_median")}


def interp(ppl_b, pts):
    """pts: sorted by n, ppl decreasing. Returns (N_eq, status) by log-log interpolation."""
    lp = math.log(ppl_b)
    if ppl_b < pts[-1][1]:
        return None, f"> {pts[-1][0] / 1e6:.0f} M"
    if ppl_b > pts[0][1]:
        return None, f"< {pts[0][0] / 1e6:.0f} M"
    for (n0, p0), (n1, p1) in zip(pts, pts[1:]):
        if p1 <= ppl_b <= p0:
            t = (math.log(p0) - lp) / (math.log(p0) - math.log(p1))
            return math.exp(math.log(n0) + t * (math.log(n1) - math.log(n0))), "interpolated"
    return None, "not monotone"


def fit(pts):
    """PPL = E + a N^-alpha: grid over alpha and E, a by linear least squares in PPL, best grid point by the squared
    error in log PPL (not an exact least-squares fit in log PPL; that one moves the fitted sizes by at most 0.3M)."""
    n = np.array([p[0] for p in pts], float)
    y = np.array([p[1] for p in pts], float)
    best = None
    for alpha in np.linspace(0.05, 1.5, 291):
        x = n ** -alpha
        for E in np.linspace(0, y.min() * 0.999, 200):
            a = max(1e-12, float(np.dot(x, y - E) / np.dot(x, x)))
            err = float(np.sum((np.log(E + a * x) - np.log(y)) ** 2))
            if best is None or err < best[0]:
                best = (err, E, a, alpha)
    _, E, a, alpha = best
    return {"E": E, "a": a, "alpha": alpha, "rmse_log": math.sqrt(best[0] / len(pts))}


def fit_inverse(ppl, f):
    if ppl <= f["E"]:
        return float("inf")
    return (f["a"] / (ppl - f["E"])) ** (1 / f["alpha"])


def main():
    dense = [(name, load(rel)) for name, rel in DENSE]
    dense = [(name, d) for name, d in dense if d]
    pts = sorted((d["n"], d["ppl"]) for _, d in dense)
    f = fit(pts) if len(pts) >= 3 else None
    out = {"dense": {k: v for k, v in dense}, "fit": f, "seed_spread": SEED_SPREAD, "b": {}}
    for name, rel in BS:
        b = load(rel)
        if not b:
            continue
        neq, status = interp(b["ppl"], pts)
        cands = []
        for sb in (-1, 1):
            for sn in (-1, 1):
                shifted = [(n, p * (1 + sn * SEED_SPREAD)) for n, p in pts]
                v, _ = interp(b["ppl"] * (1 + sb * SEED_SPREAD), shifted)
                cands.append(v)
        lo = min([c for c in cands if c] or [None]) if any(cands) else None
        hi = max([c for c in cands if c] or [None]) if any(cands) else None
        nfit = fit_inverse(b["ppl"], f) if f else None
        if neq and nfit and math.isfinite(nfit):
            lo, hi = min(lo, nfit), max(hi, nfit)
        out["b"][name] = {**b, "n_eq": neq, "status": status, "range": [lo, hi], "n_eq_fit": nfit,
                          "table_values_read_per_token": 3 * 4 * 32 * 384,
                          "range_note": "unbounded (outside the measured dense range)" if None in cands else ""}
    json.dump(out, open(os.path.join(ROOT, "report", "dense_equiv.json"), "w"), indent=1)
    print("| Modell | Val-PPL | N ohne Emb. | MACs/Token vorwärts |")
    print("|---|---|---|---|")
    for name, d in dense:
        print(f"| {name} | {d['ppl']:.3f} | {d['n'] / 1e6:.1f} M | {d['macs'] / 1e6:.1f} M |")
    if f:
        print(f"\nFit: PPL = {f['E']:.2f} + {f['a']:.3g} * N^-{f['alpha']:.3f} (rmse log {f['rmse_log']:.4f})")
    print("\n| B | Val-PPL | gleichwertige dichte Größe | Bereich (±0,4 %) | Fit | MACs/Token | aktiv / Tabelle |")
    print("|---|---|---|---|---|---|---|")
    for name, b in out["b"].items():
        neq = f"{b['n_eq'] / 1e6:.0f} M" if b["n_eq"] else b["status"]
        rng = (f"{b['range'][0] / 1e6:.0f}–{b['range'][1] / 1e6:.0f} M" if b["range"][0] and b["range"][1] else "–")
        nf = f"{b['n_eq_fit'] / 1e6:.0f} M" if b["n_eq_fit"] and math.isfinite(b["n_eq_fit"]) else "–"
        print(f"| {name} | {b['ppl']:.3f} | {neq} | {rng} | {nf} | {b['macs'] / 1e6:.1f} M | "
              f"{b['active'] / 1e6:.1f} M / {b['table'] / 1e9:.2f} Mrd. |")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    ns = [d["n"] for _, d in dense]
    ax[0].plot(ns, [d["ppl"] for _, d in dense], "o-", color="C0", label="dicht (A, D-50M … D-400M)")
    for (name, d) in dense:
        ax[0].annotate(name, (d["n"], d["ppl"]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    if f:
        xs = np.geomspace(min(ns) / 1.5, max(ns) * 4, 100)
        ax[0].plot(xs, f["E"] + f["a"] * xs ** -f["alpha"], ":", color="C0", alpha=0.6, label="Fit E + a·N^-α")
    for i, (name, b) in enumerate(out["b"].items()):
        ax[0].axhline(b["ppl"], color=f"C{i + 1}", ls="--", lw=1, label=f"{name} ({b['ppl']:.2f})")
        if b["n_eq"]:
            ax[0].plot([b["n_eq"]], [b["ppl"]], "s", color=f"C{i + 1}")
    ax[0].set(xscale="log", yscale="log", xlabel="Parameter ohne Embeddings (dicht)", ylabel="Val-PPL Wikipedia",
              title="Gegenwert der Tabelle (500 M Tokens)")
    ax[0].legend(fontsize=7)
    ax[1].plot([d["macs"] for _, d in dense], [d["ppl"] for _, d in dense], "o-", color="C0", label="dicht")
    for i, (name, b) in enumerate(out["b"].items()):
        ax[1].plot([b["macs"]], [b["ppl"]], "s", color=f"C{i + 1}", label=name)
    ax[1].set(xscale="log", yscale="log", xlabel="MACs pro Token (vorwärts)", ylabel="Val-PPL Wikipedia",
              title="Qualität gegen Rechenaufwand pro Token")
    ax[1].legend(fontsize=7)
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
    for a in ax:
        a.yaxis.set_major_locator(FixedLocator([16, 18, 20, 22, 24, 26, 28]))
        a.yaxis.set_major_formatter(ScalarFormatter())
        a.yaxis.set_minor_formatter(NullFormatter())
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "report", "dense_equiv.png"), dpi=120)


if __name__ == "__main__":
    main()
