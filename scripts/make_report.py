"""Collect runs/<phase>/*/ into figures (report/*.png) and markdown tables (report/<phase>_tables.md).

Usage: python scripts/make_report.py probe ep1 ep3
"""
import glob
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "report")

# validated categorical slots 1-3 (dataviz reference palette, light mode); seeds differ by line style
# slots 4/5 (yellow, magenta) pass the adjacent-pair checks of the reference palette for line charts
COLOR = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "B-v2a": "#eda100", "B-v2b": "#e87ba4",
         "B-1M": "#eb6834"}   # B-1M never appears together with B
STYLE = {0: "-", 1: "--"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LABEL = {"A": "A Baseline", "B": "B Speicher", "C": "C groß dicht",
         "B-v2a": "B-v2a ohne WD Keys", "B-v2b": "B-v2b + Temperatur", "B-1M": "B-1M (3 Schichten, 1M)"}
MEMORY_MODELS = ["B", "B-v2a", "B-v2b", "B-1M"]
# phases that are compared against the runs of another phase (same budget and schedule)
COMPARE_WITH = {"v2": ["ep1"], "v2b_ep3": ["ep3"]}
FULLVAL_DIAG = "diagnostics_fullval.json"     # scripts/diagnose_memory.py --windows 241 --out ...

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "legend.frameon": False, "lines.linewidth": 2.0,
})


def load_runs(phase):
    runs = []
    for ph in [phase] + COMPARE_WITH.get(phase, []):
        for d in sorted(glob.glob(os.path.join(ROOT, "runs", ph, "*"))):
            p = os.path.join(d, "run-info.json")
            if not os.path.exists(p):
                continue
            info = json.load(open(p))
            if info.get("status") != "done":
                continue
            m = pd.read_csv(os.path.join(d, "metrics.csv"))
            runs.append({"dir": d, "name": os.path.basename(d), "model": info["model_name"], "phase": ph,
                         "seed": info["seeds"]["init_seed"], "info": info, "metrics": m})
    return runs


def fig_sharpness(phase, runs):
    """Softmax sharpness over the top-k (v2 runs log it at every evaluation; v1 runs only have the
    end-of-training diagnostics) and the learned score scale (B-v2b)."""
    logged = [r for r in runs if "mem_val_eff_entries" in r["metrics"].columns]
    if not logged:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for r in logged:
        m = r["metrics"]
        ls = STYLE.get(r["seed"], ":")
        lab = f"{r['model']} (Seed {r['seed']})"
        axes[0].plot(m.tokens / 1e6, m.mem_val_eff_entries, color=COLOR[r["model"]], ls=ls, label=lab)
        axes[1].plot(m.tokens / 1e6, m.mem_score_scale_mean, color=COLOR[r["model"]], ls=ls, label=lab)
    refs = []
    for r in sorted(runs, key=lambda r: r["seed"]):
        diag = os.path.join(r["dir"], FULLVAL_DIAG)
        if not os.path.exists(diag):
            diag = os.path.join(r["dir"], "diagnostics.json")
        if r["model"] == "B" and os.path.exists(diag):
            v = json.load(open(diag))["softmax_eff_entries_per_head_mean"]
            axes[0].axhline(v, color=COLOR["B"], ls=STYLE.get(r["seed"], ":"), lw=1)
            refs.append(f"{v:.1f}".replace(".", ","))
    if refs:
        axes[0].text(0, max(float(x.replace(",", ".")) for x in refs) + 0.6,
                     f"B am Ende (Seeds {' / '.join(refs)})", color=INK2, fontsize=8)
    k = logged[0]["info"]["model_config"]["mem_knn"]
    axes[0].set_ylim(0, k + 1)
    axes[0].set_ylabel(f"effektiv gemischte Einträge je Kopf (von {k})")
    axes[0].set_title("Schärfe der Softmax über die Top-k (Val)", color=INK, loc="left")
    axes[1].set_ylabel("Score-Skala (Mittel über Köpfe)")
    axes[1].set_title("Gelernte Temperatur-Skala", color=INK, loc="left")
    for ax in axes:
        ax.set_xlabel("Tokens (Mio.)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(OUT, f"{phase}_sharpness.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_val_ppl(phase, runs):
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ends = []
    for r in runs:
        m = r["metrics"][r["metrics"].step > 0]
        ax.plot(m.tokens / 1e6, m.val_ppl, color=COLOR[r["model"]], ls=STYLE.get(r["seed"], ":"),
                label=f"{LABEL[r['model']]} (Seed {r['seed']})")
        ends.append((m.val_ppl.iloc[-1], m.tokens.iloc[-1] / 1e6, f"{r['model']}-s{r['seed']} {m.val_ppl.iloc[-1]:.1f}"))
    # end labels, spread vertically (log axis) so they do not collide
    ends.sort()
    placed = []
    for y, x, txt in ends:
        y_lab = y if not placed else max(y, placed[-1] * 1.09)
        placed.append(y_lab)
        ax.annotate(txt, (x, y), xytext=(x * 1.01 + 0.2, y_lab), textcoords="data", color=INK2, fontsize=8,
                    va="center", annotation_clip=False)
    ax.set_yscale("log")
    ax.set_xlabel("Trainings-Tokens (Mio.)")
    ax.set_ylabel("Validierungs-Perplexity (BPE, log)")
    ax.set_title(f"Validierungs-Perplexity über das Training – {phase}", color=INK, loc="left")
    lo = min(r["metrics"].val_ppl.min() for r in runs)
    ax.set_ylim(lo * 0.9, lo * 8)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUT, f"{phase}_val_ppl.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_relative_to_a(phase, runs):
    """Val-PPL of every run relative to the mean of the A seeds at the same token count."""
    a = [r["metrics"].set_index("tokens").val_ppl for r in runs if r["model"] == "A"]
    if not a:
        return None
    a_mean = pd.concat(a, axis=1).mean(axis=1)
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for r in runs:
        v = r["metrics"].set_index("tokens").val_ppl
        rel = 100 * (v / a_mean - 1)
        rel = rel[rel.index >= 8e6]                     # the first few evals are dominated by warm-up
        ax.plot(rel.index / 1e6, rel.values, color=COLOR[r["model"]], ls=STYLE.get(r["seed"], ":"),
                label=f"{LABEL[r['model']]} (Seed {r['seed']})")
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xlabel("Trainings-Tokens (Mio.)")
    ax.set_ylabel("Val-PPL relativ zu Mittel(A) (%)")
    ax.set_title(f"Abstand zur Baseline über das Training – {phase}", color=INK, loc="left")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(OUT, f"{phase}_relative_to_A.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_train_vs_val(phase, runs):
    models = sorted({r["model"] for r in runs})
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, model in zip(axes, models):
        for r in [r for r in runs if r["model"] == model]:
            m = r["metrics"][r["metrics"].step > 0]
            ls = STYLE.get(r["seed"], ":")
            ax.plot(m.tokens / 1e6, m.train_loss, color=INK2, ls=ls, lw=1.5, label=f"Train (Seed {r['seed']})")
            ax.plot(m.tokens / 1e6, m.val_loss, color=COLOR[model], ls=ls, label=f"Val (Seed {r['seed']})")
        ax.set_title(LABEL[model], color=INK, loc="left")
        ax.set_xlabel("Tokens (Mio.)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Loss (nats/Token)")
    lo = min(min(r["metrics"].val_loss.min(), r["metrics"].train_loss.min()) for r in runs)
    axes[0].set_ylim(lo - 0.1, lo + 2.0)
    fig.suptitle(f"Train- vs. Validierungs-Loss – {phase} (Abstand = Auswendiglernen)", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    path = os.path.join(OUT, f"{phase}_train_vs_val.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_memory_health(phase, runs):
    bruns = [r for r in runs if r["model"] in MEMORY_MODELS]
    if not bruns:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for r in bruns:
        m = r["metrics"]
        ls = STYLE.get(r["seed"], ":")
        axes[0].plot(m.tokens / 1e6, 100 * m.mem_val_usage, color=COLOR[r["model"]], ls=ls, label=f"{r['model']} Val-Set (Seed {r['seed']})")
        mt = m[m.step > 0]
        axes[0].plot(mt.tokens / 1e6, 100 * mt.mem_train_usage_interval, color=INK2, ls=ls, lw=1.5,
                     label=f"{r['model']} Training, je Intervall (Seed {r['seed']})")
        axes[1].plot(m.tokens / 1e6, 100 * m.mem_val_top1pct_share, color=COLOR[r["model"]], ls=ls, label=f"{r['model']} (Seed {r['seed']})")
    axes[0].axhline(60, color=INK2, lw=1, ls=":")
    axes[0].text(0, 61, "Kriterium 60 %", color=INK2, fontsize=8)
    axes[0].set_ylim(0, 102)
    axes[0].set_ylabel("Einträge mind. 1× gelesen (%)")
    axes[0].set_title("Key-Nutzung", color=INK, loc="left")
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Anteil aller Zugriffe (%)")
    axes[1].set_title("Anteil der Zugriffe auf das Top-1 % (Val)", color=INK, loc="left")
    for ax in axes:
        ax.set_xlabel("Tokens (Mio.)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(OUT, f"{phase}_memory_health.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_access_distribution(phase, runs):
    bruns = [r for r in runs if r["model"] in MEMORY_MODELS]
    if not bruns:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    bruns = [r for r in bruns if os.path.exists(os.path.join(r["dir"], "mem_access_val.npz"))]
    if not bruns:
        plt.close(fig)
        return None
    for r in bruns:
        ls = STYLE.get(r["seed"], ":")
        val = np.load(os.path.join(r["dir"], "mem_access_val.npz"))["counts"]
        tr = np.load(os.path.join(r["dir"], "mem_access_train.npy"))
        rank = np.arange(1, val.size + 1)
        axes[0].loglog(rank, np.sort(tr)[::-1] / tr.sum(), color=INK2, ls=ls, lw=1.5, label=f"{r['model']} Training gesamt (Seed {r['seed']})")
        axes[0].loglog(rank, np.sort(val)[::-1] / val.sum(), color=COLOR[r["model"]], ls=ls, label=f"{r['model']} Val-Set (Seed {r['seed']})")
        # Lorenz-style curve: share of reads covered by the most-read x% of entries
        cs = np.cumsum(np.sort(val)[::-1]) / val.sum()
        axes[1].plot(100 * rank / val.size, 100 * cs, color=COLOR[r["model"]], ls=ls, label=f"{r['model']} Val-Set (Seed {r['seed']})")
    axes[1].plot([0, 100], [0, 100], color=INK2, lw=1, ls=":", label="Gleichverteilung")
    axes[0].set_xlabel("Rang des Eintrags (nach Zugriffen)")
    axes[0].set_ylabel("Anteil der Zugriffe")
    axes[0].set_title("Zugriffe pro Eintrag, sortiert", color=INK, loc="left")
    axes[1].set_xlabel("meistgelesene Einträge (%)")
    axes[1].set_ylabel("kumulierter Anteil der Zugriffe (%)")
    axes[1].set_title("Konzentration der Zugriffe", color=INK, loc="left")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(OUT, f"{phase}_access_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fmt_m(x):
    return f"{x / 1e6:.1f} M"


def tables(phase, runs):
    lines = [f"### Einzelläufe ({phase})", "",
             "| Lauf | Params gesamt | ohne Emb. | aktiv/Token (ohne Emb.) | MACs/Token | Val-PPL | Test-PPL | Val-PPL (Wort) "
             "| Train tok/s | Decode b=1 tok/s | Prefill tok/s | VRAM Train (GiB) | Trainzeit | Key-Nutzung Val | Top-1 %-Anteil | KL |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(runs, key=lambda r: (r["model"], r["seed"])):
        i, res = r["info"], r["info"]["results"]
        p = i["params"]
        mem = res.get("mem_val")
        lines.append("| " + " | ".join([
            f"{r['model']}-s{r['seed']}", fmt_m(p["total"]), fmt_m(p["non_embedding"]),
            fmt_m(p["active_non_embedding_per_token"]), fmt_m(i["macs_per_token"]["total"]),
            f"{res['val_ppl']:.2f}", f"{res['test_ppl']:.2f}", f"{res['val_word_ppl']:.2f}",
            f"{res['train_tok_s_median']:,.0f}", f"{res['decode_b1_tok_s']:.0f}", f"{res['prefill_tok_s']:,.0f}",
            f"{res['peak_train_vram_gib']:.2f}", f"{i['train_time_s'] / 60:.0f} min",
            f"{100 * mem['usage']:.1f} %" if mem else "–", f"{100 * mem['top1pct_share']:.1f} %" if mem else "–",
            f"{mem['kl']:.2f}" if mem else "–"]) + " |")

    def ppl(model):
        return [r["info"]["results"]["val_ppl"] for r in sorted(runs, key=lambda r: r["seed"]) if r["model"] == model]

    A, C = ppl("A"), ppl("C")
    summary = {}
    for bname in [m for m in MEMORY_MODELS if ppl(m)]:
        B = ppl(bname)
        if not (A and C):
            break
        pa, pb, pc = np.mean(A), np.mean(B), np.mean(C)
        spread = max(abs(A[0] - A[-1]), abs(B[0] - B[-1]))     # 0 if only one seed
        single_seed = len(A) < 2 or len(B) < 2
        G = (pa - pb) / (pa - pc) if pa != pc else float("nan")
        usage = [r["info"]["results"]["mem_val"]["usage"] for r in runs if r["model"] == bname]
        top1 = [r["info"]["results"]["mem_val"]["top1pct_share"] for r in runs if r["model"] == bname]
        healthy = min(usage) >= 0.60 and max(top1) <= 0.50
        diff = pa - pb
        real = abs(diff) > 2 * spread
        if not healthy:
            verdict = "nicht schlüssig (Tabelle nicht gesund)"
        elif diff > 0 and real and G >= 0.5:
            verdict = "lohnt sich"
        elif diff > 0 and real:
            verdict = f"unklar ({bname} besser als A, aber knapp: G < 0,5)"
        elif not real:
            verdict = f"lohnt sich nicht ({bname} auf A-Niveau: Unterschied innerhalb 2·s)"
        else:
            verdict = f"lohnt sich nicht ({bname} schlechter als A)"
        summary[bname] = dict(ppl_A=pa, ppl_B=pb, ppl_C=pc, seeds_A=A, seeds_B=B, spread=spread, G=G,
                              usage_min=min(usage), top1_max=max(top1), healthy=healthy, verdict=verdict)
        lines += ["", f"### Auswertung gegen die Erfolgskriterien ({phase}, {bname})", "",
                  "| Größe | Wert |", "|---|---|",
                  f"| PPL A (Mittel; Seeds) | {pa:.2f} ({', '.join(f'{x:.2f}' for x in A)}) |",
                  f"| PPL {bname} (Mittel; Seeds) | {pb:.2f} ({', '.join(f'{x:.2f}' for x in B)}) |",
                  f"| PPL C | {pc:.2f} |",
                  f"| Seed-Spanne s | {spread:.3f} (Schwelle 2·s = {2 * spread:.3f}) |",
                  f"| PPL_A − PPL_{bname} | {diff:+.3f} ({'über' if real else 'innerhalb'} 2·s) |",
                  f"| Lückenschluss G = (A−{bname})/(A−C) | {G:.2f} |",
                  f"| Key-Nutzung Val (min. über Seeds) | {100 * min(usage):.1f} % |",
                  f"| Anteil Top-1 % der Einträge (max. über Seeds) | {100 * max(top1):.1f} % |",
                  f"| Tabelle gesund (≥ 60 % und Top-1 % ≤ 50 %) | {'ja' if healthy else 'nein'} |",
                  f"| **Urteil** | **{verdict}** |"]
        if single_seed:
            lines.append("\n_Nur ein Seed je Modell: das Seed-Rauschen ist hier nicht messbar, das Urteil ist nur vorläufig._")
    lines += paired_vs_b(runs)
    lines += stage1b_quicktest(runs)
    path = os.path.join(OUT, f"{phase}_tables.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT, f"{phase}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    return path


def stage1b_quicktest(runs):
    """Criterion fixed before the run (REPORT.md, stage 1b): PPL(B-1M) <= 0.90 x PPL(A) on the held-out
    Wikipedia validation set at the end of training; WikiText-103 validation only reported."""
    by = {r["model"]: r for r in runs if r["seed"] == 0}
    if "A" not in by or "B-1M" not in by:
        return []
    a, b = by["A"]["info"]["results"], by["B-1M"]["info"]["results"]
    ratio = b["val_ppl"] / a["val_ppl"]
    out = ["", "### Schnelltest Stufe 1b: A gegen B-1M (500 M frische Wikipedia-Tokens, je Seed 0)", "",
           "| | Val-PPL Wikipedia | Val-PPL WikiText-103 | Train tok/s | VRAM Train (GiB) | Trainzeit |",
           "|---|---|---|---|---|---|"]
    for name in ("A", "B-1M"):
        r = by[name]
        res = r["info"]["results"]
        out.append(f"| {name} | {res['val_ppl']:.2f} | {res.get('val2_ppl', float('nan')):.2f} | "
                   f"{res['train_tok_s_median']:,.0f} | {res['peak_train_vram_gib']:.2f} | "
                   f"{r['info']['train_time_s'] / 60:.0f} min |")
    out += ["", "| Kriterium | Wert |", "|---|---|",
            f"| PPL(B-1M) / PPL(A), Wikipedia-Val | {ratio:.4f} ({100 * (ratio - 1):+.2f} %) |",
            f"| WikiText-103-Val (nur berichtet) | {100 * (b.get('val2_ppl', 0) / a.get('val2_ppl', 1) - 1):+.2f} % |",
            f"| **≥ 10 % niedriger (Verhältnis ≤ 0,90)** | **{'ja → großer Test' if ratio <= 0.90 else 'nein → kein großer Test'}** |"]
    return out


def paired_vs_b(runs):
    """Variant vs. B at the same init seed (identical initial weights and token stream), plus softmax
    sharpness on the full validation set (diagnostics_fullval.json, same script for all runs).
    Gates fixed before the v2b_ep3 runs (V2_SHARPNESS.md):
      fix greift   : effective mixed entries per head <= 0.9 x B, for both seeds
      klar besser  : variant better than B at both seeds and mean improvement > 2*s,
                     s = max seed spread (val PPL) of B and of the variant"""
    out = []
    by = {(r["model"], r["seed"]): r for r in runs}
    for v in [m for m in MEMORY_MODELS if m != "B"]:
        seeds = sorted(sd for (m, sd) in by if m == v and ("B", sd) in by)
        if not seeds:
            continue
        out += ["", f"### Paarweiser Vergleich {v} gegen B (gleicher Seed = gleiche Startgewichte, gleiche Daten)", "",
                "| Seed | Val-PPL B | Val-PPL " + v + " | Δ Val | Δ Val % | Test-PPL B | Test-PPL " + v + " | Δ Test % "
                "| eff. Einträge B | eff. Einträge " + v + " | Top-1-Gewicht B | Top-1-Gewicht " + v + " | Skala je Kopf |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        deltas, eff_ok = [], []
        for sd in seeds:
            rb, rv = by[("B", sd)]["info"]["results"], by[(v, sd)]["info"]["results"]
            db = os.path.join(by[("B", sd)]["dir"], FULLVAL_DIAG)
            dv = os.path.join(by[(v, sd)]["dir"], FULLVAL_DIAG)
            gb = json.load(open(db)) if os.path.exists(db) else {}
            gv = json.load(open(dv)) if os.path.exists(dv) else {}
            eb, ev = gb.get("softmax_eff_entries_per_head_mean"), gv.get("softmax_eff_entries_per_head_mean")
            tb, tv = gb.get("softmax_top1_weight_mean"), gv.get("softmax_top1_weight_mean")
            d = rv["val_ppl"] - rb["val_ppl"]
            deltas.append(d)
            eff_ok.append(eb is not None and ev is not None and ev <= 0.9 * eb)
            scale = rv.get("mem_score_scale_per_head")
            f = lambda x, n=2: "–" if x is None else f"{x:.{n}f}"   # noqa: E731
            out.append("| " + " | ".join([
                str(sd), f"{rb['val_ppl']:.2f}", f"{rv['val_ppl']:.2f}", f"{d:+.3f}",
                f"{100 * d / rb['val_ppl']:+.2f} %", f"{rb['test_ppl']:.2f}", f"{rv['test_ppl']:.2f}",
                f"{100 * (rv['test_ppl'] / rb['test_ppl'] - 1):+.2f} %", f(eb), f(ev), f(tb, 4), f(tv, 4),
                ", ".join(f"{x:.2f}" for x in scale) if scale else "–"]) + " |")
        vb = [by[("B", sd)]["info"]["results"]["val_ppl"] for sd in seeds]
        vv = [by[(v, sd)]["info"]["results"]["val_ppl"] for sd in seeds]
        s_ = max(abs(vb[0] - vb[-1]), abs(vv[0] - vv[-1]))
        mean_d = float(np.mean(deltas))
        better = all(x < 0 for x in deltas) and -mean_d > 2 * s_ and len(seeds) > 1
        out += ["", "| Gate | Wert |", "|---|---|",
                f"| mittlere Differenz {v} − B (Val-PPL) | {mean_d:+.3f} |",
                f"| s = max. Seed-Spanne (B, {v}) | {s_:.3f} → 2·s = {2 * s_:.3f} |",
                f"| **{v} klar besser als B** (beide Seeds besser und Mittel > 2·s) | **{'ja' if better else 'nein'}** |",
                f"| **Fix greift** (eff. Einträge ≤ 0,9 × B bei beiden Seeds) | **{'ja' if eff_ok and all(eff_ok) else 'nein'}** |"]
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    for phase in sys.argv[1:]:
        runs = load_runs(phase)
        if not runs:
            print(f"{phase}: no finished runs")
            continue
        outs = [fig_val_ppl(phase, runs), fig_relative_to_a(phase, runs), fig_train_vs_val(phase, runs), fig_memory_health(phase, runs),
                fig_access_distribution(phase, runs), fig_sharpness(phase, runs), tables(phase, runs)]
        print(phase, [o for o in outs if o])
        print(open(os.path.join(OUT, f"{phase}_tables.md")).read())


if __name__ == "__main__":
    main()
