"""Status of the Hampter queue (stage 1c) -> block between the HAMPTER-STATUS markers in REPORT.de.md and
report/hampter_status.json. Called by scripts/run_hampter.py after every run; safe to run at any time.
Criteria exactly as fixed before the start (REPORT.md, section "Stage 1c").
"""
import csv
import json
import math
import os
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
BEGIN, END = "<!-- HAMPTER-STATUS:BEGIN -->", "<!-- HAMPTER-STATUS:END -->"
ROWS = [  # (key, run dir, label)
    ("ref_B", "s1b/B-1M-s0", "B-1M s0 (Schnelltest, dichter Optimizer, Referenz)"),
    ("ref_A", "s1b/A-s0", "A s0 (Schnelltest, Referenz)"),
    ("B0", "hampter/B-1M-sparse-s0", "B-1M-sparse s0"),
    ("Aeq", "hampter/A-eqtime-s0", "A s0 bei gleicher Rechenzeit"),
    ("A1", "hampter/A-s1", "A s1"),
    ("B1", "hampter/B-1M-sparse-s1", "B-1M-sparse s1"),
]
OPT_MAX_REL, STABLE_MAX_RATIO, CLEAR_MAX_RATIO = 0.02, 0.90, 0.95


def de(x, nd=2):
    """German number format (1.234,56)."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "–"
    return f"{x:,.{nd}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def pct(x, nd=1):
    return "–" if x is None else ("+" if x >= 0 else "−") + de(abs(100 * x), nd) + " %"


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def thermal(run_dir):
    rows = read_csv(os.path.join(run_dir, "gpu_thermal.csv"))
    if not rows:
        return None

    def col(k):
        return np.array([float(r[k]) for r in rows if r.get(k) not in ("", None)])
    out = {k: float(col(k).max()) for k in ("edge_c", "junction_c", "mem_c", "power_w", "vram_used_gib")
           if col(k).size}
    load = [float(r["sclk_mhz"]) for r in rows if r.get("power_w") and float(r["power_w"]) > 100 and r["sclk_mhz"]]
    out["sclk_mhz_median_load"] = float(np.median(load)) if load else None
    out["sclk_mhz_max"] = float(col("sclk_mhz").max()) if col("sclk_mhz").size else None
    out["n_samples"] = len(rows)
    return out


def run_summary(rel):
    d = os.path.join(RUNS, rel)
    p = os.path.join(d, "run-info.json")
    if not os.path.exists(p):
        return {"status": "ausstehend"}
    info = json.load(open(p))
    m = read_csv(os.path.join(d, "metrics.csv"))
    s = {"status": info.get("status"), "total_tokens": info["train_config"]["total_tokens"],
         "git": info["git"]["commit"][:7] + ("-dirty" if info["git"]["dirty"] else ""), "thermal": thermal(d)}
    if m:
        s["last_eval_tokens"] = float(m[-1]["tokens"])
        s["last_eval_ppl"] = float(m[-1]["val_ppl"])
    tl = read_csv(os.path.join(d, "train_log.csv"))
    tok = np.array([float(r["tok_s"]) for r in tl[2:]]) if len(tl) > 2 else None
    if tok is not None and tok.size:
        s["tok_s_median"], s["tok_s_p5"] = float(np.median(tok)), float(np.percentile(tok, 5))
    if info.get("abort_check"):
        s["abort_check"] = info["abort_check"]
    if info.get("status") == "done":
        r = info["results"]
        s.update({"val_ppl": r["val_ppl"], "val2_ppl": r.get("val2_ppl"), "train_time_s": info["train_time_s"],
                  "peak_train_vram_gib": r["peak_train_vram_gib"]})
    return s


def criteria(S):
    out = {}
    rb, b0, b1, a0, a1, aeq = (S[k].get("val_ppl") for k in ("ref_B", "B0", "B1", "ref_A", "A1", "Aeq"))
    if b0 is not None:
        rel = b0 / rb - 1
        out["optimizer_ok"] = {"value": rel, "limit": OPT_MAX_REL, "met": rel <= OPT_MAX_REL}
    if None not in (b0, b1, a0, a1):
        mean_a = (a0 + a1) / 2
        ratios = [b0 / mean_a, b1 / mean_a]
        out["stable"] = {"mean_A": mean_a, "ratios": ratios, "limit": STABLE_MAX_RATIO,
                         "met": all(r <= STABLE_MAX_RATIO for r in ratios)}
    if None not in (b0, aeq):
        ratio = b0 / aeq
        verdict = ("klarer Vorteil" if ratio <= CLEAR_MAX_RATIO else "konkurrenzfähig" if ratio <= 1.0
                   else "nicht konkurrenzfähig")
        out["equal_time"] = {"ratio": ratio, "verdict": verdict,
                             "train_time_B_s": S["B0"].get("train_time_s"), "train_time_Aeq_s": S["Aeq"].get("train_time_s")}
    return out


def status_word(s):
    st = s.get("status")
    if st == "done":
        return "fertig"
    if st == "running":
        return (f"läuft ({de(s['last_eval_tokens'] / 1e6, 0)} M Tokens, PPL {de(s['last_eval_ppl'])})"
                if "last_eval_tokens" in s else "läuft")
    return {"aborted": "**abgebrochen**", "diverged": "**divergiert**"}.get(st, st or "ausstehend")


def render(S, C):
    now = time.strftime("%Y-%m-%d %H:%M")
    L = [BEGIN, "", f"**Zwischenstand** (automatisch erzeugt von `scripts/hampter_status.py`, Stand {now})", "",
         "| Lauf | Status | Tokens | Val-PPL Wikipedia | Val-PPL WikiText | Trainzeit | tok/s Median (5 %-Quantil) "
         "| VRAM Train | max. edge / Hotspot / Speicher | max. Leistung | Takt unter Last |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, _, label in ROWS:
        s = S[key]
        th = s.get("thermal") or {}
        temps = (" / ".join(de(th.get(k), 0) for k in ("edge_c", "junction_c", "mem_c")) + " °C") if th else "–"
        L.append("| " + " | ".join([
            label, status_word(s),
            f"{de(s['total_tokens'] / 1e6, 0)} M" if "total_tokens" in s else "–",
            de(s.get("val_ppl"), 3), de(s.get("val2_ppl"), 2),
            f"{de(s['train_time_s'] / 60, 0)} min" if s.get("train_time_s") else "–",
            f"{de(s['tok_s_median'], 0)} ({de(s['tok_s_p5'], 0)})" if s.get("tok_s_median") else "–",
            f"{de(s['peak_train_vram_gib'])} GiB" if s.get("peak_train_vram_gib") else "–",
            temps, f"{de(th.get('power_w'), 0)} W" if th.get("power_w") else "–",
            f"{de(th.get('sclk_mhz_median_load'), 0)} MHz" if th.get("sclk_mhz_median_load") else "–",
        ]) + " |")
    L.append("")
    ac = S["B0"].get("abort_check")
    if ac:
        L.append(f"**Abbruchregel** (B-1M-sparse s0 bei {de(ac['tokens'] / 1e6, 1)} M Tokens): PPL {de(ac['val_ppl'], 3)} "
                 f"gegenüber {de(ac['ref_val_ppl'], 3)} beim bisherigen B-1M s0 = {pct(ac['rel_diff'], 2)} "
                 f"(Grenze {pct(ac['max_rel'], 0)}) → **{'abgebrochen' if ac['aborted'] else 'weiter'}**.")
    else:
        L.append("**Abbruchregel** (B-1M-sparse s0 bei 100 M Tokens): noch nicht erreicht.")
    bud_p = os.path.join(RUNS, "hampter", "a_eqtime_budget.json")
    if os.path.exists(bud_p):
        b = json.load(open(bud_p))
        L.append("")
        L.append(f"**Budget A bei gleicher Rechenzeit:** {de(b['b_sparse_s0_train_time_s'], 0)} s Trainzeit von "
                 f"B-1M-sparse s0 × {de(b['a_tok_s'], 0)} tok/s (A s0 im Schnelltest) = {de(b['tokens'] / 1e6, 1)} M "
                 f"Tokens ({b['steps']} Schritte)"
                 + ("." if b.get("fits_in_one_pass", True) else ", **mehr als der 1,5-Mrd.-Token-Strom, Daten wiederholen sich**."))
    L += ["", "| Kriterium (vorher festgelegt) | Bedingung | Messwert | Ergebnis |", "|---|---|---|---|"]
    rb = S["ref_B"].get("val_ppl")
    o = C.get("optimizer_ok")
    L.append(f"| Optimizer ok | PPL(B-1M-sparse s0) ≤ 1,02 × {de(rb, 3)} = {de(rb * 1.02, 3) if rb else '–'} | "
             + (f"{de(S['B0']['val_ppl'], 3)} ({pct(o['value'], 2)}) | **{'erfüllt' if o['met'] else 'nicht erfüllt'}** |"
                if o else "– | ausstehend |"))
    st = C.get("stable")
    if st:
        L.append(f"| Stabil | beide B-1M-sparse-Seeds ≤ 0,90 × Mittel(A s0, A s1) = {de(0.9 * st['mean_A'], 3)} | "
                 f"s0 {de(st['ratios'][0], 3)}×, s1 {de(st['ratios'][1], 3)}× (Mittel A {de(st['mean_A'], 3)}) | "
                 f"**{'erfüllt' if st['met'] else 'nicht erfüllt'}** |")
    else:
        L.append("| Stabil | beide B-1M-sparse-Seeds ≤ 0,90 × Mittel(A s0, A s1) | – | ausstehend |")
    eq = C.get("equal_time")
    if eq:
        L.append(f"| Gleiche Rechenzeit | PPL(B-1M-sparse s0) / PPL(A gleiche Zeit): ≤ 1,00 konkurrenzfähig, ≤ 0,95 "
                 f"klarer Vorteil | {de(eq['ratio'], 3)} ({pct(eq['ratio'] - 1, 1)}); Trainzeit B "
                 f"{de((eq['train_time_B_s'] or 0) / 60, 0)} min, A {de((eq['train_time_Aeq_s'] or 0) / 60, 0)} min | "
                 f"**{eq['verdict']}** |")
    else:
        L.append("| Gleiche Rechenzeit | PPL(B-1M-sparse s0) / PPL(A gleiche Zeit): ≤ 1,00 konkurrenzfähig, ≤ 0,95 "
                 "klarer Vorteil | – | ausstehend |")
    ths = [S[k]["thermal"] for k in ("B0", "Aeq", "A1", "B1") if S[k].get("thermal")]
    if ths:
        mx = {k: max(t.get(k, float("nan")) for t in ths) for k in ("edge_c", "junction_c", "mem_c", "power_w")}
        L += ["", f"**GPU-Höchstwerte über alle Hampter-Läufe** (alle 10 s gemessen, `gpu_thermal.csv` je Lauf): "
                  f"edge {de(mx['edge_c'], 0)} °C, Hotspot {de(mx['junction_c'], 0)} °C, Speicher {de(mx['mem_c'], 0)} °C "
                  f"(Grenzen laut Treiber 110 / 110 / 108 °C), Leistung {de(mx['power_w'], 0)} W."]
    L += ["", END]
    return "\n".join(L)


def main():
    S = {key: run_summary(rel) for key, rel, _ in ROWS}
    C = criteria(S)
    os.makedirs(os.path.join(ROOT, "report"), exist_ok=True)
    with open(os.path.join(ROOT, "report", "hampter_status.json"), "w") as f:
        json.dump({"runs": S, "criteria": C}, f, indent=2)
    block = render(S, C)
    rp = os.path.join(ROOT, "REPORT.de.md")
    text = open(rp).read()
    if BEGIN in text and END in text:
        a, rest = text.split(BEGIN, 1)
        _, b = rest.split(END, 1)
        text = a + block + b
    else:
        text = text.rstrip("\n") + "\n\n" + block + "\n"
    with open(rp, "w") as f:
        f.write(text)
    print(block)


if __name__ == "__main__":
    main()
