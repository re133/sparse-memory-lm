"""Step 3 evaluation against the pre-registered criteria (REPORT.md, "Schritt 3"): Q, Q+T, Q+D from runs/qwen_cloud.

  python scripts/qwen_step3_eval.py      -> report/qwen/step3_summary.json + markdown on stdout

Hilft es?  token PPL on val_new: "deutlich" if Q+T <= 0.95 Q and Q+T <= 0.98 Q+D, "etwas" if Q+T <= 0.98 Q, else "nicht".
Schadet es? (for Q+T and Q+D vs Q) mean accuracy of the 6 tasks drops by <= 1.0 pp; no task drops by more than
  max(2 pp, 2 x standard error of the difference, sqrt(se_Q^2 + se_X^2) from lm-eval); PPL on val_known and
  val_known_same at most +1 %; chat judged blind by the user (> 3 of 12 worse -> "schadet"), filled in separately.
Wissen eingepflanzt?  fact test: acc_train(Q+T) - acc_train(Q+D) >= 10 pp and acc_heldout(Q+T) >= acc_heldout(Q+D) - 2 pp
  (paired over the same items; 95 % bootstrap intervals of the differences are reported).
"""
import json
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "runs", "qwen_cloud")
TASKS = ["mmlu", "arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande"]


def ppl(run):
    x = json.load(open(os.path.join(R, run, "run-info.json")))["results"]
    return {k: {"ppl": v["ppl"], "knowledge_ppl": v.get("knowledge_ppl")} for k, v in x.items()}


def general(path):
    t = json.load(open(os.path.join(R, path)))["tasks"]
    return {k: {"acc": 100 * t[k]["acc,none"], "se": 100 * t[k].get("acc_stderr,none", float("nan"))} for k in TASKS}


def facts(m):
    return {it["id"]: it for it in json.load(open(os.path.join(R, m, "facts.json")))["items"]}


def boot_diff(a, b, rng, n=10000):
    a, b = np.array(a, float), np.array(b, float)
    idx = rng.integers(0, len(a), (n, len(a)))
    d = (a[idx] - b[idx]).mean(1) * 100
    return float((a - b).mean() * 100), [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def main():
    out = {}
    P = {m: ppl(r) for m, r in (("Q", "Q"), ("QT", "QT-s0"), ("QD", "QD-s0"))}
    out["ppl"] = P
    q, qt, qd = (P[m]["val_new"]["ppl"] for m in ("Q", "QT", "QD"))
    verdict = ("hilft deutlich" if qt <= 0.95 * q and qt <= 0.98 * qd else "hilft etwas" if qt <= 0.98 * q
               else "hilft nicht")
    out["hilft"] = {"QT/Q": qt / q, "QT/QD": qt / qd, "QD/Q": qd / q, "verdict": verdict}

    G = {"Q": general("Q/general_ac.json"), "QT": general("QT/general.json"), "QD": general("QD/general.json")}
    out["general"] = G
    sch = {}
    for m in ("QT", "QD"):
        drops = {}
        for t in TASKS:
            d = G[m][t]["acc"] - G["Q"][t]["acc"]
            se = math.sqrt(G[m][t]["se"] ** 2 + G["Q"][t]["se"] ** 2)
            drops[t] = {"diff_pp": d, "limit_pp": -max(2.0, 2 * se), "ok": d >= -max(2.0, 2 * se)}
        mean_d = np.mean([G[m][t]["acc"] for t in TASKS]) - np.mean([G["Q"][t]["acc"] for t in TASKS])
        known = {k: P[m][k]["ppl"] / P["Q"][k]["ppl"] - 1 for k in ("val_known", "val_known_same")}
        sch[m] = {"tasks": drops, "mean_diff_pp": float(mean_d), "mean_ok": bool(mean_d >= -1.0),
                  "tasks_ok": all(v["ok"] for v in drops.values()), "known_ppl_rel": known,
                  "known_ok": all(v <= 0.01 for v in known.values())}
        sch[m]["without_chat"] = "schadet nicht (bis auf Chat)" if (sch[m]["mean_ok"] and sch[m]["tasks_ok"]
                                                                    and sch[m]["known_ok"]) else "schadet"
    out["schadet"] = sch

    F = {m: facts(m) for m in ("Q", "QT", "QD")}
    ids = sorted(F["Q"])
    rng = np.random.default_rng(0)
    fx = {}
    for split in ("train", "heldout"):
        sel = [i for i in ids if F["Q"][i]["split"] == split]
        acc = {m: 100 * np.mean([F[m][i]["correct"] for i in sel]) for m in F}
        d, ci = boot_diff([F["QT"][i]["correct"] for i in sel], [F["QD"][i]["correct"] for i in sel], rng)
        dq, ciq = boot_diff([F["QT"][i]["correct"] for i in sel], [F["Q"][i]["correct"] for i in sel], rng)
        fx[split] = {"n": len(sel), "acc": acc, "QT_minus_QD_pp": d, "QT_minus_QD_ci95": ci,
                     "QT_minus_Q_pp": dq, "QT_minus_Q_ci95": ciq}
    fx["verdict"] = ("Wissen eingepflanzt" if fx["train"]["QT_minus_QD_pp"] >= 10 and fx["heldout"]["QT_minus_QD_pp"] >= -2
                     else "nicht erreicht")
    out["facts"] = fx
    json.dump(out, open(os.path.join(ROOT, "report", "qwen", "step3_summary.json"), "w"), indent=1)

    print("| Set | Q | Q+T | Q+D |\n|---|---|---|---|")
    for k in ("val_new", "val_known", "val_known_same", "mem_probe"):
        print(f"| {k} | {P['Q'][k]['ppl']:.3f} | {P['QT'][k]['ppl']:.3f} | {P['QD'][k]['ppl']:.3f} |")
    print(f"| val_new (Wissens-Tokens) | {P['Q']['val_new']['knowledge_ppl']:.2f} | {P['QT']['val_new']['knowledge_ppl']:.2f} | "
          f"{P['QD']['val_new']['knowledge_ppl']:.2f} |")
    print(f"\nHilft: QT/Q {out['hilft']['QT/Q']:.3f}, QT/QD {out['hilft']['QT/QD']:.3f}, QD/Q {out['hilft']['QD/Q']:.3f} -> {verdict}")
    print("\n| Aufgabe | Q | Q+T | Q+D |\n|---|---|---|---|")
    for t in TASKS:
        print(f"| {t} | {G['Q'][t]['acc']:.1f} ± {G['Q'][t]['se']:.1f} | {G['QT'][t]['acc']:.1f} ({sch['QT']['tasks'][t]['diff_pp']:+.1f}, "
              f"Grenze {sch['QT']['tasks'][t]['limit_pp']:.1f}) | {G['QD'][t]['acc']:.1f} ({sch['QD']['tasks'][t]['diff_pp']:+.1f}, "
              f"Grenze {sch['QD']['tasks'][t]['limit_pp']:.1f}) |")
    for m in ("QT", "QD"):
        s = sch[m]
        print(f"{m}: mean {s['mean_diff_pp']:+.2f} pp ok={s['mean_ok']}, tasks ok={s['tasks_ok']}, known "
              f"{ {k: round(100 * v, 2) for k, v in s['known_ppl_rel'].items()} } ok={s['known_ok']} -> {s['without_chat']}")
    for split in ("train", "heldout"):
        f = fx[split]
        print(f"facts {split}: Q {f['acc']['Q']:.1f} %, QT {f['acc']['QT']:.1f} %, QD {f['acc']['QD']:.1f} %; "
              f"QT-QD {f['QT_minus_QD_pp']:+.1f} pp {[round(c, 1) for c in f['QT_minus_QD_ci95']]}, "
              f"QT-Q {f['QT_minus_Q_pp']:+.1f} pp {[round(c, 1) for c in f['QT_minus_Q_ci95']]}")
    print("Wissen eingepflanzt:", fx["verdict"])


if __name__ == "__main__":
    main()
