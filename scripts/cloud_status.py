"""Status of the cloud queue -> block between the CLOUD-STATUS markers in REPORT.md and report/cloud_status.json.
Criteria exactly as fixed in REPORT.md before the build ("Kriterien für die Cloud-Läufe").
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from hampter_status import de, pct, read_csv, status_word, thermal  # noqa: E402

BEGIN, END = "<!-- CLOUD-STATUS:BEGIN -->", "<!-- CLOUD-STATUS:END -->"
ROWS = [  # (key, run dir under runs/, label)
    ("home", "hampter/B-1M-sparse-s0", "B-1M-sparse s0 zu Hause (RX 9070, PyTorch-Referenz)"),
    ("B1", "cloud/B-1M-s0", "B-1M (Cloud, Kontrolle)"),
    ("B4", "cloud/B-4M-s0", "B-4M (Cloud)"),
    ("B16", "cloud/B-16M-s0", "B-16M (Cloud)"),
]
WORTH_RATIO = 0.97


def summary(rel):
    d = os.path.join(ROOT, "runs", rel)
    p = os.path.join(d, "run-info.json")
    if not os.path.exists(p):
        return {"status": "ausstehend"}
    info = json.load(open(p))
    s = {"status": info.get("status"), "total_tokens": info["train_config"]["total_tokens"],
         "gpu": info.get("hardware", {}).get("gpu"), "params_total": info.get("params", {}).get("total"),
         "git": info["git"]["commit"][:7] + ("-dirty" if info["git"]["dirty"] else ""), "thermal": thermal(d)}
    m = read_csv(os.path.join(d, "metrics.csv"))
    if m:
        s["last_eval_tokens"], s["last_eval_ppl"] = float(m[-1]["tokens"]), float(m[-1]["val_ppl"])
    if info.get("status") == "done":
        r = info["results"]
        s.update({"val_ppl": r["val_ppl"], "val2_ppl": r.get("val2_ppl"), "train_time_s": info["train_time_s"],
                  "duration_s": info.get("duration_s"), "tok_s": r.get("train_tok_s_median"),
                  "peak_train_vram_gib": r.get("peak_train_vram_gib"), "usage": r.get("mem_val", {}).get("usage")})
    return s


def verdict(S):
    b1, b4, b16 = (S[k].get("val_ppl") for k in ("B1", "B4", "B16"))
    if b1 is None or b4 is None:
        return None, "ausstehend"
    if b4 >= b1:
        return {"b4_over_b1": b4 / b1}, "lohnt sich nicht"
    if b4 <= WORTH_RATIO * b1 and b16 is not None and b16 < b4:
        return {"b4_over_b1": b4 / b1, "b16_over_b4": b16 / b4}, "lohnt sich"
    if b4 <= WORTH_RATIO * b1 and b16 is None:
        return {"b4_over_b1": b4 / b1}, "ausstehend (B-4M erfüllt die 3 %, B-16M fehlt)"
    return {"b4_over_b1": b4 / b1, **({"b16_over_b4": b16 / b4} if b16 else {})}, "unklar"


def render(S, V, word):
    L = [BEGIN, "", f"**Zwischenstand Cloud** (automatisch, `scripts/cloud_status.py`, Stand {time.strftime('%Y-%m-%d %H:%M')})",
         "", "| Lauf | Status | GPU | Val-PPL Wikipedia | Val-PPL WikiText | Trainzeit | tok/s | VRAM Train | Nutzung | "
         "max. Temp. GPU / Speicher | max. Leistung |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, _, label in ROWS:
        s = S[key]
        th = s.get("thermal") or {}
        L.append("| " + " | ".join([
            label, status_word(s), s.get("gpu") or "–", de(s.get("val_ppl"), 3), de(s.get("val2_ppl"), 2),
            f"{de(s['train_time_s'] / 60, 0)} min" if s.get("train_time_s") else "–",
            de(s.get("tok_s"), 0), f"{de(s['peak_train_vram_gib'], 1)} GiB" if s.get("peak_train_vram_gib") else "–",
            pct(s["usage"] - 0, 1).lstrip("+") if s.get("usage") is not None else "–",
            (f"{de(th.get('edge_c'), 0)} / {de(th.get('mem_c'), 0)} °C") if th else "–",
            f"{de(th.get('power_w'), 0)} W" if th.get("power_w") else "–"]) + " |")
    L += ["", "| Kriterium (vorher festgelegt) | Messwert | Ergebnis |", "|---|---|---|"]
    if V:
        txt = f"B-4M / B-1M = {de(V['b4_over_b1'], 4)} ({pct(V['b4_over_b1'] - 1, 2)}; Grenze 0,97)"
        if "b16_over_b4" in V:
            txt += f"; B-16M / B-4M = {de(V['b16_over_b4'], 4)} ({pct(V['b16_over_b4'] - 1, 2)})"
        L.append(f"| lohnt sich: B-4M ≥ 3 % besser als B-1M (Cloud) und B-16M besser als B-4M | {txt} | **{word}** |")
    else:
        L.append(f"| lohnt sich: B-4M ≥ 3 % besser als B-1M (Cloud) und B-16M besser als B-4M | – | {word} |")
    home, b1 = S["home"].get("val_ppl"), S["B1"].get("val_ppl")
    if home and b1:
        L += ["", f"Kontrolle: B-1M in der Cloud (Triton-Kernels, H200) gegenüber zu Hause (PyTorch-Referenz, RX 9070): "
                  f"{de(b1, 3)} gegenüber {de(home, 3)} ({pct(b1 / home - 1, 2)}; Seed-Spanne zu Hause 0,39 %)."]
    L += ["", END]
    return "\n".join(L)


def main():
    S = {key: summary(rel) for key, rel, _ in ROWS}
    V, word = verdict(S)
    os.makedirs(os.path.join(ROOT, "report"), exist_ok=True)
    with open(os.path.join(ROOT, "report", "cloud_status.json"), "w") as f:
        json.dump({"runs": S, "criteria": V, "verdict": word}, f, indent=2)
    block = render(S, V, word)
    rp = os.path.join(ROOT, "REPORT.md")
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
