# v2 „sharpness“: Varianten von B (vorbereitet, noch nicht gelaufen)

## Anlass

Stufe 1 (1 Epoche) hat gezeigt: Die Speichertabelle ist gesund (≈ 99 % Nutzung) und trägt messbar bei
(ohne sie +17 % PPL). B ist trotzdem nur ≈ 2 % besser als A. Auffällig ist die Gewichtung **innerhalb**
der Top-32: Jeder Kopf mischt effektiv 30,5 von 32 Einträgen (Top-1-Gewicht 0,066, gleichverteilt
0,031). Die Score-Skala ist kaum gewachsen (Key-Norm 0,41 → 0,48, BatchNorm-γ ≈ 1,09). Die Schicht
ruft also nicht gezielt ab, sondern mittelt grob.

Zwei mögliche Bremsen:

1. **Weight Decay auf den Sub-Keys** (v1: 0,1, wie in der Meta-Referenz) zieht die Key-Norm und damit die
   Score-Skala Richtung null.
2. Die **Query-BatchNorm** fixiert die Skala der Query. Schärfer werden kann die Softmax dann nur noch
   über die BatchNorm-γ (1.024 einzelne Parameter je Dimension) oder über die Keys.

## Varianten (per Konfiguration schaltbar, Defaults = v1)

| Name (`--model`) | `mem_keys_weight_decay` | `mem_score_scale` | Unterschied zu B |
|---|---|---|---|
| `B` | `True` | `"none"` | – (Stufe 1) |
| `B-v2a` | `False` | `"none"` | Sub-Keys ohne Weight Decay |
| `B-v2b` | `False` | `"learned"` | v2a + lernbare Skala s_h = exp(log s_h) je Kopf: `w = softmax(s_h · scores)` |

- Die Skala wirkt **nur auf die Gewichtung**, nie auf die Auswahl der Top-k (s_h > 0 erhält die
  Reihenfolge; per Test geprüft). Startwert `--mem_score_scale_init` (Default 1,0 = exakt v1).
- `log_score_scale` liegt in der No-Decay-Gruppe mit der Basis-LR; die Sub-Keys bei v2a/v2b ebenfalls.
- Die Initialisierung verbraucht in v2a/v2b keine zusätzlichen Zufallszahlen. **B-s0, B-v2a-s0 und
  B-v2b-s0 starten also mit identischen Gewichten** (gleiches für s1) und sehen denselben Token-Strom.
  Paarweise Vergleiche sind damit sauberer als der Vergleich über Seeds (Rest-Rauschen nur durch
  nicht-deterministische GPU-Kernels).
- Neu protokolliert (alle Speichermodelle): effektiv gemischte Einträge je Kopf und Top-1-Gewicht auf
  dem Val-Set, mittlere Score-Skala, mittlere Key-Norm (`metrics.csv`), Skala je Kopf am Ende
  (`run-info.json`).

**Risiko für v2b:** Auch die BatchNorm-γ hätte schon als Temperatur wirken können und hat sich in einer
Epoche nur um ≈ 9 % bewegt. Bei Basis-LR 6e-4 kann sich log s_h mit Adam um höchstens ≈ 6e-4 pro
Schritt ändern (≈ 2 über 3600 Schritte bei konsistentem Vorzeichen). Bleibt s_h nahe 1, ist das ein
Befund („das Modell will nicht schärfer“), kein Fehler; als Folgeschritt wäre dann ein höherer
Startwert (`--mem_score_scale_init 4`) oder eine eigene LR für die Skala zu prüfen.

## Laufen lassen (erst, wenn die GPU frei ist)

```bash
cd ../AngryAnt-v2-sharpness          # dieser Worktree; data/ und .venv/ sind Symlinks auf ../AngryAnt
.venv/bin/python -m pytest -q tests  # 35 Tests inkl. GPU-Fälle (25 ohne GPU)
.venv/bin/python scripts/run_suite.py v2          # B-v2a/B-v2b × Seeds 0/1, 1 Epoche, ≈ 4 × 28 min
.venv/bin/python scripts/make_report.py v2        # vergleicht mit runs/ep1/{A,B,C}-* (gleiches Budget)
```

`runs/ep1` aus `main` muss dafür im Worktree liegen (CSV/JSON sind versioniert; für die
Zugriffs-Grafiken die `.npz`/`.npy` aus `../AngryAnt/runs/ep1` dazukopieren oder verlinken).

## Bewertung (vor den Läufen festgelegt)

Gleiche Kriterien und Schwellen wie in Stufe 1 (REPORT.md), je Variante gegen A und C aus `runs/ep1`:
G = (PPL_A − PPL_v)/(PPL_A − PPL_C) ≥ 0,5 und Unterschied zu A > 2·s, Tabelle gesund. Zusätzlich
berichtet: paarweise Differenz zu B bei gleichem Seed, effektiv gemischte Einträge je Kopf und die
gelernte Skala. Eine Variante gilt nur dann als „schärfer“, wenn die effektiv gemischten Einträge
deutlich unter den 30,5 von v1 liegen.
