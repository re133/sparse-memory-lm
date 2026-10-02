### Einzelläufe (v2b_ep3)

| Lauf | Params gesamt | ohne Emb. | aktiv/Token (ohne Emb.) | MACs/Token | Val-PPL | Test-PPL | Val-PPL (Wort) | Train tok/s | Decode b=1 tok/s | Prefill tok/s | VRAM Train (GiB) | Trainzeit | Key-Nutzung Val | Top-1 %-Anteil | KL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A-s0 | 40.6 M | 21.2 M | 21.2 M | 45.3 M | 24.81 | 24.99 | 38.49 | 91,659 | 219 | 383,220 | 7.89 | 64 min | – | – | – |
| A-s1 | 40.6 M | 21.2 M | 21.2 M | 45.3 M | 24.65 | 24.84 | 38.21 | 91,478 | 217 | 384,725 | 7.89 | 64 min | – | – | – |
| B-s0 | 141.3 M | 121.9 M | 21.3 M | 45.4 M | 23.80 | 24.02 | 36.72 | 71,935 | 203 | 264,825 | 9.47 | 82 min | 100.0 % | 11.0 % | 0.63 |
| B-s1 | 141.3 M | 121.9 M | 21.3 M | 45.4 M | 23.47 | 23.76 | 36.13 | 71,640 | 206 | 265,278 | 9.47 | 82 min | 100.0 % | 11.0 % | 0.59 |
| B-v2b-s0 | 141.3 M | 121.9 M | 21.3 M | 45.4 M | 23.74 | 23.99 | 36.61 | 71,857 | 205 | 264,525 | 9.47 | 82 min | 100.0 % | 12.1 % | 0.70 |
| B-v2b-s1 | 141.3 M | 121.9 M | 21.3 M | 45.4 M | 23.44 | 23.74 | 36.08 | 71,757 | 203 | 263,124 | 9.47 | 82 min | 100.0 % | 11.3 % | 0.62 |
| C-s0 | 161.3 M | 122.7 M | 122.7 M | 173.9 M | 19.15 | 19.51 | 28.68 | 34,607 | 164 | 131,352 | 7.91 | 170 min | – | – | – |

### Auswertung gegen die Erfolgskriterien (v2b_ep3, B)

| Größe | Wert |
|---|---|
| PPL A (Mittel; Seeds) | 24.73 (24.81, 24.65) |
| PPL B (Mittel; Seeds) | 23.63 (23.80, 23.47) |
| PPL C | 19.15 |
| Seed-Spanne s | 0.334 (Schwelle 2·s = 0.669) |
| PPL_A − PPL_B | +1.097 (über 2·s) |
| Lückenschluss G = (A−B)/(A−C) | 0.20 |
| Key-Nutzung Val (min. über Seeds) | 100.0 % |
| Anteil Top-1 % der Einträge (max. über Seeds) | 11.0 % |
| Tabelle gesund (≥ 60 % und Top-1 % ≤ 50 %) | ja |
| **Urteil** | **unklar (B besser als A, aber knapp: G < 0,5)** |

### Auswertung gegen die Erfolgskriterien (v2b_ep3, B-v2b)

| Größe | Wert |
|---|---|
| PPL A (Mittel; Seeds) | 24.73 (24.81, 24.65) |
| PPL B-v2b (Mittel; Seeds) | 23.59 (23.74, 23.44) |
| PPL C | 19.15 |
| Seed-Spanne s | 0.305 (Schwelle 2·s = 0.610) |
| PPL_A − PPL_B-v2b | +1.141 (über 2·s) |
| Lückenschluss G = (A−B-v2b)/(A−C) | 0.20 |
| Key-Nutzung Val (min. über Seeds) | 100.0 % |
| Anteil Top-1 % der Einträge (max. über Seeds) | 12.1 % |
| Tabelle gesund (≥ 60 % und Top-1 % ≤ 50 %) | ja |
| **Urteil** | **unklar (B-v2b besser als A, aber knapp: G < 0,5)** |

### Paarweiser Vergleich B-v2b gegen B (gleicher Seed = gleiche Startgewichte, gleiche Daten)

| Seed | Val-PPL B | Val-PPL B-v2b | Δ Val | Δ Val % | Test-PPL B | Test-PPL B-v2b | Δ Test % | eff. Einträge B | eff. Einträge B-v2b | Top-1-Gewicht B | Top-1-Gewicht B-v2b | Skala je Kopf |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 23.80 | 23.74 | -0.059 | -0.25 % | 24.02 | 23.99 | -0.11 % | 28.51 | 25.46 | 0.0949 | 0.1361 | 1.74, 1.72, 1.73, 1.73 |
| 1 | 23.47 | 23.44 | -0.030 | -0.13 % | 23.76 | 23.74 | -0.07 % | 28.28 | 25.64 | 0.0973 | 0.1318 | 1.74, 1.74, 1.74, 1.75 |

| Gate | Wert |
|---|---|
| mittlere Differenz B-v2b − B (Val-PPL) | -0.044 |
| s = max. Seed-Spanne (B, B-v2b) | 0.334 → 2·s = 0.669 |
| **B-v2b klar besser als B** (beide Seeds besser und Mittel > 2·s) | **nein** |
| **Fix greift** (eff. Einträge ≤ 0,9 × B bei beiden Seeds) | **nein** |
