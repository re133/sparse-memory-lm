### Einzelläufe (probe)

| Lauf | Params gesamt | ohne Emb. | aktiv/Token (ohne Emb.) | MACs/Token | Val-PPL | Test-PPL | Val-PPL (Wort) | Train tok/s | Decode b=1 tok/s | Prefill tok/s | VRAM Train (GiB) | Trainzeit | Key-Nutzung Val | Top-1 %-Anteil | KL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A-s0 | 40.6 M | 21.2 M | 21.2 M | 45.3 M | 138.92 | 140.84 | 272.81 | 91,542 | 213 | 383,125 | 7.89 | 4 min | – | – | – |
| B-s0 | 141.3 M | 121.9 M | 21.3 M | 45.4 M | 135.68 | 137.64 | 265.59 | 73,043 | 197 | 269,315 | 9.47 | 5 min | 81.8 % | 33.0 % | 2.03 |
| C-s0 | 161.3 M | 122.7 M | 122.7 M | 173.9 M | 107.75 | 110.12 | 204.37 | 34,441 | 165 | 131,093 | 7.91 | 10 min | – | – | – |

### Auswertung gegen die Erfolgskriterien (probe)

| Größe | Wert |
|---|---|
| PPL A (Mittel; Seeds) | 138.92 (138.92) |
| PPL B (Mittel; Seeds) | 135.68 (135.68) |
| PPL C | 107.75 |
| Seed-Spanne s | 0.000 (Schwelle 2·s = 0.000) |
| PPL_A − PPL_B | +3.237 (über 2·s) |
| Lückenschluss G = (A−B)/(A−C) | 0.10 |
| Key-Nutzung Val (min. über Seeds) | 81.8 % |
| Anteil Top-1 % der Einträge (max. über Seeds) | 33.0 % |
| Tabelle gesund (≥ 60 % und Top-1 % ≤ 50 %) | ja |
| **Urteil** | **unklar (B besser als A, aber knapp: G < 0,5)** |

_Nur ein Seed je Modell: das Seed-Rauschen ist hier nicht messbar, das Urteil ist nur vorläufig._
