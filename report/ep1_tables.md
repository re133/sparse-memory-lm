### Einzelläufe (ep1)

| Lauf | Params gesamt | ohne Emb. | aktiv/Token (ohne Emb.) | MACs/Token | Val-PPL | Test-PPL | Val-PPL (Wort) | Train tok/s | Decode b=1 tok/s | Prefill tok/s | VRAM Train (GiB) | Trainzeit | Key-Nutzung Val | Top-1 %-Anteil | KL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A-s0 | 40.6 M | 21.2 M | 21.2 M | 45.3 M | 35.26 | 35.24 | 57.39 | 91,705 | 217 | 383,407 | 7.89 | 21 min | – | – | – |
| A-s1 | 40.6 M | 21.2 M | 21.2 M | 45.3 M | 34.93 | 34.77 | 56.80 | 91,640 | 218 | 384,460 | 7.89 | 21 min | – | – | – |
| B-s0 | 141.3 M | 121.9 M | 21.3 M | 45.4 M | 34.63 | 34.56 | 56.24 | 72,445 | 199 | 264,882 | 9.47 | 27 min | 99.2 % | 18.4 % | 1.08 |
| B-s1 | 141.3 M | 121.9 M | 21.3 M | 45.4 M | 34.12 | 34.05 | 55.30 | 72,241 | 203 | 264,573 | 9.47 | 27 min | 99.7 % | 15.5 % | 0.91 |
| C-s0 | 161.3 M | 122.7 M | 122.7 M | 173.9 M | 25.80 | 26.02 | 40.25 | 34,468 | 165 | 130,976 | 7.91 | 57 min | – | – | – |

### Auswertung gegen die Erfolgskriterien (ep1)

| Größe | Wert |
|---|---|
| PPL A (Mittel; Seeds) | 35.09 (35.26, 34.93) |
| PPL B (Mittel; Seeds) | 34.38 (34.63, 34.12) |
| PPL C | 25.80 |
| Seed-Spanne s | 0.511 (Schwelle 2·s = 1.022) |
| PPL_A − PPL_B | +0.718 (innerhalb 2·s) |
| Lückenschluss G = (A−B)/(A−C) | 0.08 |
| Key-Nutzung Val (min. über Seeds) | 99.2 % |
| Anteil Top-1 % der Einträge (max. über Seeds) | 18.4 % |
| Tabelle gesund (≥ 60 % und Top-1 % ≤ 50 %) | ja |
| **Urteil** | **lohnt sich nicht (B auf A-Niveau: Unterschied innerhalb 2·s)** |
