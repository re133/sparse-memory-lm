### Einzelläufe (ep3)

| Lauf | Params gesamt | ohne Emb. | aktiv/Token (ohne Emb.) | MACs/Token | Val-PPL | Test-PPL | Val-PPL (Wort) | Train tok/s | Decode b=1 tok/s | Prefill tok/s | VRAM Train (GiB) | Trainzeit | Key-Nutzung Val | Top-1 %-Anteil | KL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A-s0 | 40.6 M | 21.2 M | 21.2 M | 45.3 M | 24.81 | 24.99 | 38.49 | 91,659 | 219 | 383,220 | 7.89 | 64 min | – | – | – |
| A-s1 | 40.6 M | 21.2 M | 21.2 M | 45.3 M | 24.65 | 24.84 | 38.21 | 91,478 | 217 | 384,725 | 7.89 | 64 min | – | – | – |
| B-s0 | 141.3 M | 121.9 M | 21.3 M | 45.4 M | 23.80 | 24.02 | 36.72 | 71,935 | 203 | 264,825 | 9.47 | 82 min | 100.0 % | 11.0 % | 0.63 |
| B-s1 | 141.3 M | 121.9 M | 21.3 M | 45.4 M | 23.47 | 23.76 | 36.13 | 71,640 | 206 | 265,278 | 9.47 | 82 min | 100.0 % | 11.0 % | 0.59 |
| C-s0 | 161.3 M | 122.7 M | 122.7 M | 173.9 M | 19.15 | 19.51 | 28.68 | 34,607 | 164 | 131,352 | 7.91 | 170 min | – | – | – |

### Auswertung gegen die Erfolgskriterien (ep3)

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
