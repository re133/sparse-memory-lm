### Einzelläufe (s1b)

| Lauf | Params gesamt | ohne Emb. | aktiv/Token (ohne Emb.) | MACs/Token | Val-PPL | Test-PPL | Val-PPL (Wort) | Train tok/s | Decode b=1 tok/s | Prefill tok/s | VRAM Train (GiB) | Trainzeit | Key-Nutzung Val | Top-1 %-Anteil | KL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A-s0 | 40.6 M | 21.2 M | 21.2 M | 45.3 M | 25.67 | – | 129.19 | 91,563 | 220 | 385,525 | 7.89 | 91 min | – | – | – |
| B-1M-s0 | 444.9 M | 425.6 M | 23.1 M | 47.1 M | 21.80 | – | 101.17 | 33,383 | 175 | 150,398 | 11.70 | 249 min | 100.0 % | 11.8 % | 0.62 |

### Schnelltest Stufe 1b: A gegen B-1M (500 M frische Wikipedia-Tokens, je Seed 0)

| | Val-PPL Wikipedia | Val-PPL WikiText-103 | Train tok/s | VRAM Train (GiB) | Trainzeit |
|---|---|---|---|---|---|
| A | 25.67 | 78.38 | 91,563 | 7.89 | 91 min |
| B-1M | 21.80 | 66.28 | 33,383 | 11.70 | 249 min |

| Kriterium | Wert |
|---|---|
| PPL(B-1M) / PPL(A), Wikipedia-Val | 0.8494 (-15.06 %) |
| WikiText-103-Val (nur berichtet) | -15.44 % |
| **≥ 10 % niedriger (Verhältnis ≤ 0,90)** | **ja → großer Test** |
