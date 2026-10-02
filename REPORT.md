# Stufe 1: Verbessert eine Product-Key-Speicherschicht ein kleines Sprachmodell bei gleichem Rechenaufwand?

> **Kurzfassung.** Ja, aber nur wenig. Bei gleichem Rechenaufwand pro Token senkt die Speicherschicht (B)
> die Validierungs-Perplexity gegenüber der Baseline (A) um 2,0 % nach 1 Epoche und um 4,4 % nach 3 Epochen.
> Nach den **vor den Läufen festgelegten** Kriterien heißt das: **1 Epoche „lohnt sich nicht“** (Unterschied
> innerhalb des Seed-Rauschens), **3 Epochen „unklar“** (Unterschied echt, aber B schließt nur 20 % der
> Lücke zum gleich großen dichten Modell C, gefordert waren 50 %). Die Tabelle ist gesund (≈ 100 % Nutzung)
> und wird stark genutzt; ein Implementierungsfehler wurde nicht gefunden. Auffällig ist eine fast flache
> Gewichtung innerhalb der Top-32. Dafür ist der Branch `v2-sharpness` vorbereitet (nicht gelaufen).
> Kosten von B trotz gleicher FLOPs: −21 % Trainings-, −31 % Prefill-Durchsatz, +1,6 GiB VRAM.

Die Erfolgskriterien wurden festgelegt, bevor ein Lauf gestartet wurde (Commit `51176ad`, präzisiert in
`49f1202` vor dem ersten Ergebnis).

## Erfolgskriterien (vor den Läufen festgelegt)

Vorgabe (wörtlich übernommen):

- **lohnt sich:** Key-Nutzung deutlich über 50 % **und** B schließt mindestens die halbe
  Perplexity-Lücke zwischen A und C
- **unklar:** B besser als A, aber knapp → erst Implementierung und Tabellengesundheit prüfen
- **lohnt sich nicht:** B trotz gesunder Tabelle auf A-Niveau

Operationalisierung (so wird es ausgewertet, getrennt für den 1-Epochen- und den 3-Epochen-Lauf):

| Größe | Definition |
|---|---|
| PPL | Token-Perplexity (GPT-2-BPE) auf dem **gesamten** WikiText-103-Validierungsset am Ende des Trainings, nicht überlappende 1024er-Fenster, Modell im Eval-Modus. A und B: Mittel über 2 Init-Seeds, C: 1 Seed. |
| Lückenschluss G | G = (PPL_A − PPL_B) / (PPL_A − PPL_C). „Mindestens die halbe Lücke“ heißt G ≥ 0,5. |
| Key-Nutzung | Anteil der 262.144 Einträge, die auf dem Validierungsset (≈ 247 k Tokens, ≈ 31,7 M Lesezugriffe) mindestens einmal gelesen werden (Definition „memory usage“ nach Lample et al. 2019). „Deutlich über 50 %“ lege ich als **≥ 60 % bei beiden B-Seeds** fest (meine Interpretation). |
| Tabellengesundheit | Nutzung wie oben, dazu KL(Zugriffsgewichte ‖ Gleichverteilung) und der Anteil der Zugriffe, der auf das meistgelesene 1 % der Einträge fällt. Eine Tabelle mit ≥ 60 % Nutzung, bei der aber > 50 % aller Zugriffe auf 1 % der Einträge fallen, gilt **nicht** als gesund. |
| Zufallsrauschen | Seed-Spanne s = max(\|PPL_A,s0 − PPL_A,s1\|, \|PPL_B,s0 − PPL_B,s1\|). Ein Unterschied A↔B gilt erst als echt, wenn \|PPL_A − PPL_B\| > 2·s. |
| „knapp“ (→ unklar) | B besser als A um mehr als 2·s, aber G < 0,5. |
| „auf A-Niveau“ (→ lohnt sich nicht) | \|PPL_A − PPL_B\| ≤ 2·s, oder B schlechter als A. |

Fälle, die die Vorgabe nicht abdeckt: Ist die Tabelle **nicht** gesund (Nutzung < 60 % oder starke
Konzentration), ist das Ergebnis **nicht schlüssig**, unabhängig von der Perplexity. Dann wird zuerst
die Speicherschicht repariert (Query-Normalisierung, Lernraten), bevor ein Urteil gefällt wird.

## Versuchsaufbau

**Daten.** WikiText-103 (raw, HF `Salesforce/wikitext`), GPT-2-BPE via `tiktoken`: 117,98 M Trainings-,
247 k Validierungs-, 283 k Test-Tokens. Training auf nicht überlappenden 1025er-Fenstern, deren
Reihenfolge pro Epoche nur vom Daten-Seed (1234) abhängt. Alle Modelle sehen also exakt denselben
Token-Strom in derselben Reihenfolge. 1 Epoche = 3600 Schritte à 32 × 1024 = 32.768 Tokens.

**Modelle** (Llama-Stil: RMSNorm, RoPE, SwiGLU, keine Biases, Ein-/Ausgabe-Embedding geteilt, Kontext 1024):

| | Aufbau | Params ohne Emb. | Params gesamt | aktiv/Token ohne Emb. | MACs/Token (Forward) |
|---|---|---|---|---|---|
| A | d=384, 12 Layer, 6 Köpfe, SwiGLU 1024 | 21,24 M | 40,56 M | 21,24 M | 45,27 M |
| B | wie A, FFN von Layer 7 (Index 6) → PKM | 121,94 M | 141,26 M | 21,33 M | 45,35 M |
| C | d=768, 16 Layer, 12 Köpfe, SwiGLU 2304 | 122,71 M | 161,34 M | 122,71 M | 173,90 M |

C ist auf B's Parameter **ohne Embeddings** abgeglichen (wie A „ohne Embeddings“ spezifiziert ist).
„Aktiv/Token“ für B = alle dichten Parameter außer der Wertetabelle (inkl. aller Sub-Keys, die
vollständig gescannt werden) + die 4 × 32 × 384 tatsächlich gelesenen Werte. MACs enthalten LM-Kopf
und Attention (mittlerer kausaler Kontext 512).

**Speicherschicht (B).** 512² = 262.144 Einträge, 4 Köpfe, Top-k 32, Key-Dim 256 (2 × 128), Werte-Dim 384,
geteilte Wertetabelle (100,7 M Parameter). Aufbau nach der Meta-Referenzimplementierung
(`facebookresearch/memory`, `lingua/product_key/memory.py`): eigene Sub-Keys je Kopf, exakte Top-k über
das Produkt (Top-k je Hälfte, dann k×k-Kandidaten), Softmax über die k Scores je Kopf, Köpfe summiert,
swilu-Ausgang `W2(m(x) ⊙ silu(W1 x))` („Memory+“), Initialisierung wie dort. Zusätzlich **BatchNorm auf
der Query** (Lample et al. 2019, Abschn. 4.5: hebt die Nutzung bei 1 M Einträgen von 25,8 % auf 80,3 %;
PEER nutzt es ebenso). Im Training sieht BatchNorm Statistiken über die Batch (also auch spätere Tokens);
alle Auswertungen laufen im Eval-Modus mit festen Statistiken, ein Unit-Test prüft die Kausalität dort.
Rechenaufwand des Ersatzes: entferntes FFN 1,18 M MACs/Token, Speicherschicht 1,26 M (Query 0,39 M,
Sub-Key-Scores 0,52 M, Werte lesen 0,05 M, swilu 0,30 M).

**Optimierung** (identisch für A/B/C): AdamW (β = 0,9/0,95, ε = 1e-8, Weight Decay 0,1 auf Matrizen),
Spitzen-LR 6e-4, linearer Warmup über 5 % der Schritte, Cosine auf 10 %, Gradient-Clipping 1,0, bf16-Autocast
mit fp32-Gewichten. **Ausnahme Speicherwerte** (laut Papern): LR 1e-3 absolut (Lample et al.: „higher Adam
learning rate of 10⁻³“ für die sparse aktualisierten Werte; Meta: `value_fixed_lr=0.001`), gleicher
Verlaufs-Multiplikator, kein Weight Decay, eigenes Clipping (wie Meta `train.py`). Verhältnis Werte/Rest
also 1,7× statt 4× bei Lample. Mikro-Batch 8 (A, B) bzw. 4 (C) mit Gradient-Akkumulation auf 32 Sequenzen.

**Seeds.** Daten-Seed 1234 für alle. Init-Seeds 0 und 1 für A und B, 0 für C. GPU-Kernels (Atomics in
`embedding_bag`-Backward, Flash-Attention-Backward) sind nicht bitgenau deterministisch.

**Messungen.** Validierungs-PPL alle 4 M (1 Epoche) bzw. 8 M Tokens (3 Epochen) über das ganze Val-Set,
dazu der mittlere Train-Loss im selben Intervall. Tokens/s im Training ohne Evaluierungszeit (Median über
10-Schritt-Fenster). Inferenz: (a) Batch-1-Decoding mit KV-Cache, 128 Prompt- + 256 neue Tokens,
gierig; (b) Batched-Forward 16 × 1024 („Prefill“). Spitzen-VRAM = `torch.cuda.max_memory_allocated`
während des Trainings (ohne Evaluierung). Für B: Key-Nutzung, KL und Konzentration auf dem Val-Set bei
jeder Evaluierung, Nutzung im Training je Intervall, Zugriffshistogramm über das gesamte Training und
über das Val-Set, sowie eine Index-Stichprobe (die ersten 65.536 Val-Tokens in Textreihenfolge:
Indizes [Token, Kopf, k] als int32, rohe Scores, Token-IDs) in `runs/*/B-*/mem_index_sample.npz`.

**Unit-Tests** (`tests/test_pkm.py`, CPU und GPU, fp64): Product-Key-Top-k = Brute-Force-Top-k über alle
n² Keys (Scores und Index-Mengen exakt); Gradient der Wertetabelle ist genau auf den gelesenen Zeilen
≠ 0 (beide Implementierungen); beide Wertelese-Implementierungen stimmen in Ausgabe und Gradienten
überein; Kausalität im Eval-Modus (mit und ohne Speicher); KV-Cache-Decoding = voller Forward.

## Ergebnisse

### Probelauf: 20 M Tokens, je 1 Seed (Pipeline-Check, nicht für das Urteil)

| Lauf | Val-PPL | Test-PPL | Val-PPL (Wort) | Train tok/s | Decode b=1 tok/s | Prefill tok/s | VRAM Train | Key-Nutzung Val | Top-1 %-Anteil | KL |
|---|---|---|---|---|---|---|---|---|---|---|
| A-s0 | 138,92 | 140,84 | 272,8 | 91.542 | 213 | 383.125 | 7,89 GiB | – | – | – |
| B-s0 | 135,68 | 137,64 | 265,6 | 73.043 | 197 | 269.315 | 9,47 GiB | 81,8 % | 33,0 % | 2,03 |
| C-s0 | 107,75 | 110,12 | 204,4 | 34.441 | 165 | 131.093 | 7,91 GiB | – | – | – |

![Val-PPL Probelauf](report/probe_val_ppl.png)

- B liegt 2,3 % unter A (−3,2 PPL), C 22 % darunter. Lückenschluss G = 0,10. Mit nur einem Seed ist das
  Rauschen nicht messbar; der Probelauf diente nur dazu, die Pipeline zu prüfen.
- **Die Key-Nutzung bricht am Anfang ein und erholt sich dann.** Nach der Initialisierung werden 99 %
  der Einträge gelesen, nach 2 M Tokens nur noch 9 % (85 % aller Zugriffe auf 1 % der Einträge),
  danach steigt die Nutzung stetig auf 82 % bei 20 M Tokens (Top-1 %-Anteil 33 %, weiter fallend).
  Die Key-Normen bleiben dabei gleichmäßig (max/min ≈ 1,3) und fast alle 512 Sub-Keys je Hälfte werden
  gelesen. Der Einbruch kommt also nicht von „Hub-Keys“ mit großer Norm, sondern daher, dass die Queries
  früh im Training wenig divers sind (Residual-Stream noch niedrigrangig) und sich nur wenige
  Kombinationen der beiden Hälften durchsetzen. Für Stufe 3 heißt das: Die Zugriffsverteilung hängt
  stark vom Trainingsstand ab, Messungen an früh gestoppten Modellen sind nicht repräsentativ.

![Tabellengesundheit Probelauf](report/probe_memory_health.png)

- Git: A-s0 lief auf `51176ad`, B-s0 und C-s0 auf `859408d`. Dazwischen änderten sich nur `REPORT.md`
  und das Auswerte-Skript, nicht der Trainingscode. Das `dirty: true` in deren `run-info.json` kommt
  allein vom damals noch nicht versionierten Ordner `runs/` (danach behoben: Ausgaben zählen nicht mehr).

### 1 Epoche: 118 M Tokens (A, B je 2 Seeds; C 1 Seed)

| Lauf | Val-PPL | Test-PPL | Val-PPL (Wort) | Train tok/s | Decode b=1 tok/s | Prefill tok/s | VRAM Train | Trainzeit | Key-Nutzung Val | Top-1 %-Anteil | KL |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A-s0 | 35,26 | 35,24 | 57,4 | 91.705 | 217 | 383.407 | 7,89 GiB | 21 min | – | – | – |
| A-s1 | 34,93 | 34,77 | 56,8 | 91.640 | 218 | 384.460 | 7,89 GiB | 21 min | – | – | – |
| B-s0 | 34,63 | 34,56 | 56,2 | 72.445 | 199 | 264.882 | 9,47 GiB | 27 min | 99,2 % | 18,4 % | 1,08 |
| B-s1 | 34,12 | 34,05 | 55,3 | 72.241 | 203 | 264.573 | 9,47 GiB | 27 min | 99,7 % | 15,5 % | 0,91 |
| C-s0 | 25,80 | 26,02 | 40,3 | 34.468 | 165 | 130.976 | 7,91 GiB | 57 min | – | – | – |

**Auswertung gegen die vorab festgelegten Kriterien:**

| Größe | Wert |
|---|---|
| PPL A (Mittel; Seeds) | 35,09 (35,26 / 34,93) |
| PPL B (Mittel; Seeds) | 34,38 (34,63 / 34,12) |
| PPL C | 25,80 |
| Seed-Spanne s | 0,51 (aus B; A: 0,32) → Schwelle 2·s = 1,02 |
| PPL_A − PPL_B | +0,72 (**innerhalb** 2·s) |
| Lückenschluss G | **0,08** |
| Key-Nutzung (min.) / Top-1 %-Anteil (max.) | 99,2 % / 18,4 % → **Tabelle gesund** |
| **Urteil** | **lohnt sich nicht** (B auf A-Niveau, trotz gesunder Tabelle) |

![Val-PPL 1 Epoche](report/ep1_val_ppl.png)

![Abstand zu A, 1 Epoche](report/ep1_relative_to_A.png)

**Was die Zahlen sagen, ohne Schönfärberei:**

- B ist in allen vier A/B-Paarungen besser als A, und das gleichmäßig ab etwa 30 M Tokens um ≈ 2 %
  (Val wie Test). Das spricht für einen echten, aber kleinen Effekt. Nach dem vorab festgelegten
  Kriterium reicht er nicht: 0,72 PPL liegen unter 2·s = 1,02, und mit zwei Seeds je Modell ist auch
  „B schlägt A in allen Paarungen“ nicht belastbar (bei Zufall träte das in 1 von 6 Fällen auf).
- **Selbst wenn der Effekt echt ist, ist er viel zu klein.** B schließt 8 % der Lücke zu C, gefordert
  waren 50 %. Der Vorsprung wächst über die Epoche auch nicht, er bleibt bei ≈ 2 % (siehe Grafik).
  Die 100 M zusätzlichen Parameter von B bringen also bei weitem nicht, was dieselbe Zahl dichter
  Parameter in C bringt (−26,5 % PPL), allerdings bei 3,8× so vielen MACs pro Token für C.
- **C ist bei 1 Epoche deutlich untertrainiert** (≈ 1 Token je Parameter statt ≈ 20). Die Lücke A↔C
  wäre bei einem ausgereizten C eher noch größer; der Vergleich ist für B damit eher günstig.
- **Kosten von B:** gleiche FLOPs wie A, aber 21 % weniger Trainingsdurchsatz, 31 % weniger
  Prefill-Durchsatz, 7 % langsameres Batch-1-Decoding und 1,6 GiB mehr VRAM (Wertetabelle mit
  Gradient und Adam-Zuständen: 100,7 M × 16 Byte).

**Tabellengesundheit und Implementierung** (Pflichtprüfung, bevor ein Urteil zählt):

![Tabellengesundheit 1 Epoche](report/ep1_memory_health.png)

- Nutzung 99,2 / 99,7 %, KL 1,08 / 0,91, das meistgelesene 1 % der Einträge bekommt 18 / 16 % der
  Zugriffe, nur 0,3–0,8 % der Einträge werden auf dem Val-Set nie gelesen. Im Training wurde jeder
  Eintrag gelesen (seltenster ≈ 2.400×). Der Einbruch am Anfang (s. Probelauf) erholt sich nach
  ≈ 20 M Tokens vollständig.
- **Die Speicherschicht trägt tatsächlich etwas bei** (`scripts/diagnose_memory.py`, erste 32.768
  Val-Tokens, `runs/ep1/B-*/diagnostics.json`): Setzt man ihre Ausgabe auf null, steigt die PPL von 31,3
  auf 35,5 (s0) bzw. 30,6 auf 35,7 (s1). Liest man statt der gefundenen zufällige Einträge, steigt sie
  auf 36,0 bzw. 36,5. Es kommt also darauf an, *welche* Einträge die Suche findet. Die Ausgabenorm der
  Schicht (2,5–2,8) liegt über der der benachbarten dichten FFNs (1,2–2,6). Die 4 Köpfe lesen pro Token
  fast immer 128 verschiedene Einträge.
- Die Unit-Tests (exakte Top-k, Gradient nur in gelesene Werte, Kausalität, KV-Cache) sind grün.
  Einen Implementierungsfehler habe ich nicht gefunden.
- **Auffälligkeit: Die Gewichtung innerhalb der Top-32 ist fast flach.** Effektiv mischt jeder Kopf
  30,5 von 32 Einträgen (exp(Entropie)), das Top-1-Gewicht liegt bei 0,066 (gleichverteilt: 0,031).
  Die Score-Skala ist seit der Initialisierung kaum gewachsen (Key-Norm 0,41 → 0,48, BatchNorm-γ ≈ 1,09).
  Die Schicht ruft also nicht gezielt wenige Einträge ab, sondern mittelt grob über 128. Das passt zum
  Befund „ersetzt ein FFN und etwas mehr, aber nicht viel mehr“: Ohne Speicher fehlt dem Modell eine
  ganze Schicht (+14–16 % PPL), mit Speicher ist es nur 2 % besser als A.

**Zugriffsmuster (für Stufe 3, `report/ep1_access_stats.json`):**

![Zugriffsverteilung 1 Epoche](report/ep1_access_distribution.png)

| | B-s0 | B-s1 |
|---|---|---|
| Anteil der Zugriffe auf die Top 1 / 10 / 20 / 50 % (Val) | 18 / 55 / 71 / 91 % | 16 / 50 / 66 / 89 % |
| Val-Zugriffe, die die Top 20 % aus dem **Training** abdecken | 68 % | 63 % |
| Rangkorrelation der Zugriffszahlen Training ↔ Val (Spearman) | 0,89 | 0,89 |
| Zugriffe, die schon in den letzten 1 / 16 / 256 Tokens gelesen wurden | 0,7 / 11,7 / 52,5 % | 0,9 / 11,5 / 50,2 % |

Die Verteilung ist deutlich schief, aber kein extremer Zipf: Ein Cache mit den 20 % meistgelesenen
Einträgen (im Training ermittelt; hier 52 k Einträge × 384 × 2 Byte ≈ 40 MB in bf16) fängt ≈ 65 % der
Lesezugriffe auf dem Val-Set ab. Aufeinanderfolgende Tokens lesen
fast disjunkte Einträge (< 1 % Wiederholung zum direkten Vorgänger); über ein Fenster von 256 Tokens
wiederholt sich aber die Hälfte. Für die SSD-Auslagerung heißt das: ≈ 35 % der 128 Lesezugriffe pro
Token müssten auch mit einem Hot-Set-Cache noch zufällig von der SSD kommen.

![Train- vs. Val-Loss 1 Epoche](report/ep1_train_vs_val.png)

Bei 1 Epoche gibt es erwartungsgemäß kein Auswendiglernen (jedes Fenster wird genau einmal gesehen);
dass Train über Val liegt, ist der nachlaufende Intervall-Mittelwert, s. Grenzen.

### 3 Epochen: 354 M Tokens, eigener Cosine-Plan (A, B je 2 Seeds; C 1 Seed)

Eigener Lauf mit Warmup (540 Schritte) und Cosine über alle 10.800 Schritte; die Zwischenstände bei
118 M Tokens sind daher **nicht** mit dem Ende des 1-Epochen-Laufs vergleichbar (dort war die LR schon
abgeklungen). Jede Epoche hat ihre eigene, vom Daten-Seed festgelegte Reihenfolge.

| Lauf | Val-PPL | Test-PPL | Val-PPL (Wort) | Train tok/s | Decode b=1 tok/s | Prefill tok/s | VRAM Train | Trainzeit | Key-Nutzung Val | Top-1 %-Anteil | KL |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A-s0 | 24,81 | 24,99 | 38,5 | 91.659 | 219 | 383.220 | 7,89 GiB | 64 min | – | – | – |
| A-s1 | 24,65 | 24,84 | 38,2 | 91.478 | 217 | 384.725 | 7,89 GiB | 64 min | – | – | – |
| B-s0 | 23,80 | 24,02 | 36,7 | 71.935 | 203 | 264.825 | 9,47 GiB | 82 min | 99,97 % | 11,0 % | 0,63 |
| B-s1 | 23,47 | 23,76 | 36,1 | 71.640 | 206 | 265.278 | 9,47 GiB | 82 min | 99,99 % | 11,0 % | 0,59 |
| C-s0 | 19,15 | 19,51 | 28,7 | 34.607 | 164 | 131.352 | 7,91 GiB | 170 min | – | – | – |

**Auswertung gegen die vorab festgelegten Kriterien:**

| Größe | Wert |
|---|---|
| PPL A (Mittel; Seeds) | 24,73 (24,81 / 24,65) |
| PPL B (Mittel; Seeds) | 23,63 (23,80 / 23,47) |
| PPL C | 19,15 |
| Seed-Spanne s | 0,33 (aus B; A: 0,16) → Schwelle 2·s = 0,67 |
| PPL_A − PPL_B | +1,10 (**über** 2·s) |
| Lückenschluss G | **0,20** |
| Key-Nutzung (min.) / Top-1 %-Anteil (max.) | 99,97 % / 11,0 % → **Tabelle gesund** |
| **Urteil** | **unklar** (B echt besser als A, aber knapp: G < 0,5) |

Auf dem Test-Set ergibt sich dasselbe Bild (A 24,92, B 23,89, C 19,51 → −4,1 %, G = 0,19).

![Val-PPL 3 Epochen](report/ep3_val_ppl.png)

![Abstand zu A, 3 Epochen](report/ep3_relative_to_A.png)

**Was die Zahlen sagen:**

- **Mit mehr Daten wächst der Vorsprung von B, bleibt aber klein.** Relativ zu A: −0,8 % bei 40 M Tokens,
  −2,0 % bei 80 M, −2,9 % bei 120 M, −4,0 % bei 176 M, dann ab ≈ 240 M Tokens konstant bei −4,4 %
  (beide Seeds, siehe Grafik). Der Lückenschluss G steigt entsprechend auf ≈ 0,20 und bleibt dort.
  Beide B-Seeds liegen jetzt klar unter beiden A-Seeds.
- **Für „lohnt sich“ fehlt mehr als die Hälfte.** C ist auch nach 3 Epochen 22,6 % besser als A
  (bei 3,8× MACs/Token und weiter untertrainiert: ≈ 3 Tokens je Parameter). B holt davon ein Fünftel.
- **Auswendiglernen** (Val-Loss minus Train-Loss am Ende, nats): A 0,014 / 0,015, B 0,035 / 0,043,
  C 0,119. In Epoche 2 und 3 fällt der Train-Loss an den Epochengrenzen sichtbar ab. B lernt etwas
  mehr auswendig als A, viel weniger als C; der Val-Loss von B fällt trotzdem bis zum Ende weiter.

![Train- vs. Val-Loss 3 Epochen](report/ep3_train_vs_val.png)

**Tabellengesundheit und Implementierung** (Pflichtprüfung bei „unklar“, `runs/ep3/B-*/diagnostics.json`,
erste 32.768 Val-Tokens, GPU):

- Nutzung 99,97 / 99,99 %, KL 0,63 / 0,59, das meistgelesene 1 % bekommt 11 % der Zugriffe; im Training
  wurde jeder Eintrag gelesen (seltenster 961× bzw. 1.632×, Median ≈ 100 k×). Gesünder als nach 1 Epoche.
- **Das Modell stützt sich nach 3 Epochen viel stärker auf den Speicher:** Ausgabe auf null → PPL
  21,8 → 31,3 (s0) bzw. 21,4 → 32,7 (s1), also +44 / +53 % (nach 1 Epoche: +14 / +16 %). Zufällige statt
  gefundener Einträge → 34,3 bzw. 37,3. Ohne Speicher ist B damit deutlich schlechter als A. Die
  Ausgabenorm der Schicht (7,4 / 8,3) ist die größte aller FFN-Positionen in der Mitte (Nachbarn 2,8–5,2).
- **Die Gewichtung innerhalb der Top-32 wird mit dem Training etwas schärfer, bleibt aber flach:**
  effektiv 28,6 / 28,3 von 32 Einträgen je Kopf (nach 1 Epoche 30,5), Top-1-Gewicht 0,094 / 0,097.
  Key-Norm 0,41 → 0,59 / 0,61, BatchNorm-γ 1,0 → 1,33 / 1,37. Die Skala wächst also, aber langsam.
- Unit-Tests grün; kein Implementierungsfehler gefunden. Die Tabelle funktioniert und wird genutzt;
  offen ist, warum sie so wenig über eine zusätzliche dichte Schicht hinaus bringt (siehe Einordnung).

![Tabellengesundheit 3 Epochen](report/ep3_memory_health.png)

**Zugriffsmuster (für Stufe 3, `report/ep3_access_stats.json`):**

| | B-s0 | B-s1 |
|---|---|---|
| Anteil der Zugriffe auf die Top 1 / 10 / 20 / 50 % (Val) | 11 / 40 / 56 / 83 % | 11 / 39 / 55 / 82 % |
| Val-Zugriffe, die die Top 20 % aus dem **Training** abdecken | 54 % | 52 % |
| Rangkorrelation der Zugriffszahlen Training ↔ Val (Spearman) | 0,90 | 0,90 |
| Zugriffe, die schon in den letzten 1 / 16 / 256 Tokens gelesen wurden | 0,9 / 9,4 / 44,8 % | 0,7 / 8,9 / 42,8 % |

Mit längerem Training verteilen sich die Zugriffe **gleichmäßiger**: Ein Hot-Set-Cache aus 20 % der
Einträge fängt nach 3 Epochen nur noch ≈ 53 % der Lesezugriffe ab (nach 1 Epoche ≈ 65 %). Für die
SSD-Auslagerung ist das eine schlechte Nachricht: Je besser die Tabelle genutzt wird, desto weniger hilft
Caching. Die Rangfolge „heißer“ Einträge ist zwischen Training und Val aber stabil (Spearman 0,90).

![Zugriffsverteilung 3 Epochen](report/ep3_access_distribution.png)

## Einordnung

**Antwort auf die Frage dieser Stufe.** Eine Product-Key-Speicherschicht verbessert das kleine Modell bei
gleichem Rechenaufwand pro Token **messbar, aber wenig**: −2,0 % PPL nach 118 M Tokens (nicht von
Seed-Rauschen zu trennen), −4,4 % nach 354 M Tokens (echt). Nach den vorab festgelegten Kriterien reicht
das in keinem der beiden Läufe für „lohnt sich“: B schließt 8 % bzw. 20 % der Lücke zum gleich großen
dichten Modell, gefordert waren 50 %. Ich rede das nicht schön: In dieser Konfiguration sind 100 M
zusätzliche Parameter in der Tabelle ungefähr so viel wert wie ein Fünftel der Wirkung derselben Zahl
dichter Parameter.

**Was dagegen spricht, dass es nur ein Bug ist:** Die Tests sind grün, die Tabelle ist gesund und wird
breit genutzt, und das Modell stützt sich nach 3 Epochen stark auf sie (+44–53 % PPL ohne Speicher;
zufällige Einträge sind noch schlechter). Der Speicher funktioniert also. Er bringt nur wenig mehr als
das FFN, das er ersetzt.

**Mögliche Gründe** (nach meiner Einschätzung der Wahrscheinlichkeit geordnet, keiner davon ist hier
belegt):

1. **Flache Gewichtung innerhalb der Top-k.** Jeder Kopf mischt 28–30 von 32 Einträgen fast gleich
   stark; die Schicht ruft kaum gezielt ab. Kandidaten: Die Query-BatchNorm fixiert die Query-Skala, und
   Weight Decay zieht die Keys klein. Dafür ist `v2-sharpness` vorbereitet (v2a: kein Weight Decay auf
   den Keys; v2b: zusätzlich lernbare Temperatur), siehe `V2_SHARPNESS.md` auf dem Branch.
2. **Datenregime.** Der Vorsprung wächst mit den Daten (2 % → 4,4 %), sättigt aber ab ≈ 240 M Tokens,
   während das Training dieselben 118 M Tokens wiederholt. Die Paper zeigen den Nutzen bei Hunderten
   Milliarden Tokens und vor allem auf faktenlastigen QA-Aufgaben; Perplexity auf WikiText-103 misst
   „Wissen abrufen“ nur indirekt.
3. **Lernrate der Werte.** 1e-3 bei Basis-LR 6e-4 ist ein Verhältnis von 1,7× (Lample: 4×).
4. **Nur eine Speicherschicht.** Meta (Memory+) nutzt drei Schichten mit geteilter Tabelle.
5. **Kleiner Rechenkern.** Mit d = 384 und 21 M dichten Parametern ist auch die Query-Netz-Kapazität
   klein.

**Kosten.** Gleiche FLOPs, aber B trainiert 21 % langsamer, ist im Prefill 31 % und im Batch-1-Decoding
≈ 6 % langsamer und braucht 1,6 GiB mehr VRAM. Das liegt an den unregelmäßigen Speicherzugriffen
(Gather/Scatter, Adam über 100 M Werte), nicht an Rechenarbeit, und ist genau der Teil, um den es in
den späteren Stufen geht.

**Für das Gesamtprojekt.** Stufe 1 widerlegt den Ansatz nicht, liefert aber noch keinen Grund, ihn zu
skalieren. Vor Stufe 2 würde ich die vorbereiteten v2-Varianten laufen lassen (≈ 2 h). Bringen sie die
Gewichtung nicht deutlich schärfer und B nicht deutlich weiter, wären mehr Daten (größerer Korpus statt
Wiederholung), mehrere Speicherschichten und eine höhere Werte-LR die nächsten Hebel. Für Stufe 3
wichtig: Die Zugriffe werden mit besserem Training gleichmäßiger verteilt, Hot-Set-Caching hilft dann
weniger (20 % der Einträge → 53 % der Zugriffe).

## Grenzen dieser Untersuchung

- **C ist bei 1 Epoche deutlich untertrainiert.** 118 M Tokens auf 122,7 M Parameter (ohne Embeddings)
  sind ≈ 1 Token pro Parameter; compute-optimal (Chinchilla) wären ≈ 20. Der Abstand A↔C und damit die
  Lücke, die B schließen soll, ist bei 1 Epoche kleiner als bei einem ausgereizten C. Der 3-Epochen-Lauf
  mildert das nur etwas (≈ 3 Tokens/Parameter), wiederholt dafür aber die Daten.
- **Seed-Rauschen ist grob geschätzt.** Mit 2 Seeds ist s eine einzelne Differenz, keine Streuung. Das
  Kriterium bleibt wie festgelegt, ist aber eher optimistisch, was die Trennschärfe angeht.
- **Der Train-Loss in den Kurven hinkt hinterher.** Er ist der Mittelwert über das jeweilige
  Eval-Intervall, in dem das Modell noch besser wird; Train > Val früh im Training ist ein Artefakt, kein
  Fehler. Aussagekräftig für Auswendiglernen ist im 3-Epochen-Lauf, ob sich der Abstand schließt oder umkehrt.
- **BatchNorm auf der Query** sieht im Training die Batch-Statistik inklusive späterer Tokens (bei 8.192
  Tokens pro Mikro-Batch ein sehr schwacher Kanal). Alle berichteten Zahlen sind im Eval-Modus mit
  festen Statistiken gemessen; die Kausalität dort ist per Test geprüft.
- **Dichter Adam auf der Wertetabelle.** Der Unit-Test zeigt, dass Gradienten nur in gelesene Einträge
  fließen. Adams Momentum bewegt aber auch ungelesene Einträge noch einige Schritte weiter. Für das
  spätere Ziel „einzelne Einträge gezielt aktualisieren“ braucht es einen sparsamen Optimierer
  (z. B. SparseAdam / lazy Adam); das war nicht Teil dieser Stufe.
- **Durchsatz** ist im PyTorch-Eager-Modus ohne eigene Kernels gemessen. Batch-1-Decoding ist durch den
  Kernel-Start-Overhead begrenzt (≈ 150–210 tok/s für alle Modelle) und sagt nichts über die
  Speicherbandbreite aus, um die es in späteren Stufen geht. B ist im Training ≈ 20 % langsamer als A,
  trotz gleicher FLOPs (Gather/Scatter auf der 100-M-Tabelle, Adam über 100 M Werte).
- **Perplexity** ist auf GPT-2-BPE-Tokens berechnet und nicht direkt mit den wortbasierten
  WikiText-103-Werten aus der Literatur vergleichbar (die Wort-PPL-Spalte rechnet die Gesamt-NLL auf die
  217.646 Wörter + `<eos>` um und ist nur grob vergleichbar).
- **Kleiner Maßstab, ein Datensatz.** Die Paper zeigen den Nutzen von Speicherschichten bei
  Billionen von Tokens und faktenlastigen QA-Aufgaben; WikiText-103 mit ≤ 354 M Tokens ist ein weit
  kleineres Regime.

## Artefakte

- `runs/<phase>/<lauf>/run-info.json`, `metrics.csv`, `train_log.csv`, `stdout.log`: in Git.
- `runs/<phase>/<lauf>/model.pt` (Gewichte, fp32), bei B zusätzlich `mem_access_train.npy`
  (Lesezugriffe je Eintrag über das ganze Training), `mem_access_val.npz` (Zugriffe und summierte
  Softmax-Gewichte je Eintrag auf dem Val-Set) und `mem_index_sample.npz` (Index-Stichprobe für Stufe 3):
  nur auf der Platte (`.gitignore`, zu groß für Git).
- Grafiken und Tabellen: `report/`, erzeugt mit `scripts/make_report.py probe ep1 ep3`.
