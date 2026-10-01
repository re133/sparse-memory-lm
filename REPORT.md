# Stufe 1: Verbessert eine Product-Key-Speicherschicht ein kleines Sprachmodell bei gleichem Rechenaufwand?

> Status: **Erfolgskriterien festgelegt, bevor ein Lauf gestartet wurde.** Ergebnisse folgen unten.

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

_(wird nach den Läufen ergänzt)_
