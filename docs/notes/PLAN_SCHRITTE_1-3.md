# Hampter: Plan für die Schritte 1–3 (zur Freigabe)

Stand 04.10.2026. Bis zur Freigabe ist nichts gebaut, gestartet oder gebucht.
Euro-Werte: EZB-Kurs vom 02.10.2026, 1 € = 1,1225 $.

## Vorab: alter Pod (erledigt)

- **Pod gelöscht:** `list-pods` liefert eine leere Liste. `list-network-volumes` liefert ebenfalls eine leere
  Liste. Laut Abrechnung enden die Plattenkosten am 04.10. gegen 13:30 Uhr. Es läuft nichts mehr, und es
  ist kein Speicher mehr belegt.
- **Korrektur der Kosten:** Die letzte Cloud-Runde hat laut Runpod-Abrechnung **25,51 $** gekostet (GPU
  24,98 $, Platte 0,52 $), nicht 21,01 $ wie von mir gemeldet. Meine Abfrage kam zu früh, als die Abrechnung
  noch nicht vollständig war. REPORT.de.md ist korrigiert.
- **Folge:** Bei 50 $ Aufladung sind noch **≈ 24,50 $ (≈ 21,80 €)** übrig, nicht ≈ 29 $. Ich plane mit
  diesem Wert. **Bitte nenne mir den genauen Stand aus dem Dashboard.** Über die Schnittstelle kann ich ihn
  nicht lesen.

---

## Schritt 1: Gegenwert der Tabelle (Cloud)

### Aufbau

Es werden dichte Llama-Modelle wie A trainiert, ohne Speicherschicht. Ich halte `head_dim = 64` fest und
setze die FFN-Breite auf ≈ 8/3 · d, gerundet auf ein Vielfaches von 64, wie bei A (384 → 1024). Breite und
Tiefe wachsen gemeinsam.

| Modell | d | Layer | Köpfe | FFN | ohne Emb. | Emb. | Train-FLOPs/Token | Checkpoint |
|---|---|---|---|---|---|---|---|---|
| A (vorhanden, zu Hause) | 384 | 12 | 6 | 1024 | 21,2 M | 19,3 M | 0,27 G | – |
| D-50M | 640 | 10 | 10 | 1728 | 49,6 M | 32,2 M | 0,53 G | 0,33 GB |
| D-100M | 768 | 14 | 12 | 2048 | 99,1 M | 38,6 M | 0,89 G | 0,55 GB |
| D-200M | 1024 | 16 | 16 | 2752 | 202 M | 51,5 M | 1,62 G | 1,0 GB |
| D-400M | 1280 | 20 | 20 | 3456 | 397 M | 64,4 M | 2,92 G | 1,8 GB |

- **Daten:** Alles bleibt wie bei A und B:
  - dieselben 500 M Wikipedia-Tokens, Daten-Seed 1234, Init-Seed 0
  - dasselbe Val-Set (Wikipedia 1,48 M Tokens), dazu WikiText-103 als Nebenwert
  - derselbe LR-Plan: 6e-4, Warmup 5 %, Cosine auf 10 %, AdamW (0,9 / 0,95), Weight Decay 0,1, Clip 1,0
  - 32.768 Tokens pro Schritt
- **Mikro-Batch:** Er wird nur an den Speicher angepasst, mit Gradient-Akkumulation auf dieselben 32.768
  Tokens. Am Rechenweg ändert das nichts.
- **A als kleinster Punkt:** A ist bereits gemessen: PPL 25,665 mit Seed 0, wie die neuen Läufe; Mittel aus zwei Seeds 25,711. Gleiche Daten, gleiches Val-Set, zu Hause.
  Dass Heim- und Cloud-Ergebnisse vergleichbar sind, zeigt B-1M: 21,8369 zu Hause gegenüber 21,8365 in der
  Cloud. Damit hat die Kurve 5 Punkte, von 21 M bis 400 M.

### Auswertung (Messung ohne Bestanden-Kriterium)

- **Kurve:** Val-PPL gegen Parameter ohne Embeddings, log–log.
- **Gleichwertige Größe** für B-1M (21,84), B-4M (20,80) und B-16M (19,96):
  - Hauptwert: stückweise lineare Interpolation von log(PPL) über log(N) zwischen den Nachbarpunkten.
  - Zum Vergleich: Potenzgesetz-Fit PPL = E + a·N^(−α) über alle Punkte.
- **Unsicherheit:**
  - Die Seed-Spanne von ≈ 0,4 % PPL (Stufe 1c: A s0/s1 0,35 %, B-1M s0/s1 0,39 %) wird auf die dichten Punkte und auf B angesetzt.
  - Daraus folgt über die lokale Steigung ein Bereich für N.
  - Dazu kommt die Differenz zwischen beiden Fit-Methoden. Angegeben wird der größere Bereich.
- **Einklammerung, vorher festgelegt:** Ist das größte gelaufene dichte Modell schlechter als ein B-Modell,
  steht dort „> größtes Modell“. Es wird nicht weiter extrapoliert. Wenn D-400M wegfällt und D-200M noch über
  19,96 liegt, heißt es für B-16M also nur „> 200 M“.
- **Rechenaufwand pro Token** in derselben Tabelle:
  - Vorwärts-MACs, beim Training ×3
  - für B zusätzlich die gelesenen Tabellen-Bytes pro Token (384 Zeilen × 384 Werte)
  - B-1M 47,1 M MACs, B-4M ≈ 50 M, B-16M ≈ 57 M (Methode wie im REPORT); dichte Modelle 0,09–0,95 GFLOP
    vorwärts

### Ablauf in der Cloud

1. **Vorbereitung zu Hause** (nur CPU, ≈ 2–3 h):
   - Presets D-50M bis D-400M
   - Tests für die dichten Modelle
   - eine Warteschlange `run_dense.py` mit Budget-Wächter
   - `setup.sh` verschlankt: keine Kernel-Tests, Probelauf nur für die dichten Modelle
   - Ende-Ablauf (Punkt 5) und Trockenlauf auf der CPU
2. **Pod-Anlage:** Ich frage dich mit Preis, bevor ich den Pod anlege.
   - Daten: Ich lade die fertigen Token-Dateien von hier hoch (≈ 1,2 GB, sha256-geprüft), parallel zur
     Python-Installation. Das ersetzt den 11-GB-Download samt Neu-Tokenisierung. Klappt der Upload nicht,
     baue ich die Daten wie beim letzten Mal auf dem Pod neu.
3. **Probelauf auf der GPU:** Jede Größe läuft ≈ 1 Minute allein. Dabei werden tok/s und VRAM-Spitze
   gemessen.
4. **Budget-Wächter, vorher festgelegt:**
   - Hochrechnung = bisherige Pod-Zeit + Σ (500 M / gemessene tok/s je Größe, als liefen sie nacheinander)
     × 1,15 + 0,4 h für das Ende.
   - D-50M und D-100M laufen immer, D-200M wenn es passt, D-400M nur wenn auch das passt.
   - Was wegfällt, wird gemeldet (ntfy, REPORT).
   - Alle zugelassenen Läufe starten **gleichzeitig**. Der Speicher reicht: geschätzt zusammen ≈ 55 GB.
5. **Ende, wie von dir freigegeben:**
   - Nach jedem fertigen Lauf: Ergebnisse pushen und den Checkpoint sofort per rsync holen.
   - Am Ende: `sha256sum -c` → Pod stoppen → Pod löschen. Das mache ich von hier über die Runpod-Schnittstelle.
   - Wenn das Sichern fehlschlägt: Pod nur stoppen und eine ntfy-Nachricht schicken, von hier aus, weil der
     Fehler hier auffällt.
   - Fallback: Der Pod stoppt sich selbst, wenn 20 Minuten nach dem Ende nichts geholt wurde.
   - Kein `submit_feedback`.
6. **Pro Lauf:** GPU-Temperatur, Leistung und Takt als CSV, alle 10 s. Weil die Läufe sich die GPU teilen,
   zeigt die CSV die ganze Karte. Die Tempo-Werte sind nicht mit Einzelläufen vergleichbar; das steht dann
   auch im REPORT.

### Zeit und Kosten

- **Trainingszeit:** Aus 6·N·D plus Attention, bei 25–40 % der H200-Spitzenleistung. Der Probelauf misst
  die echte Zahl.
- **Gesamtzeit:** Dazu kommen 30–45 min Setup, ≈ 10 min Probelauf und 10–20 min fürs Ende.

| Variante | Pod-Zeit | H200 SXM Secure (4,59 $/h) | H100 SXM Secure (3,49 $/h) |
|---|---|---|---|
| alle vier Größen | 2,9–4,7 h | 13,50–21,40 $ = **12,00–19,10 €** | 11,10–17,40 $ = **9,90–15,50 €** |
| ohne D-400M | 1,9–3,0 h | 8,80–13,80 $ = **7,80–12,30 €** | 7,00–11,10 $ = **6,20–9,90 €** |

- **Deckel:** höchstens **20 $ (17,80 €)** Pod-Kosten. Damit bleiben ≥ 4 $ Rest, und das Guthaben fällt nie
  auf 0. Bei 0 löscht Runpod Pods samt Daten.
- **Harte Pod-Zeitgrenze** ab Pod-Start: 4,3 h auf H200 bzw. 5,6 h auf H100.
- Im Starter-Paket steht noch `SMLM_MAX_HOURS=7`. Das wären ≈ 32 $, mehr als das Guthaben; ich ändere es.
- **Speicher:**
  - Pod: Volume 60 GB (Repo, Python-Umgebung ≈ 8 GB, Daten, Checkpoints ≈ 3,7 GB), Container 30 GB.
  - Plattenkosten: ein paar Cent.
  - Hier: +3,7 GB.
- **Empfehlung H100 SXM statt H200.** Für dichte Modelle dieser Größe ist die H100 praktisch gleich schnell:
  Sie hat dieselbe bf16-Rechenleistung, nur weniger und langsameren Speicher; ich rechne mit bis zu 10 %
  mehr Zeit. Ihre 80 GB reichen für alle vier Läufe gleichzeitig. Sie ist 24 % billiger, damit passt D-400M
  sicher ins Budget, und sie ist besser verfügbar (HIGH statt MEDIUM). Du hattest H200 vorgegeben, deshalb
  entscheidest du.

### Ehrliche Grenzen (kommen so in den REPORT)

- **Wenige Tokens für große Modelle:** 500 M Tokens sind für 200–400 M Parameter wenig. Chinchilla-optimal
  wären ≈ 20 Tokens pro Parameter. Die gleichwertige Größe gilt also **nur für dieses Token-Budget**.
- **LR nicht angepasst:** Der LR-Plan ist für A gewählt. Größere Modelle würden mit einer abgestimmten LR
  wahrscheinlich etwas besser. Das lässt die Tabelle eher **zu gut** aussehen.
- **Ein Seed je dichter Größe.**
- **Instabile Läufe:** Wird ein großes Modell mit LR 6e-4 instabil (NaN oder Loss-Explosion), wird das
  gemeldet und nicht automatisch neu gestartet.

---

## Schritt 2: B-16M zu Hause

### Grundzahlen

- **Tabelle:** 16,8 M Zeilen × 384.
  - fp32: 25,8 GB
  - bf16: 12,9 GB
  - 4 Bit: 3,2 GB, plus 32 MB Skalen, die immer im RAM liegen
- **Pro Token:** 3 Schichten × 4 Köpfe × 32 = **384 Zeilen**, alle verschieden.
- **Prefill:** Über 1024 Tokens sind es ≈ 270 verschiedene Zeilen pro Token.
- **Rechner:**
  - 125 GB RAM
  - NVMe: Samsung 990 PRO, `/home`. Das Repo liegt auf einer SATA-SSD, deshalb kommen die Dateien für c)
    nach `/home`.
  - Grafikkarte: 16 GB, davon belegt der Desktop gerade ≈ 1 GB.

**Trefferquote eines festen Caches** mit den im Training meistgelesenen Zeilen, gemessen auf dem Val-Set
(aus den gespeicherten Zugriffszahlen von B-16M):

| Cache | Zeilen | RAM (4 Bit) | Treffer |
|---|---|---|---|
| 5 % | 0,84 M | 155 MB | 41 % |
| 10 % | 1,7 M | 310 MB | 54 % |
| 30 % | 5,0 M | 930 MB | 79 % |
| 50 % | 8,4 M | 1,55 GB | 91 % |

Die Zugriffe sind recht gleichmäßig verteilt. Ein kleiner Cache hilft deshalb weniger als erhofft. Ein
nachgeladener Cache (LRU) wird zusätzlich gemessen.

### Die drei Varianten

- **a) Tabelle im Grafikspeicher**, bf16 und 4 Bit, mit den vorhandenen Kerneln und Decode-Graphen.
  - bf16 braucht ≈ 12 GiB plus Rest. Zusammen mit dem Desktop bleiben ≈ 2,3 GiB frei. Das ist knapp über
    deiner 2-GB-Regel; bleibt weniger frei, lasse ich bf16-im-VRAM aus und notiere es.
  - 4 Bit passt locker.
- **b) Tabelle im RAM**, bf16 und zur Kontrolle fp32:
  - Pro Schicht gehen die Indizes zur CPU, die CPU sammelt die Zeilen, und nur diese gehen zur GPU.
  - Das sind drei Hin- und Rückwege pro Token, deshalb gibt es hier keine Decode-Graphen.
  - fp32 im RAM reproduziert zugleich den Cloud-Wert 19,960 zu Hause.
- **c) 4-Bit-Datei auf der NVMe per mmap**, mit RAM-Cache für häufige Zeilen.
  - Fehlende Zeilen werden je Schicht **parallel** angefordert (`madvise`), nicht einzeln per Seitenfehler.
    Mit Seitenfehlern liefe nur eine Anfrage gleichzeitig, und das wäre kalt sehr langsam.
  - Jede Zeile kostet eine 4-KB-Seite. Das wird ehrlich so gezählt.

### Messungen

- **Tempo:**
  - Decode (Batch 1): tok/s und ms/Token
  - Prefill: tok/s
- **Speicher:**
  - VRAM: rocm-smi und PyTorch
  - RAM: RSS
  - Seiten-Cache der Datei: `fincore`
- **NVMe:** Lesezugriffe/s aus `/sys/block/nvme1n1/stat`.
- **Cache:** Trefferquote.
- **c) in drei Zuständen:**
  - **kalt:** Datei vorher mit `posix_fadvise(DONTNEED)` aus dem Seiten-Cache werfen, Kontrolle mit
    `fincore`. **Dafür ist kein root nötig.**
  - **begrenzt:** Prozess in einer cgroup mit RAM-Grenze (`systemd-run --user -p MemoryMax=…`). Sonst legt
    Linux die ganze 3,2-GB-Datei in den freien RAM, und c) wäre nach dem Aufwärmen nur b). Ich prüfe vorher,
    dass der Seiten-Cache wirklich auf diese Grenze angerechnet wird. Klappt beides nicht, frage ich dich
    nach root.
  - **warm, ohne Grenze:** obere Schranke.
- **Qualität:**
  - Zuerst wird die Val-PPL für a) bf16 und a) 4 Bit gemessen; beide sind für B-16M noch unbekannt.
    Erwartung gegenüber fp32 19,960: ≤ +0,1 % bzw. ≤ +0,5 %.
  - b) und c) lesen dieselben Zahlen und müssen deshalb ihrer a)-Variante entsprechen (|Δ| ≤ 0,01 %).
  - b) fp32 muss 19,960 treffen.

### Erwartung (Schätzung, kein Versprechen)

| | Decode | Prefill |
|---|---|---|
| a) | wie gehabt | wie gehabt |
| b) | etwas langsamer (Hin- und Rückwege, keine Graphen) | 5–10× langsamer (CPU-Sammeln, PCIe) |
| c) warm | ähnlich wie b) | stark langsamer, weil die NVMe-Zugriffe begrenzen: grob 2.000–10.000 tok/s |
| c) kalt | merklich langsamer, bis der Cache gefüllt ist | |

**Wahrscheinliche Aussage:** Zum Schreiben reicht RAM oder NVMe, zum schnellen Einlesen großer Texte eher
nicht.

### Zeit und Speicher

- **Meine Arbeit (nur CPU):** ≈ 4–6 h. Dazu gehören:
  - Umwandlung fp32 → bf16/4 Bit, auf der CPU per mmap, ≈ 26 GB RAM
  - Lesepfade b) und c), Cache
  - Messskript
  - CPU-Unit-Tests
- **GPU:** ≈ 60–90 min, teilbar in zwei Fenster. **Ich frage vorher nach einem Fenster und sammle bis dahin.**
- **Platte:** ≈ 16 GB auf der NVMe (bf16- und 4-Bit-Datei).
- **RAM:** bis ≈ 30 GB (b) fp32).
- **Kosten:** keine Cloud-Kosten.

---

## Schritt 3: Tabelle in Qwen (nur vorbereiten)

### Basis

- **Modell:** `Qwen/Qwen3.5-0.8B`.
  - **Lizenz Apache 2.0** laut Modellkarte. Bei der Vorbereitung zitiere ich die LICENSE-Datei auf einer
    festgehaltenen Version.
  - Veröffentlicht am 02.03.2026. Wissensstand laut Drittquelle Nov. 2025; Qwen selbst nennt keinen.
- **Aufbau:**
  - 24 Blöcke: 18 × Gated DeltaNet (lineare Attention) und 6 × normale Attention, im Muster 3:1
  - d = 1024, FFN 3584
  - Vokabular 248.320
- **Variante:** Das genannte Modell ist die nachtrainierte, multimodale Fassung. Ich nutze nur den Textteil.
  Es gibt auch `Qwen3.5-0.8B-Base`; ich bleibe aber bei deiner Vorgabe.

### Zusatzgedächtnis

- **Einbau:** Qwen bleibt vollständig und eingefroren. Hinter Block 6, 12 und 18 kommt je ein **zusätzlicher**
  Block: h ← h + g · Speicher(RMSNorm(h)).
  - eine gemeinsame Tabelle wie bei B
  - Werte-Breite 1024, damit kein Umrechnungs-Layer nötig ist
  - 1 M Zeilen = 1,07 Mrd. Werte
  - Regler g je Block, **Start 0**
- **Trainiert werden:**
  - Tabelle
  - Query-Netze mit BatchNorm
  - Regler
  - **Sub-Keys:** Ich würde sie mittrainieren, sie gehören zur Suche. **Bitte bestätigen.**
- **Unit-Tests:**
  - Bei g = 0 sind die Logits **bitgleich** zu Qwen allein, im Train- und im Eval-Modus. Dabei achte ich
    auf BatchNorm und NaN·0.
  - Im ersten Schritt bekommt nur g einen Gradienten, danach auch die Tabelle.
  - Eingefrorene Gewichte ändern sich nie.
  - Kleine CPU-Tests mit einer zufälligen Mini-Qwen-Konfiguration.

### Daten

- **Tokenisierung:** neu mit Qwens Tokenizer. Gespeichert wird als uint32, weil das Vokabular über 65.536
  Einträge hat.
- **„Neues Wissen“:** Wikipedia-Artikel, die **nach** Qwens Wissensstand angelegt wurden.
  - Quelle: aktueller enwiki-Dump; nur die letzten Teildateien, denn sie enthalten die neuesten Seiten-IDs.
    Die Größe prüfe ich bei der Vorbereitung.
  - **Echten Stichtag messen:** Qwens PPL nach Anlege-Monat (2025-01 … 2026-09). Ein Sprung zeigt den
    echten Stichtag. Genommen wird nur, was danach liegt; ohne sichtbaren Sprung ab 03/2026.
- **„Bekanntes Wissen“:** unser altes Wikipedia-Val-Set (Dump 2023). Qwen kennt es sehr wahrscheinlich.

### Vorschlag zur Messung (zur Freigabe)

**Läufe:**

- **Q:** Qwen allein, g = 0, kein Training.
- **Q+T:** mit Tabelle.
- **Q+D:** Kontrolle mit einem kleinen dichten Zusatzblock an denselben Stellen, gleicher Rechenaufwand,
  gleiche Daten und Schritte. Ohne diese Kontrolle wäre ein Gewinn nicht von bloßer Anpassung an den
  Wikipedia-Stil zu unterscheiden. **Empfohlen, kostet einen zweiten Lauf.**

**Hilft es?** Entscheidend ist die PPL auf zurückgehaltenen *neuen* Artikeln, die das Training nicht gesehen
hat.

| Urteil | Bedingung |
|---|---|
| hilft deutlich | Q+T ≤ 0,95 × Q **und** ≤ 0,98 × Q+D |
| hilft etwas | Q+T ≤ 0,98 × Q |
| hilft nicht | sonst |

Nur berichtet: PPL auf trainierten neuen Artikeln (was die Tabelle speichern kann) und auf altem Wikipedia.

**Schadet es?** „Schadet nicht“ verlangt alle drei Punkte:

1. **Standard-Test** mit lm-evaluation-harness, feste Stichproben: MMLU, ARC-Easy, ARC-Challenge, HellaSwag,
   PIQA, WinoGrande.
   - Der Mittelwert fällt um höchstens 1,0 Prozentpunkte.
   - Keine Aufgabe fällt um mehr als max(2 Pp., 2 × Standardfehler).
2. **Altes Wikipedia:** PPL höchstens +1 %.
3. **Chat:** 12 feste Fragen (6 deutsch, 6 englisch: Wissen, Denken, Anweisungen, eine Frage zu 2026),
   gierige Ausgabe, je zwei Antworten **verblindet** nebeneinander. Du urteilst; ich zähle aus.

Die Schwellen sind mein Vorschlag. Du kannst sie ändern, bevor etwas läuft.

### Abschätzung

- **Rechenaufwand:** ≈ 4 GFLOP pro Trainings-Token. Gezählt sind Vorwärtslauf und der Rückwärtslauf durch
  die eingefrorenen Blöcke oberhalb der ersten Speicherschicht.
- **Tokens:** Geplant sind ≈ 120 M Trainings-Tokens, also etwa 2 Durchgänge über ≈ 60 M neue Artikel. Die
  genaue Zahl steht erst nach dem Dump fest.

| | Speicher | Zeit | Kosten |
|---|---|---|---|
| H200 | Qwen 1,6 GB + Tabelle fp32 4,3 GB + Adam 8,6 GB + Aktivierungen; die Logits über 248 k Tokens brauchen eine stückweise Loss-Berechnung → ≈ 30–40 GB | je Lauf 30–45 min; mit Setup, 2 Läufen und Tests ≈ 2–2,5 h | 9,20–11,50 $ = **8,20–10,20 €** (H100: 7,00–8,70 $ = 6,20–7,80 €) |
| zu Hause | 1 M × 1024 mit Adam ≈ 13 GB → passt nicht neben Qwen in 16 GB; ginge nur mit ≈ 256 k Zeilen | grob 7 h je Lauf | – |

- **Risiko zu Hause:** Ob Gated DeltaNet unter ROCm schnelle Kernel hat (flash-linear-attention,
  causal-conv1d), ist offen. Ohne sie läuft eine langsame PyTorch-Schleife.
- **Empfehlung:** zu Hause vorbereiten, testen und einen kurzen Probelauf machen; der echte Lauf in der Cloud.
- **Budget:** Nach Schritt 1 reicht das Guthaben dafür voraussichtlich **nicht**. Es bräuchte ≈ 15 $ mehr.
  Darüber entscheidest du später.
- **Pakete:** `transformers` mit Qwen3.5-Support ist nicht installiert. Ich lege dafür eine eigene Umgebung
  an (`.venv-qwen`), damit die bestehende unverändert bleibt.
- **Vorbereitungszeit:** ≈ 5–8 h meiner Arbeit. Dazu gehören Dump holen und filtern, Tokenisieren, Code,
  Tests und Abschätzung.
- Kein Training vor deiner Freigabe der Kriterien.

---

## Nebenbei: README-Entwurf

`README_DRAFT.md` (≈ 1–2 h) mit diesen Teilen:

- Idee und Aufbau
- Ergebnisse mit Kurven: Stufe 1, B-1M/4M/16M, die Kurve aus Schritt 1
- Nachbauen: Daten, Training, Cloud
- Ehrliche Grenzen: ein Seed, Token-Budget, Rechenzeit gegenüber A, Kosten

Nur ein Entwurf, nichts wird veröffentlicht.

## Ordnung

- Jeder Schritt bekommt einen Abschnitt in REPORT.de.md.
- Die Kriterien für Schritt 3 kommen erst nach deiner Freigabe hinein, vor jedem Lauf.
- Nach jedem abgeschlossenen Schritt bekommst du eine kurze md-Datei aufs Handy.
- **Reihenfolge nach deinem Go:**
  - Zuerst Schritt 1 vorbereiten.
  - Dann frage ich vor der Pod-Anlage.
  - Während der Pod läuft: Code für Schritt 2. Danach frage ich nach einem GPU-Fenster.
  - Schritt 3 und das README nebenher.

## Was ich von dir brauche

1. **Guthaben:** den genauen Stand aus dem Dashboard.
2. **Grafikkarte:** H200 SXM wie vorgegeben oder H100 SXM (empfohlen, ≈ 24 % billiger)?
3. **ntfy:** In der ntfy-App das Thema **`<zufälliges-thema>`** abonnieren. Es ist zufällig erzeugt, damit
   niemand mitliest. Bisher war kein Thema eingetragen, deshalb wären Nachrichten ins Leere gegangen.
4. **Schritt 3:**
   - Sub-Keys mittrainieren: ja oder nein?
   - Kontrolllauf Q+D: ja oder nein?
   - Datenquelle „neue Wikipedia-Artikel nach dem Wissensstand“: einverstanden?
5. **Go** für den Plan.
