# Projekt: Sparse-Memory-LM (Arbeitstitel)

## Die Idee
Langfristiges Ziel: Ein Sprachmodell mit dem Wissen eines sehr großen 
Modells (Richtung 1T Parameter), das lokal auf Consumer-Hardware läuft.

Der Engpass auf Heim-Hardware ist nicht die Rechenleistung, sondern der 
schnelle Speicher: Bei normalen Modellen (auch MoE) müssen pro Token 
Gigabytes an Gewichten gelesen werden. 

Ansatz: Ein kleiner Rechenkern (normaler Transformer, "der Grips") plus 
eine riesige, dünn genutzte Key-Value-Speicherschicht ("das Wissen"). 
Pro Token werden per Product-Key-Suche nur wenige Einträge gelesen. 
Dadurch könnte die Speicherschicht später auf einer NVMe-SSD liegen 
statt im RAM/VRAM, und einzelne Einträge könnten gezielt aktualisiert 
werden (Weiterlernen ohne Vergessen).

Grundlagen: Lample et al. 2019 "Large Memory Layers with Product Keys", 
Meta 2024 "Memory Layers at Scale", He 2024 "Mixture of A Million 
Experts". Lies dir die relevanten Details (Query-Normalisierung, 
Lernraten der Werte, Umgang mit ungenutzten Keys) aus den Papern bzw. 
der Referenzimplementierung an, bevor du implementierst.

## Diese Stufe (Stufe 1): Funktioniert es überhaupt?
Frage: Verbessert eine Product-Key-Speicherschicht ein kleines 
Sprachmodell bei gleichem Rechenaufwand pro Token?

Ausdrücklich NICHT in dieser Stufe: eigene Kernel, SSD-Auslagerung, 
eigene Laufzeitumgebung, Performance-Optimierung über das Übliche hinaus.

## Hardware
Linux, AMD-GPU mit 16 GB (ROCm, gfx1201), 128 GB RAM. 
Prüfe zuerst, ob PyTorch mit ROCm die GPU korrekt nutzt (kleiner 
Matmul-Test gegen CPU). Wenn nicht, stopp und berichte, bevor du 
weitermachst.

## Daten
WikiText-103 (faktenreich, Standard-Benchmark). Vorhandenen Tokenizer 
nutzen (z.B. GPT-2-BPE), keinen eigenen trainieren.

## Drei Modelle, identisch trainiert
A) Baseline: kleiner Transformer, ca. 20M Parameter ohne Embeddings.
B) Gedächtnis: wie A, aber eine FFN-Schicht in der Mitte wird durch 
   eine Product-Key-Memory-Schicht ersetzt. Start: 512x512 = 262.144 
   Einträge, Top-k 32, 4 Köpfe. Rechenaufwand pro Token muss etwa 
   gleich A sein.
C) Großes dichtes Modell: Gesamtparameter etwa wie B, aber normal dicht.

Gleiche Token-Anzahl, gleicher Lernraten-Plan (außer wo das Paper für 
die Speicherschicht ausdrücklich etwas anderes vorgibt), feste Seeds. 
Token-Budget konfigurierbar; erst ein kurzer Probelauf (z.B. 20M 
Tokens), dann der echte Lauf.

## Messen und protokollieren
- Validierungs-Perplexity über den Trainingsverlauf
- Tokens/s im Training und in der Inferenz
- aktive Parameter pro Token, Gesamtparameter, VRAM-Verbrauch
- Für B zusätzlich: Anteil der Einträge, die überhaupt benutzt werden, 
  und Verteilung der Zugriffe (wie oft wird welcher Eintrag gelesen). 
  Speichere eine Stichprobe der gelesenen Indizes pro Token für 
  spätere Analyse (Stufe 3: Zugriffsmuster für SSD).

## Korrektheit
- Unit-Test: Die Product-Key-Suche liefert bei kleiner Tabelle exakt 
  dieselben Top-k wie eine Brute-Force-Suche über alle Einträge.
- Unit-Test: Gradient fließt nur in die tatsächlich gelesenen Werte.

## Ordnung
- Git-Repo anlegen, jeder Lauf mit Commit-Hash protokolliert.
- Pro Lauf eine run-info.json (Konfiguration, Hardware, Seeds, Dauer) 
  und eine CSV mit den Messwerten.
- Am Ende ein Bericht REPORT.md: Tabelle A/B/C, Perplexity-Kurven als 
  Grafik, ehrliche Einordnung. Wenn B nicht besser als A ist, sag das 
  klar und nenne mögliche Gründe. Nichts schönreden.

Bevor du anfängst: Fasse mir in wenigen Sätzen zusammen, wie du A, B 
und C konkret dimensionierst und wie lange du die Läufe schätzt.
