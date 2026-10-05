# Cloud-Läufe B-1M / B-4M / B-16M auf Runpod (1 × H200): Anleitung

Kurz: Pod in der Runpod-Konsole anlegen → Starter-Paket hochkopieren → ein Befehl → warten (Handy-Nachricht)
→ der Pod stoppt sich selbst → Ergebnisse per `git pull`, Checkpoints per `rsync` holen → Pod löschen.

**Was läuft:** B-1M (Kontrolle), B-4M, B-16M mit denselben Daten, Einstellungen und demselben Val-Set wie
B-1M-sparse s0. Je 500 M Tokens, Triton-Kernels. Die Kriterien stehen in REPORT.md (Abschnitt
„Kriterien für die Cloud-Läufe“). Nach jedem Lauf landet ein Zwischenstand in REPORT.md auf GitHub.

Kein Runpod-Plugin und kein API-Key nötig: Du legst den Pod selbst in der Weboberfläche an. Zum Stoppen am
Ende benutzt der Pod den Schlüssel, den Runpod automatisch in jeden Pod legt und der nur für diesen Pod gilt.

## 0. Einmalig vorbereitet (schon erledigt)

- **Privates Repo:** `github.com/re133/sparse-memory-lm`, Code und Ergebnisse. Der Pod pusht mit einem
  Deploy-Key, der nur für dieses Repo gilt.
- **Starter-Paket `~/smlm-cloud-kit/`** auf deinem PC:
  - `setup.sh` und `stop_pod.sh`
  - `deploy_key` (privat, nicht weitergeben)
  - `cloud.env` (fertig; nur `NTFY_TOPIC` kannst du optional eintragen)

## 1. Runpod-Konto vorbereiten

1. **Guthaben:** mindestens **40 $** aufladen, besser Auto-Pay oder eine Warnung bei niedrigem Guthaben
   einschalten.
   - **Wichtig:** Fällt das Guthaben auf 0 $, stoppt Runpod alle Pods. Pods ohne Netzwerk-Volume werden
     dabei **gelöscht, samt Daten**.
2. **SSH-Key:** Deinen öffentlichen Key (`~/.ssh/id_ed25519.pub`) unter *Settings → SSH Public Keys*
   eintragen, als ganze Zeile mit `ssh-ed25519 …` am Anfang.

## 2. Pod anlegen (Runpod-Konsole → Pods → Deploy)

| Einstellung | Wert | Warum |
|---|---|---|
| Cloud | **Secure Cloud** | öffentliche IP, ohne die gehen `scp`/`rsync` nicht |
| GPU | **1 × H200 SXM (141 GB)** | B-16M braucht ≈ 102 GiB; H100 (80/94 GB) reicht nicht |
| Template | **Runpod PyTorch** (aktuelle Version, CUDA 12.8 oder neuer) | Treiber, SSH und `runpodctl` sind dabei |
| Container Disk | 40 GB | Systempakete; wird bei jedem Stop geleert |
| **Volume Disk** | **150 GB**, Mount-Pfad `/workspace` | Repo, Python-Umgebung, Daten (13 GB), Checkpoints (≈ 35 GB) – überlebt einen Stop |
| Preis | **On-Demand** (nicht Spot/Interruptible) | Spot-Pods können mitten im Lauf weggenommen werden |
| Optional | Umgebungsvariable `NTFY_TOPIC` | sonst in `cloud.env` |

**Deploy On-Demand** klicken und warten, bis der Pod läuft. Im Tab **Connect** steht unter
**SSH over exposed TCP** ein Befehl wie `ssh root@<IP> -p <PORT> -i ~/.ssh/id_ed25519`. IP und Port
notieren.

## 3. Hochladen und starten

```bash
scp -P <PORT> -r ~/smlm-cloud-kit root@<IP>:/workspace/
ssh root@<IP> -p <PORT>
tmux new -s setup 'bash /workspace/smlm-cloud-kit/setup.sh'
```

Mit `Strg-b`, dann `d` löst du dich von der Sitzung; die Verbindung darfst du trennen.

**Was `setup.sh` macht** (ca. 40–55 min, jeder fertige Schritt wird bei einem Neustart übersprungen):

1. **Pakete** (git, tmux, rsync) installieren.
2. **GPU prüfen** (Treiber bringt Runpod mit).
3. **Repo** mit dem Deploy-Key nach `/workspace/AngryAnt` klonen.
4. **Python-Umgebung** mit PyTorch (CUDA) und Triton einrichten.
5. **Daten:**
   - WikiText-103 und Wikipedia 20231101.en von Hugging Face laden, in festgepinnten Versionen (≈ 11 GB).
   - Neu tokenisieren und per **SHA-256 prüfen, dass alles bytegleich mit deinem PC ist**.
6. **Alle Tests**: GPU und CPU-Interpreter. **Nur wenn alles grün ist, geht es weiter.**
   Dazu gehören zwei Tests, die nur auf großen GPUs laufen: Auswahl mit 4096 Keys je Hälfte (B-16M) und
   Tabellen mit mehr als 2³¹ Elementen.
7. **Probelauf auf der H200:** B-1M, B-4M und B-16M je 1 M Tokens, einmal komplett. Dabei werden
   Kompilieren, VRAM-Spitze, Auswertung, Checkpoint-Speichern (26 GB bei B-16M) und Inferenz geprüft,
   bevor bezahlte Stunden laufen. Dauer ≈ 10 min. Die VRAM-Spitzen stehen im Setup-Log.
8. **Runpod-API prüfen**, dann die Warteschlange in der tmux-Sitzung `queue` starten.

**Wenn etwas fehlschlägt:**
- Das Log wird nach GitHub gepusht (`runs/cloud/setup_failed.log`), du bekommst eine Nachricht, und der Pod
  stoppt sich.
- Pod in der Konsole wieder starten, einloggen, Log ansehen: `/workspace/.smlm-setup/setup.log`.
- Danach `bash /workspace/smlm-cloud-kit/setup.sh` erneut starten.

## 4. Während der Läufe

**Bitte während der Läufe nichts nach `main` pushen.** Der Pod holt sich zwar vor jedem Push den neuesten
Stand, aber ein Konflikt, z. B. in REPORT.md, würde seine Pushes blockieren. Die Ergebnisse lägen dann nur
im Pod.

- **Zwischenstand:** nach jedem Lauf in **REPORT.md** auf GitHub (Abschnitt „Zwischenstand Cloud“),
  dazu eine ntfy-Nachricht.
- **Live:** `ssh root@<IP> -p <PORT>`, dann `tmux attach -t queue` oder
  `tail -f /workspace/AngryAnt/runs/cloud/queue.log`.
- **GPU-Werte:** Temperatur, Leistung und Takt alle 10 s in `runs/cloud/<lauf>/gpu_thermal.csv`.
- **Sicherungen:**
  - Ein Lauf, dessen Logs sich 30 min nicht ändern, wird beendet; die Warteschlange macht weiter.
  - Nach 12 h insgesamt wird alles gestoppt.

## 5. Ende

- **Am Ende der Warteschlange:**
  - Eine Prüfsummenliste aller Checkpoints wird gepusht (`runs/cloud/checkpoints.sha256`).
  - Der Pod **stoppt sich über die Runpod-API**: GPU freigegeben, keine Rechenkosten mehr. `/workspace` mit
    den Checkpoints bleibt erhalten.
- **In der Konsole kontrollieren**, dass der Pod wirklich *Stopped* ist. Wenn nicht (Nachricht „could NOT
  be stopped“): Pod aufklappen → Stop.

## 6. Ergebnisse und Checkpoints holen

1. **Ergebnisse** (klein, über GitHub):

   ```bash
   cd /mnt/sandisk/Sparse-Memory-LM/AngryAnt && git pull
   ```

2. **Checkpoints** (≈ 35 GB: B-1M 1,7 GB, B-4M ≈ 6,6 GB, B-16M ≈ 26 GB):
   - Pod in der Konsole wieder **starten**. Ist die GPU inzwischen vergeben, bietet Runpod an, ihn
     **mit 0 GPUs** zu starten. Das genügt zum Kopieren und ist billiger.
   - IP und Port können sich geändert haben, also im Tab **Connect** nachsehen.
   - Dann holen und prüfen:

   ```bash
   rsync -avP --partial -e "ssh -p <PORT>" root@<IP>:/workspace/AngryAnt/runs/cloud/ runs/cloud/
   sha256sum -c runs/cloud/checkpoints.sha256
   ```

   Alle Zeilen müssen `OK` zeigen. Erst dann weiter.

## 7. Pod löschen

- In der Konsole **Terminate**. Erst damit enden alle Kosten. Ein gestoppter Pod kostet für 150 GB Volume
  0,20 $/GB/Monat, also ≈ 1 $/Tag.
- **Deploy-Key entfernen:**
  `gh repo deploy-key list -R re133/sparse-memory-lm`, dann `gh repo deploy-key delete <ID> -R re133/sparse-memory-lm`.

## Kosten (H200 SXM, On-Demand; Preise laut runpod.io/pricing, Stand Abruf 2026-10-03)

Secure Cloud 4,59 $/h, Community Cloud 3,59 $/h. Abgerechnet wird sekundengenau. Maßgeblich ist der Preis,
den die Konsole beim Anlegen zeigt.

| Abschnitt | Dauer (Schätzung) | Kosten (Secure) |
|---|---|---|
| Setup (Python, Daten, Tests, Probelauf) | 40–55 min | 3,10–4,20 $ |
| B-1M | 25–45 min | 1,90–3,40 $ |
| B-4M | 30–55 min | 2,30–4,20 $ |
| B-16M | 40–75 min | 3,10–5,70 $ |
| Checkpoints holen (Pod wieder gestartet; mit 0 GPUs billiger) | 20–60 min | 0–4,60 $ |
| Volume, solange der Pod nur gestoppt ist | pro Tag | ≈ 1 $ |
| **Summe bei zügigem Löschen** | **≈ 2,5–4,5 h** | **≈ 11–22 $** |

Die Laufzeiten sind von der RX 9070 hochgerechnet (H200: ≈ 7,5× Bandbreite, ≈ 5× Rechenleistung; das
kleine Modell lastet die H200 aber nicht aus) und **bis Faktor 2 unsicher**. Die Grenze von 12 h kostet
höchstens ≈ 55 $.

*Frühere Variante für IONOS (Treiberinstallation, IONOS-API-Stopp): Git-Historie, Commit `36eccfe`.*
