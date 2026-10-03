# Cloud-Läufe B-1M / B-4M / B-16M auf IONOS (H200-S): Anleitung

Kurz: Maschine mieten → Starter-Paket hochkopieren → ein Befehl → warten (Handy-Nachricht) → die Maschine
stoppt sich selbst → Ergebnisse per `git pull`, Checkpoints per `rsync` holen → Maschine löschen.

**Was läuft:** B-1M (Kontrolle), B-4M, B-16M mit denselben Daten, Einstellungen und demselben Val-Set wie
B-1M-sparse s0. Je 500 M Tokens, Triton-Kernels. Die Kriterien stehen in REPORT.md (Abschnitt
„Kriterien für die Cloud-Läufe“). Nach jedem Lauf landet ein Zwischenstand in REPORT.md auf GitHub.

## 0. Einmalig vorbereitet (schon erledigt)

- **Privates Repo:** `github.com/re133/sparse-memory-lm`, Code und Ergebnisse. Die VM pusht mit einem
  Deploy-Key, der nur für dieses Repo gilt.
- **Starter-Paket `~/smlm-cloud-kit/`** auf deinem PC:
  - `setup.sh` und `ionos_stop.sh`
  - `deploy_key` (privat, nicht weitergeben)
  - `cloud.env`: Hier fehlen noch deine IONOS-Angaben (Schritt 2).

## 1. Maschine mieten (IONOS DCD)

1. **Rechenzentrum:** DCD → *Virtual Data Centers* → ein Rechenzentrum in **de/fra/2** (Frankfurt) anlegen
   oder öffnen. Cloud GPU VMs gibt es nur dort.
2. **GPU-VM anlegen:**
   - Typ **Cloud GPU VM**, Vorlage **H200-S**: 1 × H200 141 GB, 15 vCPU, 267 GiB RAM, 1 TB Speicher.
   - Image: **Ubuntu** (aktuelle LTS).
   - Deinen **SSH-Public-Key** hinterlegen.
   - **Öffentliche IP** über eine Netzwerkkarte am Internetzugang.
3. **Freischaltung:** Standardmäßig ist genau eine H200-S-VM erlaubt; größere Vorlagen nur über den
   Support.
4. **Provisionieren** und warten, bis die VM läuft. IP-Adresse notieren.
5. **Falls die DCD die GPU-VM nicht anbietet:** Laut einer Doku-Seite lassen sich GPU-VMs nur über die
   Cloud-API verwalten. Dann so:
   `POST https://api.ionos.com/cloudapi/v6/datacenters/<DC-ID>/servers` mit
   `"type": "GPU"`, `"templateUuid"` der H200-S-Vorlage und einem Volume mit `"imageAlias": "ubuntu:latest"`
   (IONOS-Doku „Create a Cloud GPU VM“).

## 2. `cloud.env` ausfüllen (auf deinem PC)

```bash
nano ~/smlm-cloud-kit/cloud.env
```

- **`IONOS_TOKEN`:**
  - Erzeugen in DCD → *Management* → *Token Manager* → *Generate token*, kurze Laufzeit wählen
    (z. B. 2 Tage).
  - Damit stoppt sich die VM am Ende selbst. **Ein `shutdown` im Betriebssystem stoppt die Abrechnung bei
    IONOS nicht.**
- **`IONOS_DATACENTER_ID` und `IONOS_SERVER_ID`:** Beide UUIDs stehen im DCD-Inspector, wenn du die
  GPU-VM anklickst.
- **`NTFY_TOPIC` (optional, empfohlen):**
  - Einen langen zufälligen Namen ausdenken und in der **ntfy**-App auf dem Handy abonnieren.
  - Dann kommen Nachrichten bei Start, nach jedem Lauf, bei Fehlern und am Ende.

## 3. Hochladen und starten

```bash
scp -r ~/smlm-cloud-kit root@<IP>:
ssh root@<IP>
tmux new -s setup 'bash ~/smlm-cloud-kit/setup.sh'
```

Mit `Strg-b`, dann `d` löst du dich von der Sitzung; die Verbindung darfst du trennen.

**Was `setup.sh` macht** (ca. 30–45 min, jeder Schritt wird bei einem Neustart übersprungen, wenn er schon
fertig ist):

1. **Pakete** installieren.
2. **NVIDIA-Treiber** installieren (das IONOS-Image hat keinen).
   - Falls nötig startet die VM **einmal neu** und macht danach von selbst weiter.
   - Wieder ansehen: `ssh root@<IP>` und `tmux attach -t setup`.
3. **Repo** mit dem Deploy-Key klonen.
4. **Python-Umgebung** mit PyTorch (CUDA) und Triton einrichten.
5. **Daten:**
   - WikiText-103 und Wikipedia 20231101.en von Hugging Face laden, in festgepinnten Versionen (≈ 11 GB).
   - Neu tokenisieren und per **SHA-256 prüfen, dass alles bytegleich mit deinem PC ist**.
6. **Alle Tests**: GPU und CPU-Interpreter. **Nur wenn alles grün ist, geht es weiter.**
7. **IONOS-API prüfen**, dann die Warteschlange in der tmux-Sitzung `queue` starten.

**Wenn etwas fehlschlägt:**
- Das Log wird nach GitHub gepusht (`runs/cloud/setup_failed.log`), du bekommst eine Nachricht, und die
  VM stoppt sich.
- Neu starten in der DCD, einloggen, Log ansehen: `/root/.smlm-setup/setup.log`.
- Danach einfach `bash ~/smlm-cloud-kit/setup.sh` erneut starten.

## 4. Während der Läufe

- **Zwischenstand:** nach jedem Lauf in **REPORT.md** auf GitHub (Abschnitt „Zwischenstand Cloud“),
  dazu eine ntfy-Nachricht.
- **Live:** `ssh root@<IP>`, dann `tmux attach -t queue` oder
  `tail -f /root/AngryAnt/runs/cloud/queue.log`.
- **GPU-Werte:** Temperatur, Leistung und Takt alle 10 s in `runs/cloud/<lauf>/gpu_thermal.csv`.
- **Sicherungen:**
  - Ein Lauf, dessen Logs sich 30 min nicht ändern, wird beendet; die Warteschlange macht weiter.
  - Nach 12 h insgesamt wird alles gestoppt.

## 5. Ende

- **Am Ende der Warteschlange:**
  - Eine Prüfsummenliste aller Checkpoints wird gepusht (`runs/cloud/checkpoints.sha256`).
  - Die VM **stoppt sich über die IONOS-API**: Rechenkosten aus, Festplatte mit den Checkpoints bleibt.
- **In der DCD kontrollieren**, dass die VM wirklich gestoppt ist. Wenn nicht (Nachricht „could NOT be
  stopped“): *Power → Stop* von Hand.

## 6. Ergebnisse und Checkpoints holen

1. **Ergebnisse** (klein, über GitHub):

   ```bash
   cd /mnt/sandisk/Sparse-Memory-LM/AngryAnt && git pull
   ```

2. **Checkpoints** (≈ 35 GB: B-1M 1,7 GB, B-4M ≈ 6,6 GB, B-16M ≈ 26 GB):
   - VM in der DCD wieder **starten**. Sie bekommt dabei **eine neue IP**.
   - Dann holen und prüfen:

   ```bash
   rsync -avP --partial root@<NEUE-IP>:/root/AngryAnt/runs/cloud/ runs/cloud/
   sha256sum -c runs/cloud/checkpoints.sha256
   ```

   Alle Zeilen müssen `OK` zeigen. Erst dann weiter.

## 7. Maschine löschen

- In der DCD die **VM und ihr Volume löschen**, dazu eine reservierte IP, falls vorhanden. Gestoppt kostet
  die 1-TB-Platte weiter, laut Preisliste 0,15 €/GB/30 Tage, also ≈ 5 €/Tag.
- **Token widerrufen** (Token Manager).
- **Deploy-Key entfernen:**
  `gh repo deploy-key list -R re133/sparse-memory-lm`, dann `gh repo deploy-key delete <ID> -R re133/sparse-memory-lm`.

## Kosten (H200-S: 3,00 €/h inkl. GPU, CPU, RAM und 1 TB)

| Abschnitt | Dauer (Schätzung) | Kosten |
|---|---|---|
| Setup (Treiber, Python, Daten, Tests) | 30–45 min | 1,50–2,30 € |
| B-1M | 25–45 min | 1,30–2,30 € |
| B-4M | 30–55 min | 1,50–2,80 € |
| B-16M | 40–75 min | 2,00–3,80 € |
| Checkpoints holen (VM läuft wieder) | 20–60 min, je nach Leitung | 1,00–3,00 € |
| Platte, solange die VM nur gestoppt ist | pro Tag | ≈ 5 € |
| **Summe bei zügigem Löschen** | **≈ 2,5–4,5 h** | **≈ 8–15 €** |

Die Laufzeiten sind von der RX 9070 hochgerechnet (H200: ≈ 7,5× Bandbreite, ≈ 5× Rechenleistung; das
kleine Modell lastet die H200 aber nicht aus) und **bis Faktor 2 unsicher**. Die Grenze von 12 h kostet
höchstens 36 €.
