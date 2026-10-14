#!/bin/bash
# Step 1 "Gegenwert der Tabelle": one-command setup on a fresh Runpod Pod (1 x H100 SXM, "Runpod PyTorch" template,
# volume disk at /workspace), then the dense queue (scripts/run_dense.py) in tmux session "queue".
#
#   bash /workspace/smlm-cloud-kit/setup_dense.sh          (run inside tmux)
#
# Differences to setup.sh (B-1M/4M/16M): the token files come from the Hetzner storage box (sha256 against
# cloud/data_sha256.txt; fallback: rebuild from Hugging Face as before), only the dense-relevant tests run, the
# preflight is part of run_dense.py (it also feeds the budget gate), checkpoints are backed up to the storage box,
# and an independent watchdog stops the Pod when its uptime reaches the cost cap.
# Stages that are done are skipped, so the script can simply be started again.
#
# Also on a plain VM (IONOS H200 NVL, prepared by the kit's ionos-cloud-init.yaml: key, packages, driver, reboot):
# detected by the missing Runpod variables (or SMLM_PLATFORM=vm). There nothing stops by itself (if run.env sets
# SMLM_STOP_CMD, the watchdog and a failed setup run it); the messages say so instead of asking for the Runpod console.
# At the end Claude checks the backup on the box from home, then the instance is deleted by hand.
# On a VM the script also notes when the instance was first up (billing_start, for the cost count after a reboot) and
# installs an @reboot entry that runs it again, so a rebooted VM continues its queue (runs resume from their checkpoints).
set -euo pipefail
if [ "$(id -u)" != "0" ]; then echo "please run as root"; exit 1; fi
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SMLM_KIT="$KIT"
. "$KIT/cloud.env"
# optional per-Pod settings (e.g. step 10: SMLM_DENSE_RUNS, SMLM_DENSE_DATA, SMLM_DENSE_OUT, cost cap and price)
[ -f "$KIT/run.env" ] && . "$KIT/run.env"
REPO_DIR="${REPO_DIR:-/workspace/AngryAnt}"
STATE=/workspace/.smlm-setup
mkdir -p "$STATE"
LOG="$STATE/setup.log"
exec > >(tee -a "$LOG") 2>&1
say() { echo "$(date '+%F %T') [setup] $*"; }
notify() { [ -n "${NTFY_TOPIC:-}" ] && curl -s -m 20 -H "Title: SMLM Cloud" -d "$1" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null || true; }
# Runpod puts RUNPOD_POD_ID into every Pod (ssh sessions may not inherit it: PID 1's environment counts too)
if [ -z "${SMLM_PLATFORM:-}" ]; then
  SMLM_PLATFORM=vm
  { [ -n "${RUNPOD_POD_ID:-}" ] || grep -qz '^RUNPOD_POD_ID=' /proc/1/environ 2>/dev/null; } && SMLM_PLATFORM=runpod
fi
export SMLM_PLATFORM
if [ "$SMLM_PLATFORM" = runpod ]; then
  NOSTOP_MSG="Pod konnte sich NICHT selbst stoppen - bitte in der Runpod-Konsole stoppen!"
else
  NOSTOP_MSG="Die Instanz stoppt sich nicht selbst - bitte in der Cloud-Konsole löschen oder stoppen!"
fi
done_() { touch "$STATE/$1.done"; }
is_done() { [ -f "$STATE/$1.done" ]; }

fail() {
  trap - ERR
  say "FAILED: $*"
  if [ -d "$REPO_DIR/.git" ]; then
    mkdir -p "$REPO_DIR/runs/cloud_dense"
    cp "$LOG" "$REPO_DIR/runs/cloud_dense/setup_failed.log" || true
    (cd "$REPO_DIR" && git add -f runs/cloud_dense/setup_failed.log && git commit -qm "Cloud dense setup failed: $*" \
      && git pull -q --rebase --autostash origin main && git push -q origin HEAD:main) || true
  fi
  notify "Setup FEHLGESCHLAGEN: $*"
  bash "$KIT/stop_pod.sh" || notify "$NOSTOP_MSG"
  exit 1
}
trap 'fail "line $LINENO"' ERR

say "start (kit $KIT, repo $REPO_DIR)"

# ---- 0 independent watchdog: stop the Pod when its uptime reaches the cost cap (whatever else happens)
CAP_S=$(python3 -c "print(int(${SMLM_COST_CAP_USD:-18} / ${SMLM_PRICE_USD_H:-3.49} * 3600))")
UP_S=$(ps -o etimes= -p 1 | tr -d ' ')
if [ "$SMLM_PLATFORM" = vm ]; then
  # billing runs since the instance was created: cloud-init rebooted once after the driver, so the first boot counts
  # (birth time of /var/lib/cloud/instances if it lies less than an hour before this boot, as in run_dense.py).
  # Noted once; after a later reboot the count goes on from there instead of starting at zero.
  if [ ! -s "$STATE/billing_start" ]; then
    NOW=$(date +%s); BOOT=$((NOW - UP_S)); START=$BOOT
    BORN=$(stat -c %W /var/lib/cloud/instances 2>/dev/null || echo 0)
    if [ "${BORN:-0}" -gt 0 ] && [ $((BOOT - BORN)) -ge 0 ] && [ $((BOOT - BORN)) -lt 3600 ]; then START=$BORN; fi
    echo "$START" > "$STATE/billing_start"
  fi
  UP_S=$(( $(date +%s) - $(cat "$STATE/billing_start") ))
  # after a reboot: run this script again (stages that are done are skipped, the queue resumes its runs)
  if [ -d /etc/cron.d ] && [ ! -f /etc/cron.d/smlm-setup ]; then
    echo "@reboot root sleep 60 && bash $KIT/setup_dense.sh >> $STATE/reboot.log 2>&1" > /etc/cron.d/smlm-setup
    say "installed /etc/cron.d/smlm-setup (setup + queue again after a reboot)"
  fi
fi
if [ ! -f /tmp/smlm_watchdog.pid ] || ! kill -0 "$(cat /tmp/smlm_watchdog.pid)" 2>/dev/null; then
  # the ntfy topic comes from cloud.env inside the watchdog, so it never shows up in the process list
  nohup bash -c "sleep $((CAP_S - UP_S - 360)); echo \"\$(date) watchdog: cost cap reached\"; bash $KIT/stop_pod.sh \
    || { . $KIT/cloud.env; [ -n \"\$NTFY_TOPIC\" ] && curl -s -m 20 -H 'Title: SMLM Cloud' \
    -d 'Kostendeckel erreicht. $NOSTOP_MSG' \"https://ntfy.sh/\$NTFY_TOPIC\"; }" > "$STATE/watchdog.log" 2>&1 &
  echo $! > /tmp/smlm_watchdog.pid
  say "watchdog armed ($SMLM_PLATFORM): stop or ntfy at uptime $((CAP_S - 360)) s (cap ${SMLM_COST_CAP_USD:-18} ${SMLM_CURRENCY:-\$} at ${SMLM_PRICE_USD_H:-3.49} ${SMLM_CURRENCY:-\$}/h, now $UP_S s)"
fi

mountpoint -q /workspace || [ -d /workspace ] || fail "/workspace missing"
# default for step 1 (environment ~8 GB, data 1.2 GB, checkpoints ~4 GB); a B-16M run with checkpoints needs ~250 GB
# (77 GB checkpoint, the same again while the next one is written, 26 GB model.pt, 3 GB data)
MIN_FREE_GB="${SMLM_MIN_FREE_GB:-25}"
if ! is_done data; then
  [ "$(df -P /workspace | tail -1 | awk '{print $4}')" -gt $((MIN_FREE_GB * 1024 * 1024)) ] \
    || fail "less than $MIN_FREE_GB GB free on /workspace"
fi

# ---- 1 system packages (container disk: again after every Pod start)
if ! command -v tmux >/dev/null || ! command -v rsync >/dev/null || ! command -v git >/dev/null; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -q
  apt-get install -yq git tmux rsync curl python3-venv python3-pip python3-dev build-essential
fi

# ---- 2 GPU
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || fail "no NVIDIA GPU visible (nvidia-smi)"

# ---- 3 keys (container disk: again after every start): GitHub deploy key, storage box key
install -m 700 -d /root/.ssh
install -m 600 "$KIT/deploy_key" /root/.ssh/smlm_deploy
install -m 600 "$KIT/storagebox_key" /root/.ssh/storagebox_smlm
install -m 600 "$KIT/known_hosts_storagebox" /root/.ssh/known_hosts_storagebox
grep -q "smlm_deploy" /root/.ssh/config 2>/dev/null || cat >> /root/.ssh/config <<EOF
Host github.com
  IdentityFile /root/.ssh/smlm_deploy
  IdentitiesOnly yes
EOF
grep -q "Host storagebox" /root/.ssh/config 2>/dev/null || cat >> /root/.ssh/config <<EOF
Host storagebox
  HostName ${STORAGEBOX_HOST}
  User ${STORAGEBOX_USER}
  Port 23
  IdentityFile /root/.ssh/storagebox_smlm
  IdentitiesOnly yes
  BatchMode yes
  UserKnownHostsFile /root/.ssh/known_hosts_storagebox
  StrictHostKeyChecking yes
EOF
grep -q "^github.com" /root/.ssh/known_hosts 2>/dev/null || ssh-keyscan -t ed25519 github.com >> /root/.ssh/known_hosts 2>/dev/null
ssh storagebox ls smlm >/dev/null || fail "storage box not reachable with the key"
[ -d "$REPO_DIR/.git" ] || git clone -q "$GIT_REMOTE" "$REPO_DIR"
git -C "$REPO_DIR" config user.name "${GIT_NAME:?set GIT_NAME in cloud.env}"
git -C "$REPO_DIR" config user.email "${GIT_EMAIL:?set GIT_EMAIL in cloud.env}"
cd "$REPO_DIR"
git pull -q --ff-only || true
say "repo at $(git rev-parse --short HEAD)"

# ---- 4 Python environment (on the volume)
if ! is_done venv; then
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  # pinned: every cloud run so far used 2.11.0+cu128 (and its Triton); a newer wheel could round differently
  .venv/bin/pip install -q "${TORCH_SPEC:-torch==2.11.0}" --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
  .venv/bin/pip install -q numpy pandas matplotlib pyarrow tiktoken pytest
  done_ venv
fi
.venv/bin/python - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not available"
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0),
      round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), "GiB")
EOF
# the exact environment of this run, backed up with its results
mkdir -p "${SMLM_DENSE_OUT:-runs/cloud_dense}"
.venv/bin/pip freeze > "${SMLM_DENSE_OUT:-runs/cloud_dense}/pip_freeze.txt"

# ---- 5 data: token files from the storage box, byte-identical to home (sha256)
if ! is_done data; then
  mkdir -p data
  # -rt instead of -a: /workspace is a network file system that refuses chown (rsync exit 23 although every byte
  # arrived); whether the data is right is decided by sha256 alone
  rsync -rt --partial storagebox:smlm/data/wikipedia_en_gpt2 storagebox:smlm/data/wikitext103_gpt2 data/ || true
  case "${SMLM_DENSE_DATA:-wikipedia}" in
    wikipedia_factk|wikipedia_hop2|wikipedia_hop2d|wikipedia_1500m)
      # steps 10-12: built at home by scripts/make_factk.py / make_hop2.py; 1500m: scripts/prepare_wikipedia.py
      # --train_tokens 1.5e9 (first 505 M tokens = wikipedia). No Hugging Face fallback, a wrong file stops the Pod
      case "$SMLM_DENSE_DATA" in wikipedia_1500m) D=wikipedia_en_gpt2_1500m ;; *) D="${SMLM_DENSE_DATA}_gpt2" ;; esac
      rsync -rt --partial "storagebox:smlm/data/$D" data/ || true
      (cd data && grep " $D/" ../cloud/data_sha256.txt | sha256sum -c --quiet) \
        || fail "$D missing or wrong on the storage box (sha256)"
      say "$D from the storage box, sha256 ok" ;;
  esac
  if (cd data && grep -E " (wikipedia_en_gpt2|wikitext103_gpt2)/" ../cloud/data_sha256.txt | sha256sum -c --quiet); then
    say "data from the storage box, sha256 ok"
  else
    say "storage box data missing or wrong - rebuilding from Hugging Face"
    .venv/bin/python cloud/fetch_data.py
  fi
  done_ data
fi

# ---- 6 tests for the dense models (GPU)
if ! is_done tests; then
  if ! .venv/bin/python -m pytest -q ${SMLM_SETUP_TESTS:-tests/test_dense.py tests/test_stage1b.py} > "$STATE/tests_gpu.log" 2>&1; then
    tail -40 "$STATE/tests_gpu.log"; fail "tests (see $STATE/tests_gpu.log)"
  fi
  tail -2 "$STATE/tests_gpu.log"
  done_ tests
fi

# ---- 7 queue (every script must at least compile: a syntax error once left a Pod idle for 2 h)
for f in scripts/*.py smlm/*.py; do .venv/bin/python -m py_compile "$f" || fail "syntax error in $f"; done
# (preflight + budget gate + runs + backup are in run_dense.py)
if [ "$SMLM_PLATFORM" = runpod ] || [ -n "${SMLM_STOP_CMD:-}" ]; then
  bash "$KIT/stop_pod.sh" --check || say "WARNING: Runpod API not usable - the Pod will NOT stop by itself"
else
  say "plain VM: nothing stops by itself; after the backup check from home the instance is deleted by hand"
fi
trap - ERR
if tmux has-session -t queue 2>/dev/null; then
  say "queue already running (tmux attach -t queue)"
else
  RUNENV=""; [ -f "$KIT/run.env" ] && RUNENV="&& . $KIT/run.env"
  tmux new-session -d -s queue "cd $REPO_DIR && set -a && . $KIT/cloud.env $RUNENV && set +a && SMLM_KIT=$KIT .venv/bin/python scripts/run_dense.py 2>&1 | tee -a runs/cloud_dense_queue_stdout.log"
  say "queue started: tmux attach -t queue   (log: $REPO_DIR/runs/cloud_dense/queue.log)"
  sleep 90
  pgrep -f "scripts/run_dense.py" >/dev/null || { trap 'fail "line $LINENO"' ERR; fail "queue died right after the start (see runs/*queue_stdout.log)"; }
  say "queue alive after 90 s"
fi
