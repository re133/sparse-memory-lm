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
set -euo pipefail
if [ "$(id -u)" != "0" ]; then echo "please run as root"; exit 1; fi
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SMLM_KIT="$KIT"
. "$KIT/cloud.env"
REPO_DIR="${REPO_DIR:-/workspace/AngryAnt}"
STATE=/workspace/.smlm-setup
mkdir -p "$STATE"
LOG="$STATE/setup.log"
exec > >(tee -a "$LOG") 2>&1
say() { echo "$(date '+%F %T') [setup] $*"; }
notify() { [ -n "${NTFY_TOPIC:-}" ] && curl -s -m 20 -H "Title: SMLM Cloud" -d "$1" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null || true; }
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
  notify "Setup FEHLGESCHLAGEN: $* - der Pod stoppt sich"
  bash "$KIT/stop_pod.sh" || notify "Pod konnte sich NICHT selbst stoppen - bitte in der Runpod-Konsole stoppen!"
  exit 1
}
trap 'fail "line $LINENO"' ERR

say "start (kit $KIT, repo $REPO_DIR)"

# ---- 0 independent watchdog: stop the Pod when its uptime reaches the cost cap (whatever else happens)
CAP_S=$(python3 -c "print(int(${SMLM_COST_CAP_USD:-18} / ${SMLM_PRICE_USD_H:-3.49} * 3600))")
UP_S=$(ps -o etimes= -p 1 | tr -d ' ')
if [ ! -f /tmp/smlm_watchdog.pid ] || ! kill -0 "$(cat /tmp/smlm_watchdog.pid)" 2>/dev/null; then
  nohup bash -c "sleep $((CAP_S - UP_S - 360)); echo \"\$(date) watchdog: cost cap reached\"; bash $KIT/stop_pod.sh" \
    > "$STATE/watchdog.log" 2>&1 &
  echo $! > /tmp/smlm_watchdog.pid
  say "watchdog armed: Pod stops at uptime $((CAP_S - 360)) s (cap ${SMLM_COST_CAP_USD:-18} \$ at ${SMLM_PRICE_USD_H:-3.49} \$/h, now $UP_S s)"
fi

mountpoint -q /workspace || [ -d /workspace ] || fail "/workspace missing"
[ "$(df -P /workspace | tail -1 | awk '{print $4}')" -gt $((25 * 1024 * 1024)) ] \
  || fail "less than 25 GB free on /workspace (environment ~8 GB, data 1.2 GB, checkpoints ~4 GB)"

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
git -C "$REPO_DIR" config user.name "${GIT_NAME:-leon}"
git -C "$REPO_DIR" config user.email "${GIT_EMAIL:-you@example.com}"
cd "$REPO_DIR"
git pull -q --ff-only || true
say "repo at $(git rev-parse --short HEAD)"

# ---- 4 Python environment (on the volume)
if ! is_done venv; then
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q torch --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
  .venv/bin/pip install -q numpy pandas matplotlib pyarrow tiktoken pytest
  done_ venv
fi
.venv/bin/python - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not available"
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0),
      round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), "GiB")
EOF

# ---- 5 data: token files from the storage box, byte-identical to home (sha256)
if ! is_done data; then
  mkdir -p data
  # -rt instead of -a: /workspace is a network file system that refuses chown (rsync exit 23 although every byte
  # arrived); whether the data is right is decided by sha256 alone
  rsync -rt --partial storagebox:smlm/data/wikipedia_en_gpt2 storagebox:smlm/data/wikitext103_gpt2 data/ || true
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
  if ! .venv/bin/python -m pytest -q tests/test_dense.py tests/test_stage1b.py > "$STATE/tests_gpu.log" 2>&1; then
    tail -40 "$STATE/tests_gpu.log"; fail "tests (see $STATE/tests_gpu.log)"
  fi
  tail -2 "$STATE/tests_gpu.log"
  done_ tests
fi

# ---- 7 queue (every script must at least compile: a syntax error once left a Pod idle for 2 h)
for f in scripts/*.py smlm/*.py; do .venv/bin/python -m py_compile "$f" || fail "syntax error in $f"; done
# (preflight + budget gate + runs + backup are in run_dense.py)
bash "$KIT/stop_pod.sh" --check || say "WARNING: Runpod API not usable - the Pod will NOT stop by itself"
trap - ERR
if tmux has-session -t queue 2>/dev/null; then
  say "queue already running (tmux attach -t queue)"
else
  tmux new-session -d -s queue "cd $REPO_DIR && set -a && . $KIT/cloud.env && set +a && SMLM_KIT=$KIT .venv/bin/python scripts/run_dense.py 2>&1 | tee -a runs/cloud_dense_queue_stdout.log"
  say "queue started: tmux attach -t queue   (log: $REPO_DIR/runs/cloud_dense/queue.log)"
  sleep 90
  pgrep -f "scripts/run_dense.py" >/dev/null || { trap 'fail "line $LINENO"' ERR; fail "queue died right after the start (see runs/*queue_stdout.log)"; }
  say "queue alive after 90 s"
fi
