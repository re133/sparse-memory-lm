#!/bin/bash
# AMD Instinct MI300X check: one-command setup on a Runpod Pod with the "Runpod Pytorch ROCm" template (volume disk at
# /workspace), then scripts/run_amd.py in tmux session "queue".
#
#   bash /workspace/smlm-cloud-kit/setup_amd.sh          (run inside tmux)
#
# The code is copied to $REPO_DIR beforehand (rsync from home; nothing is cloned or pushed). Python 3.12 via uv,
# PyTorch for ROCm (newest wheel, fallback to the ROCm 6.4 wheel if it cannot see the GPU), token files from the
# storage box (sha256), every script compiled, then the queue. Same watchdog at the cost cap as the other setups.
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
  rsync -rt "$LOG" storagebox:smlm/runs/amd_mi300x_setup_failed.log 2>/dev/null || true
  notify "MI300X-Setup FEHLGESCHLAGEN: $* - der Pod stoppt sich"
  bash "$KIT/stop_pod.sh" || notify "Pod konnte sich NICHT selbst stoppen - bitte in der Runpod-Konsole stoppen!"
  exit 1
}
trap 'fail "line $LINENO"' ERR

say "start (kit $KIT, repo $REPO_DIR)"

# ---- 0 independent watchdog: stop the Pod when its uptime reaches the cost cap (whatever else happens)
CAP_S=$(python3 -c "print(int(${SMLM_COST_CAP_USD:-6} / ${SMLM_PRICE_USD_H:-2.39} * 3600))")
UP_S=$(ps -o etimes= -p 1 | tr -d ' ')
if [ ! -f /tmp/smlm_watchdog.pid ] || ! kill -0 "$(cat /tmp/smlm_watchdog.pid)" 2>/dev/null; then
  nohup bash -c "sleep $((CAP_S - UP_S - 360)); echo \"\$(date) watchdog: cost cap reached\"; bash $KIT/stop_pod.sh" \
    > "$STATE/watchdog.log" 2>&1 &
  echo $! > /tmp/smlm_watchdog.pid
  say "watchdog armed: Pod stops at uptime $((CAP_S - 360)) s (cap ${SMLM_COST_CAP_USD:-6} \$ at ${SMLM_PRICE_USD_H:-2.39} \$/h, now $UP_S s)"
fi

[ "$(df -P /workspace | tail -1 | awk '{print $4}')" -gt $((20 * 1024 * 1024)) ] \
  || fail "less than 20 GB free on /workspace (environment ~10 GB, data 1.2 GB)"

# ---- 1 system packages (container disk: again after every Pod start)
if ! command -v tmux >/dev/null || ! command -v rsync >/dev/null || ! command -v curl >/dev/null; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -q
  apt-get install -yq tmux rsync curl git
fi

# ---- 2 GPU
rocm-smi --showproductname || amd-smi static -g 0 || fail "no AMD GPU visible (rocm-smi / amd-smi)"

# ---- 3 storage box key (container disk: again after every start); the code was copied to $REPO_DIR
install -m 700 -d /root/.ssh
install -m 600 "$KIT/storagebox_key" /root/.ssh/storagebox_smlm
install -m 600 "$KIT/known_hosts_storagebox" /root/.ssh/known_hosts_storagebox
grep -q "Host storagebox" /root/.ssh/config 2>/dev/null || cat >> /root/.ssh/config <<SSHCFG
Host storagebox
  HostName ${STORAGEBOX_HOST}
  User ${STORAGEBOX_USER}
  Port 23
  IdentityFile /root/.ssh/storagebox_smlm
  IdentitiesOnly yes
  BatchMode yes
  UserKnownHostsFile /root/.ssh/known_hosts_storagebox
  StrictHostKeyChecking yes
SSHCFG
ssh storagebox ls smlm >/dev/null || fail "storage box not reachable with the key"
[ -f "$REPO_DIR/smlm/model.py" ] || fail "code not found in $REPO_DIR (copy it there first)"
cd "$REPO_DIR"

# ---- 4 Python 3.12 + PyTorch for ROCm (on the volume)
if ! is_done venv_amd; then
  python3 -m pip install -q uv || pip install -q uv
  python3 -m uv venv -q -p 3.12 .venv
  for idx in https://download.pytorch.org/whl/rocm7.1 https://download.pytorch.org/whl/rocm6.4; do
    python3 -m uv pip install -q -p .venv/bin/python --reinstall torch --index-url "$idx" || continue
    if .venv/bin/python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
      say "PyTorch from $idx sees the GPU"
      break
    fi
    say "PyTorch from $idx does not see the GPU - trying the next wheel"
  done
  python3 -m uv pip install -q -p .venv/bin/python numpy pandas matplotlib pyarrow tiktoken pytest
  done_ venv_amd
fi
.venv/bin/python - <<'PYCHECK'
import torch, triton
assert torch.cuda.is_available(), "GPU not visible to PyTorch"
p = torch.cuda.get_device_properties(0)
print("torch", torch.__version__, "hip", torch.version.hip, "triton", triton.__version__, p.name,
      getattr(p, "gcnArchName", ""), round(p.total_memory / 2**30, 1), "GiB")
PYCHECK

# ---- 5 data: token files from the storage box, byte-identical to home (sha256)
if ! is_done data; then
  mkdir -p data
  rsync -rt --partial storagebox:smlm/data/wikipedia_en_gpt2 storagebox:smlm/data/wikitext103_gpt2 data/ || true
  (cd data && grep -E " (wikipedia_en_gpt2|wikitext103_gpt2)/" ../cloud/data_sha256.txt | sha256sum -c --quiet) \
    || fail "data from the storage box: sha256 mismatch"
  done_ data
fi

# ---- 6 queue (every script must at least compile)
for f in scripts/*.py smlm/*.py tests/*.py; do .venv/bin/python -m py_compile "$f" || fail "syntax error in $f"; done
bash "$KIT/stop_pod.sh" --check || say "WARNING: Runpod API not usable - the Pod will NOT stop by itself"
trap - ERR
if tmux has-session -t queue 2>/dev/null; then
  say "queue already running (tmux attach -t queue)"
else
  tmux new-session -d -s queue "cd $REPO_DIR && set -a && . $KIT/cloud.env && set +a && SMLM_KIT=$KIT .venv/bin/python scripts/run_amd.py 2>&1 | tee -a runs/amd_queue_stdout.log"
  say "queue started: tmux attach -t queue   (log: $REPO_DIR/runs/amd_mi300x/queue.log)"
  sleep 90
  pgrep -f "scripts/run_amd.py" >/dev/null || { trap 'fail "line $LINENO"' ERR; fail "queue died right after the start (see runs/amd_queue_stdout.log)"; }
  say "queue alive after 90 s"
fi
