#!/bin/bash
# One-command setup of a fresh Ubuntu machine with an NVIDIA GPU (IONOS Cloud GPU VM H200-S), then the queue.
#
#   bash ~/smlm-cloud-kit/setup.sh          (run inside tmux; see CLOUD.md)
#
# Stages (each one is skipped when already done, so the script can simply be started again):
#   1 system packages            2 NVIDIA driver (reboots once if needed and continues automatically)
#   3 GitHub deploy key + clone  4 Python environment (PyTorch CUDA wheels, Triton)
#   5 data: download at pinned revisions, rebuild, sha256 check (byte-identical to home)
#   6 all tests (GPU + CPU interpreter) - only if green:
#   7 IONOS API check, then the queue in tmux session "queue" (scripts/run_cloud.py)
# On any failure: log pushed to GitHub (if possible), phone notification (ntfy, optional), VM stopped via
# the IONOS API (so an idle H200 does not keep costing money). Nothing is deleted.
set -euo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SMLM_KIT="$KIT"
. "$KIT/cloud.env"
REPO_DIR="${REPO_DIR:-/root/AngryAnt}"
STATE=/root/.smlm-setup
mkdir -p "$STATE"
LOG="$STATE/setup.log"
exec > >(tee -a "$LOG") 2>&1
say() { echo "$(date '+%F %T') [setup] $*"; }
notify() { [ -n "${NTFY_TOPIC:-}" ] && curl -s -m 20 -d "$1" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null || true; }
done_() { touch "$STATE/$1.done"; }
is_done() { [ -f "$STATE/$1.done" ]; }

fail() {
  say "FAILED: $*"
  if [ -d "$REPO_DIR/.git" ]; then
    mkdir -p "$REPO_DIR/runs/cloud"
    cp "$LOG" "$REPO_DIR/runs/cloud/setup_failed.log" || true
    (cd "$REPO_DIR" && git add -f runs/cloud/setup_failed.log && git commit -qm "Cloud setup failed: $*" \
      && git push -q origin HEAD) || true
  fi
  notify "SMLM cloud setup FAILED: $* - stopping the VM"
  bash "$KIT/ionos_stop.sh" || notify "SMLM: VM could NOT be stopped automatically - stop it in the DCD!"
  exit 1
}
trap 'fail "line $LINENO"' ERR

say "start (kit $KIT, repo $REPO_DIR)"

# ---- 1 system packages
if ! is_done packages; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -q
  apt-get install -yq git tmux rsync curl jq python3-venv python3-pip python3-dev build-essential \
    ubuntu-drivers-common pciutils
  done_ packages
fi

# ---- 2 NVIDIA driver (not included in the IONOS images)
if ! nvidia-smi >/dev/null 2>&1; then
  if ! is_done driver; then
    say "installing the NVIDIA driver"
    ubuntu-drivers install --gpgpu || apt-get install -yq nvidia-driver-570-server-open nvidia-utils-570-server
    done_ driver
  fi
  modprobe nvidia 2>/dev/null || true
  if ! nvidia-smi >/dev/null 2>&1; then
    if is_done rebooted; then fail "nvidia-smi does not work after the reboot"; fi
    say "driver needs a reboot - setup continues automatically afterwards (tmux attach -t setup)"
    cat > /etc/systemd/system/smlm-setup.service <<EOF
[Unit]
Description=SMLM cloud setup (continue after driver reboot)
After=network-online.target
Wants=network-online.target
[Service]
Type=forking
ExecStart=/usr/bin/tmux new-session -d -s setup /bin/bash $KIT/setup.sh
[Install]
WantedBy=multi-user.target
EOF
    systemctl enable smlm-setup.service
    done_ rebooted
    trap - ERR
    reboot
    exit 0
  fi
fi
systemctl disable smlm-setup.service 2>/dev/null || true
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# ---- 3 GitHub deploy key + repository
if ! is_done repo; then
  install -m 700 -d /root/.ssh
  install -m 600 "$KIT/deploy_key" /root/.ssh/smlm_deploy
  ssh-keyscan -t ed25519 github.com >> /root/.ssh/known_hosts 2>/dev/null
  cat >> /root/.ssh/config <<EOF
Host github.com
  IdentityFile /root/.ssh/smlm_deploy
  IdentitiesOnly yes
EOF
  [ -d "$REPO_DIR/.git" ] || git clone -q "$GIT_REMOTE" "$REPO_DIR"
  git -C "$REPO_DIR" config user.name "${GIT_NAME:-leon}"
  git -C "$REPO_DIR" config user.email "${GIT_EMAIL:-you@example.com}"
  done_ repo
fi
cd "$REPO_DIR"
git pull -q --ff-only || true
say "repo at $(git rev-parse --short HEAD)"

# ---- 4 Python environment
if ! is_done venv; then
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q torch --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
  .venv/bin/pip install -q numpy pandas matplotlib pyarrow tiktoken pytest
  done_ venv
fi
.venv/bin/python - <<'EOF'
import torch, triton
assert torch.cuda.is_available(), "CUDA not available"
print("torch", torch.__version__, "cuda", torch.version.cuda, "triton", triton.__version__,
      torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), "GiB")
EOF

# ---- 5 data (byte-identical to home, checked by sha256)
if ! is_done data; then
  .venv/bin/python cloud/fetch_data.py
  done_ data
fi

# ---- 6 tests: everything must be green
if ! is_done tests; then
  if ! .venv/bin/python -m pytest -q tests > "$STATE/tests_gpu.log" 2>&1; then
    tail -40 "$STATE/tests_gpu.log"; fail "GPU tests (see $STATE/tests_gpu.log)"
  fi
  tail -2 "$STATE/tests_gpu.log"
  if ! TRITON_INTERPRET=1 .venv/bin/python -m pytest -q tests/test_kernels.py > "$STATE/tests_cpu.log" 2>&1; then
    tail -40 "$STATE/tests_cpu.log"; fail "CPU interpreter tests (see $STATE/tests_cpu.log)"
  fi
  tail -2 "$STATE/tests_cpu.log"
  done_ tests
fi

# ---- 7 queue
bash "$KIT/ionos_stop.sh" --check || say "WARNING: IONOS API not usable - the VM will NOT stop by itself at the end"
trap - ERR
if tmux has-session -t queue 2>/dev/null; then
  say "queue already running (tmux attach -t queue)"
else
  tmux new-session -d -s queue "cd $REPO_DIR && SMLM_KIT=$KIT NTFY_TOPIC=${NTFY_TOPIC:-} .venv/bin/python scripts/run_cloud.py 2>&1 | tee -a runs/cloud_queue_stdout.log"
  say "queue started: tmux attach -t queue   (log: $REPO_DIR/runs/cloud/queue.log)"
fi
notify "SMLM cloud: setup done, tests green, queue running"
