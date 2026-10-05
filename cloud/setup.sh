#!/bin/bash
# One-command setup on a fresh Runpod Pod (1 x H200, "Runpod PyTorch" template, volume disk at /workspace),
# then the queue.
#
#   bash /workspace/smlm-cloud-kit/setup.sh          (run inside tmux; see docs/notes/CLOUD.md)
#
# Everything that must survive a Pod stop lives on the volume /workspace: repository, Python environment, data,
# checkpoints, setup state. The container disk (apt packages, ~/.ssh) is reset on every start, so those steps run
# again each time (cheap). Stages that are done are skipped, so the script can simply be started again:
#   1 system packages (every start)   2 GPU check (drivers come with Runpod)
#   3 GitHub deploy key (every start) + clone   4 Python environment (PyTorch CUDA wheels, Triton)
#   5 data: download at pinned revisions, rebuild, sha256 check (byte-identical to home)
#   6 all tests (GPU + CPU interpreter) - only if green:
#   7 preflight: every configuration (B-1M, B-4M, B-16M) for 1 M tokens on this card (compile, VRAM peak,
#     final evaluation, checkpoint save, inference) - outputs deleted afterwards
#   8 Runpod API check, then the queue in tmux session "queue" (scripts/run_cloud.py)
# On any failure: log pushed to GitHub (if possible), phone notification (ntfy, optional), Pod stopped through
# the Runpod API (an idle H200 keeps costing money). Nothing is deleted.
set -euo pipefail
if [ "$(id -u)" != "0" ]; then echo "please run as root, see docs/notes/CLOUD.md"; exit 1; fi
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SMLM_KIT="$KIT"
. "$KIT/cloud.env"
REPO_DIR="${REPO_DIR:-/workspace/AngryAnt}"
STATE=/workspace/.smlm-setup
mkdir -p "$STATE"
LOG="$STATE/setup.log"
exec > >(tee -a "$LOG") 2>&1
say() { echo "$(date '+%F %T') [setup] $*"; }
notify() { [ -n "${NTFY_TOPIC:-}" ] && curl -s -m 20 -d "$1" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null || true; }
done_() { touch "$STATE/$1.done"; }
is_done() { [ -f "$STATE/$1.done" ]; }

fail() {
  trap - ERR
  say "FAILED: $*"
  if [ -d "$REPO_DIR/.git" ]; then
    mkdir -p "$REPO_DIR/runs/cloud"
    cp "$LOG" "$REPO_DIR/runs/cloud/setup_failed.log" || true
    (cd "$REPO_DIR" && git add -f runs/cloud/setup_failed.log && git commit -qm "Cloud setup failed: $*" \
      && git pull -q --rebase --autostash origin main && git push -q origin HEAD:main) || true
  fi
  notify "SMLM cloud setup FAILED: $* - stopping the Pod"
  bash "$KIT/stop_pod.sh" || notify "SMLM: Pod could NOT be stopped automatically - stop it in the Runpod console!"
  exit 1
}
trap 'fail "line $LINENO"' ERR

say "start (kit $KIT, repo $REPO_DIR)"
mountpoint -q /workspace || [ -d /workspace ] || fail "/workspace missing - the Pod needs a volume disk at /workspace"
[ "$(df -P /workspace | tail -1 | awk '{print $4}')" -gt $((80 * 1024 * 1024)) ] \
  || fail "less than 80 GB free on /workspace (data 13 GB, environment ~8 GB, checkpoints ~35 GB)"

# ---- 1 system packages (container disk: again after every Pod start)
if ! command -v tmux >/dev/null || ! command -v rsync >/dev/null || ! command -v git >/dev/null; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -q
  apt-get install -yq git tmux rsync curl python3-venv python3-pip python3-dev build-essential
fi

# ---- 2 GPU (Runpod provides the driver)
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || fail "no NVIDIA GPU visible (nvidia-smi)"

# ---- 3 GitHub deploy key (container disk: again after every start) + repository on the volume
install -m 700 -d /root/.ssh
install -m 600 "$KIT/deploy_key" /root/.ssh/smlm_deploy
grep -q "smlm_deploy" /root/.ssh/config 2>/dev/null || cat >> /root/.ssh/config <<EOF
Host github.com
  IdentityFile /root/.ssh/smlm_deploy
  IdentitiesOnly yes
EOF
grep -q "^github.com" /root/.ssh/known_hosts 2>/dev/null || ssh-keyscan -t ed25519 github.com >> /root/.ssh/known_hosts 2>/dev/null
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

# ---- 7 preflight on this card: all three configurations, 1 M tokens each
if ! is_done preflight; then
  rm -rf runs/cloud_dryrun
  if ! SMLM_CLOUD_DRYRUN=1 .venv/bin/python scripts/run_cloud.py > "$STATE/preflight.log" 2>&1 \
     || ! grep -q "QUEUE DONE" "$STATE/preflight.log"; then
    tail -40 "$STATE/preflight.log"; fail "preflight (see $STATE/preflight.log)"
  fi
  for r in B-1M-s0 B-4M-s0 B-16M-s0; do
    st=$(.venv/bin/python -c "import json;i=json.load(open('runs/cloud_dryrun/$r/run-info.json'));print(i['status'], round(i['results']['peak_train_vram_gib'],1), 'GiB peak')" 2>/dev/null || echo "missing")
    say "preflight $r: $st"
    case "$st" in done*) ;; *) fail "preflight $r: $st";; esac
  done
  rm -rf runs/cloud_dryrun
  done_ preflight
fi

# ---- 8 queue
bash "$KIT/stop_pod.sh" --check || say "WARNING: Runpod API not usable - the Pod will NOT stop by itself at the end"
trap - ERR
if tmux has-session -t queue 2>/dev/null; then
  say "queue already running (tmux attach -t queue)"
else
  # cloud.env is sourced (exported) inside the queue session, so SMLM_MAX_HOURS / SMLM_STALL_MIN / NTFY_TOPIC
  # reach run_cloud.py (a plain `.` above only sets shell variables of this script)
  tmux new-session -d -s queue "cd $REPO_DIR && set -a && . $KIT/cloud.env && set +a && SMLM_KIT=$KIT .venv/bin/python scripts/run_cloud.py 2>&1 | tee -a runs/cloud_queue_stdout.log"
  say "queue started: tmux attach -t queue   (log: $REPO_DIR/runs/cloud/queue.log)"
fi
notify "SMLM cloud: setup done, tests green, preflight ok, queue running"
