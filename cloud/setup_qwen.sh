#!/bin/bash
# Step 3 (Qwen3.5-0.8B + memory add-on): one-command setup on a fresh Runpod Pod (1 x H100 SXM, "Runpod PyTorch"
# template, volume disk at /workspace), then the queue scripts/run_qwen.py in tmux session "queue".
#
#   bash /workspace/smlm-cloud-kit/setup_qwen.sh          (run inside tmux)
#
# As setup_dense.sh: watchdog at the cost cap, keys, repository. Then: Python environment .venv-qwen (PyTorch CUDA,
# transformers 5.18.0, lm-eval 0.4.13, flash-linear-attention if it installs), Qwen3.5-0.8B at the pinned revision
# (sha256 of weights and LICENSE checked), Qwen-tokenised data from the storage box (sha256 against
# cloud/qwen_data_sha256.txt), tests (add-on tests, kernel tests, real-model check: gate 0 bit-identical), queue.
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
CAP_S=$(python3 -c "print(int(${SMLM_COST_CAP_USD:-14} / ${SMLM_PRICE_USD_H:-3.49} * 3600))")
UP_S=$(ps -o etimes= -p 1 | tr -d ' ')
if [ ! -f /tmp/smlm_watchdog.pid ] || ! kill -0 "$(cat /tmp/smlm_watchdog.pid)" 2>/dev/null; then
  nohup bash -c "sleep $((CAP_S - UP_S - 360)); echo \"\$(date) watchdog: cost cap reached\"; bash $KIT/stop_pod.sh" \
    > "$STATE/watchdog.log" 2>&1 &
  echo $! > /tmp/smlm_watchdog.pid
  say "watchdog armed: Pod stops at uptime $((CAP_S - 360)) s (cap ${SMLM_COST_CAP_USD:-14} \$ at ${SMLM_PRICE_USD_H:-3.49} \$/h, now $UP_S s)"
fi

mountpoint -q /workspace || [ -d /workspace ] || fail "/workspace missing"
[ "$(df -P /workspace | tail -1 | awk '{print $4}')" -gt $((25 * 1024 * 1024)) ] \
  || fail "less than 25 GB free on /workspace (environments ~10 GB, model 1.8 GB, data 0.3 GB, add-ons ~5 GB)"

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
if ! is_done venv_qwen; then
  python3 -m venv .venv-qwen
  .venv-qwen/bin/pip install -q --upgrade pip
  .venv-qwen/bin/pip install -q torch --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
  .venv-qwen/bin/pip install -q numpy pandas matplotlib pyarrow pytest "transformers==5.18.0" "lm_eval==0.4.13" \
    accelerate huggingface_hub
  .venv-qwen/bin/pip install -q "flash-linear-attention==0.5.2" || say "flash-linear-attention not installed (PyTorch fallback)"
  done_ venv_qwen
fi
# fla 0.5.2 refuses the Gated-DeltaNet backward on Hopper with Triton >= 3.4, < 3.7.1 (wrong results, fla issue #640);
# the PyTorch wheel brings an older Triton, so upgrade it (our own Triton kernels are re-checked by the tests below)
if .venv-qwen/bin/python -c "import fla" 2>/dev/null && ! is_done triton_upgrade; then
  .venv-qwen/bin/pip install -q "triton>=3.7.1,<3.8" && done_ triton_upgrade || say "Triton upgrade failed"
fi
.venv-qwen/bin/python -c "import triton; print('triton', triton.__version__)"
.venv-qwen/bin/python -c "import torch, transformers; assert torch.cuda.is_available(); print('torch', torch.__version__, 'transformers', transformers.__version__, torch.cuda.get_device_name(0))"
.venv-qwen/bin/python -c "import fla; print('fla', fla.__version__)" || say "fla not importable: PyTorch fallback for Gated DeltaNet"

# ---- 5 model (pinned revision) and data (storage box, sha256)
QWEN_DIR=/workspace/models/Qwen3.5-0.8B
if ! is_done model; then
  .venv-qwen/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-0.8B', revision='2fc06364715b967f1860aea9cf38778875588b17', local_dir='$QWEN_DIR')"
  (cd $QWEN_DIR && echo "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696  model.safetensors-00001-of-00001.safetensors
bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a  LICENSE" | sha256sum -c) || fail "Qwen files differ from the pinned ones"
  done_ model
fi
if ! is_done data_qwen; then
  mkdir -p data
  rsync -rt --partial storagebox:smlm/data/qwen_wiki data/ || true
  (cd data && sha256sum -c --quiet ../cloud/qwen_data_sha256.txt) || fail "Qwen data from the storage box: sha256 mismatch"
  done_ data_qwen
fi

# ---- 6 tests (GPU): add-on tests, row-sparse table, real model gate-0 check (exercises all Triton kernels at d = 1024)
if ! is_done tests_qwen2; then
  if ! .venv-qwen/bin/python -m pytest -q tests/test_qwen_memory.py tests/test_sparse_values.py \
       > "$STATE/tests_qwen.log" 2>&1; then
    tail -40 "$STATE/tests_qwen.log"; fail "tests (see $STATE/tests_qwen.log)"
  fi
  tail -2 "$STATE/tests_qwen.log"
  QWEN_DIR=$QWEN_DIR .venv-qwen/bin/python scripts/check_qwen_addon_gpu.py > "$STATE/check_qwen.log" 2>&1 \
    || { tail -20 "$STATE/check_qwen.log"; fail "real-model add-on check"; }
  tail -5 "$STATE/check_qwen.log"
  done_ tests_qwen2
fi

# ---- 7 queue (every script must at least compile: a syntax error once left a Pod idle for 2 h)
for f in scripts/*.py smlm/*.py; do .venv-qwen/bin/python -m py_compile "$f" || fail "syntax error in $f"; done
bash "$KIT/stop_pod.sh" --check || say "WARNING: Runpod API not usable - the Pod will NOT stop by itself"
trap - ERR
if tmux has-session -t queue 2>/dev/null; then
  say "queue already running (tmux attach -t queue)"
else
  tmux new-session -d -s queue "cd $REPO_DIR && set -a && . $KIT/cloud.env && set +a && QWEN_DIR=$QWEN_DIR SMLM_KIT=$KIT .venv-qwen/bin/python scripts/run_qwen.py 2>&1 | tee -a runs/qwen_queue_stdout.log"
  say "queue started: tmux attach -t queue   (log: $REPO_DIR/runs/qwen_cloud/queue.log)"
  sleep 90
  pgrep -f "scripts/run_qwen.py" >/dev/null || { trap 'fail "line $LINENO"' ERR; fail "queue died right after the start (see runs/*queue_stdout.log)"; }
  say "queue alive after 90 s"
fi
