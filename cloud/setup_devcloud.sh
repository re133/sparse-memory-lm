#!/bin/bash
# Prepare an AMD Developer Cloud VM and start the resumable queue in tmux.
#   bash cloud/setup_devcloud.sh [--dry_run] [queue arguments]
#   bash cloud/setup_devcloud.sh --transfer_help
# Use an uploaded checkout by default, or REPO_DIR=/new/path to clone the public repository.
# Python and package caches stay in the checkout; Ubuntu system packages are installed with apt.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(dirname "$SCRIPT_DIR")}"
REPO_DIR="$(realpath -m "$REPO_DIR")"
GIT_REMOTE="${GIT_REMOTE:-https://github.com/re133/sparse-memory-lm.git}"
TORCH_VERSION="${TORCH_VERSION:-2.14.1+rocm7.2}"
DRY_RUN=false
TRANSFER_HELP=false
HAS_ONLY=false
HAS_GPUS=false
HAS_OUT_DIR=false
QUEUE_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --dry_run) DRY_RUN=true ;;
    --transfer_help) TRANSFER_HELP=true ;;
    --only|--only=*) HAS_ONLY=true; QUEUE_ARGS+=("$arg") ;;
    --gpus|--gpus=*) HAS_GPUS=true; QUEUE_ARGS+=("$arg") ;;
    --out_dir|--out_dir=*) HAS_OUT_DIR=true; QUEUE_ARGS+=("$arg") ;;
    --help|-h)
      cat <<'HELP'
Usage: bash cloud/setup_devcloud.sh [--dry_run] [queue arguments]
       bash cloud/setup_devcloud.sh --transfer_help

Default queue selection: --only env,tests,speed (no experiments).
Other arguments, including --only, --gpus and --approval, go to scripts/run_devcloud.py.
Environment: REPO_DIR (uploaded checkout or new clone), TORCH_VERSION,
             GIT_REMOTE (used only for a missing checkout), NTFY_TOPIC (optional).
--dry_run prints setup and queue commands without installation, writes or GPU discovery.
--transfer_help prints PC-side upload/download commands and exits.
HELP
      exit 0 ;;
    *) QUEUE_ARGS+=("$arg") ;;
  esac
done
$HAS_ONLY || QUEUE_ARGS+=(--only env,tests,speed)

transfer_help() {
  cat <<'TRANSFER'
Run these commands on the PC BEFORE starting the VM setup; replace VM_HOST:
VM_HOST='root@YOUR_VM_IP'
LOCAL_REPO='/mnt/sandisk/Sparse-Memory-LM/wt-mi300'
LOCAL_DATA='/mnt/sandisk/Sparse-Memory-LM/AngryAnt/data'
LOCAL_PYTHON='/mnt/sandisk/Sparse-Memory-LM/AngryAnt/.venv/bin/python'
REMOTE_REPO='/root/sparse-memory-lm'
LOCAL_RESULTS="$PWD/devcloud-results"
# Prepare this cache on the PC BEFORE allocating the VM; source caches are read only.
"$LOCAL_PYTHON" - "$LOCAL_REPO" <<'TOKENIZER_PC'
import hashlib
import os
from pathlib import Path
import shutil
import sys

dest = Path(sys.argv[1]) / ".cache" / "tiktoken"
caches = [dest, *(Path(os.environ[k]) for k in ("TIKTOKEN_CACHE_DIR", "DATA_GYM_CACHE_DIR")
                  if os.environ.get(k)), Path("/tmp/data-gym-cache")]
expected = {
    "6d1cbeee0f20b3d9449abfede4726ed8212e3aee": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
    "6c7ea1a7e38e3a7f062df639a5b80947f075ffe6": "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
}
sources = {}
for name, digest in expected.items():
    sources[name] = next((d / name for d in caches if (d / name).is_file()
                         and hashlib.sha256((d / name).read_bytes()).hexdigest() == digest), None)
if not all(sources.values()):
    raise SystemExit("Missing verified GPT-2 cache. On the PC, run the following command, then retry:\n"
                     'TIKTOKEN_CACHE_DIR="$LOCAL_REPO/.cache/tiktoken" "$LOCAL_PYTHON" '
                     '-c \'import tiktoken; tiktoken.get_encoding("gpt2")\'')
dest.mkdir(parents=True, exist_ok=True)
for name, source in sources.items():
    if source.resolve() != (dest / name).resolve():
        shutil.copyfile(source, dest / name)
    print("verified", dest / name)
TOKENIZER_PC
ssh "$VM_HOST" 'command -v rsync >/dev/null || { apt-get update -q && DEBIAN_FRONTEND=noninteractive apt-get install -yq rsync; }'
ssh "$VM_HOST" "mkdir -p $(printf '%q ' "$REMOTE_REPO/data" "$REMOTE_REPO/.cache/tiktoken")"
rsync -rtv --partial --protect-args --exclude='.git' --exclude='.venv*' --exclude='data/' --exclude='runs/' --exclude='report/' --exclude='.cache/' --exclude='.scratch/' --exclude='__pycache__/' "$LOCAL_REPO/" "$VM_HOST:$REMOTE_REPO/"
rsync -rtv --partial --protect-args "$LOCAL_DATA/wikipedia_en_gpt2" "$LOCAL_DATA/wikitext103_gpt2" "$VM_HOST:$REMOTE_REPO/data/"
rsync -rtv --partial --protect-args "$LOCAL_REPO/.cache/tiktoken/" "$VM_HOST:$REMOTE_REPO/.cache/tiktoken/"
ssh "$VM_HOST" "cd $(printf '%q' "$REMOTE_REPO") && bash cloud/setup_devcloud.sh"

Pull results BEFORE releasing the VM (the default output directory is runs/devcloud):
mkdir -p "$LOCAL_RESULTS"
rsync -rtv --partial --protect-args "$VM_HOST:$REMOTE_REPO/runs/devcloud/" "$LOCAL_RESULTS/devcloud/"
rsync -rtv --partial --protect-args "$VM_HOST:$REMOTE_REPO/runs/devcloud.tar" "$VM_HOST:$REMOTE_REPO/runs/devcloud.tar.sha256" "$LOCAL_RESULTS/"
rsync -rtv --partial --protect-args "$VM_HOST:$REMOTE_REPO/.cache/devcloud/setup.log" "$VM_HOST:$REMOTE_REPO/.cache/devcloud/requirements.lock.txt" "$LOCAL_RESULTS/"
(cd "$LOCAL_RESULTS" && sha256sum --check devcloud.tar.sha256)
(cd "$LOCAL_RESULTS/devcloud" && sha256sum --check SHA256SUMS)
Adjust REMOTE_REPO if REPO_DIR differs, and the result paths if using --out_dir.
TRANSFER
}
transfer_help
$TRANSFER_HELP && exit 0

say() { printf '%s [setup] %s\n' "$(date '+%F %T')" "$*"; }
show() { printf '+ '; printf '%q ' "$@"; printf '\n'; }
run() {
  show "$@"
  if ! $DRY_RUN; then "$@"; fi
}
fail() { printf 'FAILED: %s\n' "$*" >&2; exit 1; }

if ! $DRY_RUN; then
  [ "$(id -u)" = 0 ] || fail "run as root on the Developer Cloud VM"
  [ -f /etc/os-release ] || fail "Ubuntu 22.04 or 24.04 is required"
  . /etc/os-release
  [ "${ID:-}" = ubuntu ] && [[ "${VERSION_ID:-}" = 22.04 || "${VERSION_ID:-}" = 24.04 ]] \
    || fail "expected Ubuntu 22.04 or 24.04, found ${PRETTY_NAME:-unknown}"
  if command -v tmux >/dev/null && tmux has-session -t '=devcloud' 2>/dev/null; then
    say "tmux session devcloud already exists; leave its environment alone (tmux attach -t devcloud)"
    exit 0
  fi
fi

# Keep this separate from the host Python, which may enforce PEP 668 on fresh Ubuntu.
PACKAGES=(ca-certificates git tmux rsync curl python3-venv build-essential)
if $DRY_RUN || ! dpkg-query -W -f='${db:Status-Status}\n' "${PACKAGES[@]}" 2>/dev/null | \
    awk 'BEGIN { ok = 1 } $0 != "installed" { ok = 0 } END { exit !ok }'; then
  run env DEBIAN_FRONTEND=noninteractive apt-get update -q
  run env DEBIAN_FRONTEND=noninteractive apt-get install -yq "${PACKAGES[@]}"
fi

if [ ! -f "$REPO_DIR/smlm/train.py" ]; then
  if [ -d "$REPO_DIR" ] && [ -n "$(ls -A "$REPO_DIR")" ]; then
    fail "REPO_DIR exists but is not a checkout: $REPO_DIR (nothing overwritten)"
  fi
  run git clone -- "$GIT_REMOTE" "$REPO_DIR"
fi
if ! $DRY_RUN; then
  [ -f "$REPO_DIR/scripts/run_devcloud.py" ] || fail "upload the reviewed DevCloud files first"
  cd "$REPO_DIR"
fi
STATE="$REPO_DIR/.cache/devcloud"
VENV="$REPO_DIR/.venv-devcloud"
BOOTSTRAP="$STATE/bootstrap"
run mkdir -p "$STATE/tmp" "$REPO_DIR/runs"
if ! $DRY_RUN; then
  exec > >(tee -a "$STATE/setup.log") 2>&1
  trap 'say "FAILED at line $LINENO; see $STATE/setup.log; the VM remains running"' ERR
fi

# These can silently force an architecture or interpreter mode, and expandable segments is unsafe on ROCm.
run unset PYTORCH_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF PYTORCH_HIP_ALLOC_CONF TRITON_INTERPRET HSA_OVERRIDE_GFX_VERSION \
  ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
CACHE_ENV=(
  "TRITON_CACHE_DIR=$REPO_DIR/.cache/triton"
  "TORCHINDUCTOR_CACHE_DIR=$REPO_DIR/.cache/torchinductor"
  "XDG_CACHE_HOME=$REPO_DIR/.cache"
  "UV_CACHE_DIR=$REPO_DIR/.cache/uv"
  "UV_PYTHON_INSTALL_DIR=$STATE/python"
  "PIP_CACHE_DIR=$REPO_DIR/.cache/pip"
  "HF_HOME=$REPO_DIR/.cache/huggingface"
  "TIKTOKEN_CACHE_DIR=$REPO_DIR/.cache/tiktoken"
  "TMPDIR=$STATE/tmp"
)
run export "${CACHE_ENV[@]}"

# Recheck every uploaded prepared file on every setup, including metadata and secondary validation data.
DATA_CHECK='set -euo pipefail
cd "$1/data"
awk '\''$2 ~ /^(wikipedia_en_gpt2|wikitext103_gpt2)\// { print }'\'' ../cloud/data_sha256.txt | sha256sum --check --strict'
run bash -c "$DATA_CHECK" bash "$REPO_DIR"

# GPT-2 assets normally come from Azure Blob Storage, which is not a guaranteed VM network destination.
TOKENIZER_CHECK='import hashlib, os
from pathlib import Path
cache = Path(os.environ["TIKTOKEN_CACHE_DIR"])
expected = {
    "6d1cbeee0f20b3d9449abfede4726ed8212e3aee": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
    "6c7ea1a7e38e3a7f062df639a5b80947f075ffe6": "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
}
for name, digest in expected.items():
    p = cache / name
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
        raise SystemExit(f"Missing/corrupt GPT-2 cache: {p}; follow setup_devcloud.sh --transfer_help on the PC")
    print("verified GPT-2 cache", p)'
run python3 -c "$TOKENIZER_CHECK"

if $DRY_RUN || [ ! -x "$BOOTSTRAP/bin/python" ]; then
  run python3 -m venv "$BOOTSTRAP"
fi
run "$BOOTSTRAP/bin/python" -m pip install uv
UV="$BOOTSTRAP/bin/uv"
if $DRY_RUN || [ ! -x "$VENV/bin/python" ]; then
  run "$UV" venv --managed-python --python 3.12 "$VENV"
fi
run "$VENV/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
run "$UV" pip install --python "$VENV/bin/python" "torch==$TORCH_VERSION" --index-url https://download.pytorch.org/whl/rocm7.2
run "$UV" pip install --python "$VENV/bin/python" -r "$REPO_DIR/requirements.txt" transformers==5.18.0
run bash -c '"$1" pip freeze --python "$2" > "$3"' bash "$UV" "$VENV/bin/python" "$STATE/requirements.lock.txt"
TOKENIZER_WARM='import tiktoken, tiktoken.load
def no_download(url):
    raise RuntimeError(f"Tokenizer asset absent from uploaded cache: {url}")
tiktoken.load.read_file = no_download
encoding = tiktoken.get_encoding("gpt2")
print("GPT-2 tokenizer ready from verified uploaded cache:", encoding.n_vocab, "tokens")'
run "$VENV/bin/python" -c "$TOKENIZER_WARM"
PY_CHECK='import torch, triton
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
assert torch.version.hip and torch.version.hip.startswith("7.2"), torch.version.hip
assert torch.cuda.is_available(), "GPU not visible to PyTorch"
print("torch", torch.__version__, "hip", torch.version.hip, "triton", triton.__version__)
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    arch = getattr(p, "gcnArchName", "")
    print(i, p.name, arch, p.total_memory, "bytes")
    assert arch.split(":")[0] == "gfx942", f"expected MI300X/gfx942, got {arch}"'
run "$VENV/bin/python" -c "$PY_CHECK"

# tmux may have an older server environment, so pass the clean environment explicitly to the new session.
QUEUE_ENV=(env -u PYTORCH_ALLOC_CONF -u PYTORCH_CUDA_ALLOC_CONF -u PYTORCH_HIP_ALLOC_CONF
  -u TRITON_INTERPRET -u HSA_OVERRIDE_GFX_VERSION -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES
  -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL "${CACHE_ENV[@]}" "NTFY_TOPIC=${NTFY_TOPIC:-}")
printf -v QUEUE_COMMAND '%q ' "${QUEUE_ENV[@]}" "$VENV/bin/python" -u scripts/run_devcloud.py "${QUEUE_ARGS[@]}"
printf -v STDOUT_LOG '%q' "$REPO_DIR/runs/devcloud_queue_stdout.log"
QUEUE_SHELL="set -o pipefail; $QUEUE_COMMAND 2>&1 | tee -a $STDOUT_LOG"
if ! $DRY_RUN && tmux has-session -t '=devcloud' 2>/dev/null; then
  say "tmux session devcloud already exists; no second queue started (tmux attach -t devcloud)"
else
  run tmux new-session -d -s devcloud -c "$REPO_DIR" -- bash -c "$QUEUE_SHELL"
  if ! $DRY_RUN; then
    say "queue started: tmux attach -t devcloud"
    say "queue output: $REPO_DIR/runs/devcloud_queue_stdout.log"
  fi
fi

if $DRY_RUN; then
  PREVIEW_SCRIPT="$REPO_DIR/scripts/run_devcloud.py"
  if [ ! -f "$PREVIEW_SCRIPT" ]; then
    PREVIEW_SCRIPT="$SCRIPT_DIR/../scripts/run_devcloud.py"
  fi
  if [ -f "$PREVIEW_SCRIPT" ]; then
    PREVIEW_ARGS=("${QUEUE_ARGS[@]}")
    $HAS_GPUS || PREVIEW_ARGS+=(--gpus 0)
    $HAS_OUT_DIR || PREVIEW_ARGS+=(--out_dir "$REPO_DIR/runs/devcloud")
    say "queue command preview from $PREVIEW_SCRIPT (default preview GPU: 0)"
    show env PYTHONDONTWRITEBYTECODE=1 python3 "$PREVIEW_SCRIPT" --dry_run --python "$VENV/bin/python" "${PREVIEW_ARGS[@]}"
    env PYTHONDONTWRITEBYTECODE=1 python3 "$PREVIEW_SCRIPT" --dry_run --python "$VENV/bin/python" "${PREVIEW_ARGS[@]}"
  else
    say "queue source is unavailable until clone/upload; the tmux command above is the setup plan"
  fi
  say "dry run only: no installation, files, GPU discovery or tmux session created"
else
  say "setup ready; failed queues leave the VM running; pull results before releasing it"
fi
