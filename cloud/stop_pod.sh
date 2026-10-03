#!/bin/bash
# Stop this Runpod Pod through the Runpod API: the GPU is released and compute billing stops; /workspace (volume
# disk with repo, data and checkpoints) is kept and billed as stopped volume ($0.20/GB/month).
# Runpod sets RUNPOD_POD_ID and a Pod-scoped RUNPOD_API_KEY in every Pod (docs: "Environment variables");
# SSH sessions may not inherit them, so they are also read from PID 1's environment.
#   stop_pod.sh           stop the Pod
#   stop_pod.sh --check   only check that the API is reachable with the Pod's key
set -u -o pipefail
envget() {
  local v="${!1:-}"
  if [ -z "$v" ] && [ -r /proc/1/environ ]; then
    v=$(tr '\0' '\n' < /proc/1/environ | sed -n "s/^$1=//p" | head -1)
  fi
  echo "$v"
}
POD=$(envget RUNPOD_POD_ID)
KEY=$(envget RUNPOD_API_KEY)
if [ -z "$POD" ] || [ -z "$KEY" ]; then
  echo "RUNPOD_POD_ID / RUNPOD_API_KEY not found - not running on a Runpod Pod? Cannot stop automatically."
  exit 2
fi
if [ "${1:-}" = "--check" ]; then
  curl -sf -m 30 -H "Authorization: Bearer $KEY" "https://rest.runpod.io/v1/pods/$POD" \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print('Runpod API ok: pod', d.get('id'), d.get('name'), d.get('desiredStatus'))"
  exit $?
fi
sync
code=$(curl -s -m 60 -o /tmp/runpod_stop.out -w '%{http_code}' -X POST -H "Authorization: Bearer $KEY" \
  "https://rest.runpod.io/v1/pods/$POD/stop")
echo "HTTP $code $(head -c 300 /tmp/runpod_stop.out)"
case "$code" in 200|202|204) exit 0 ;; esac
# fallback: the CLI that is preinstalled on Pods (new and old syntax)
if command -v runpodctl >/dev/null; then
  RUNPOD_API_KEY="$KEY" runpodctl pod stop "$POD" && exit 0
  RUNPOD_API_KEY="$KEY" runpodctl stop pod "$POD" && exit 0
fi
exit 1
