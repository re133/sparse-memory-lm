#!/bin/bash
# Stop this IONOS Cloud GPU VM through the Cloud API (compute billing stops; the disk and the VM stay).
# Shutting down inside the OS does NOT stop billing on IONOS (docs: "Stop a Cloud GPU VM").
# Needs IONOS_TOKEN, IONOS_DATACENTER_ID, IONOS_SERVER_ID (from ~/smlm-cloud-kit/cloud.env, see CLOUD.md).
set -u
KIT="${SMLM_KIT:-$HOME/smlm-cloud-kit}"
[ -f "$KIT/cloud.env" ] && . "$KIT/cloud.env"
if [ -z "${IONOS_TOKEN:-}" ] || [ -z "${IONOS_DATACENTER_ID:-}" ] || [ -z "${IONOS_SERVER_ID:-}" ]; then
  echo "IONOS_TOKEN / IONOS_DATACENTER_ID / IONOS_SERVER_ID not set - cannot stop the VM automatically"
  exit 2
fi
if [ "${1:-}" = "--check" ]; then          # only verify that the token can see this server
  curl -sf -H "Authorization: Bearer $IONOS_TOKEN" \
    "https://api.ionos.com/cloudapi/v6/datacenters/$IONOS_DATACENTER_ID/servers/$IONOS_SERVER_ID" \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print('IONOS API ok:', d['properties']['name'], d['properties'].get('type'), d['metadata']['state'])"
  exit $?
fi
sync
code=$(curl -s -o /tmp/ionos_stop.out -w '%{http_code}' -X POST -H "Authorization: Bearer $IONOS_TOKEN" \
  "https://api.ionos.com/cloudapi/v6/datacenters/$IONOS_DATACENTER_ID/servers/$IONOS_SERVER_ID/stop")
echo "HTTP $code $(head -c 300 /tmp/ionos_stop.out)"
[ "$code" = "202" ] || [ "$code" = "200" ]
