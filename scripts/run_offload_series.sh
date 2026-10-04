#!/bin/bash
# Step 2: all measurements of B-16M at home, one process per variant (results: report/offload/*.json).
# c-*-mem4G runs in a systemd user scope with a RAM limit, so the page cache cannot hold the whole 3.2-GB file.
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
B=scripts/bench_offload.py
run() { echo "=== $(date '+%T') $*"; "$@" 2>&1 | grep -E '"(tok_s|ppl|cache_hit_rate|aborted)"' | tr -d ' ' | paste -sd' '; }
run $PY $B --variant a-bf16 --ppl subset,full
run $PY $B --variant a-bf16 --graphs 0 --ppl none
run $PY $B --variant a-q4 --ppl subset,full
run $PY $B --variant a-q4 --graphs 0 --ppl none
run $PY $B --variant b-bf16 --ppl subset,full
run $PY $B --variant b-q4 --ppl subset
run $PY $B --variant b-fp32 --ppl subset,full
for f in 0 0.05 0.1 0.5; do run $PY $B --variant c --cache_frac $f --ppl subset; done
run $PY $B --variant c --cache_frac 0.3 --ppl subset,full
run $PY $B --variant c --cache_frac 0.1 --fifo_frac 0.2 --ppl subset
for f in 0.1 0.3; do
  run systemd-run --user --scope --quiet -p MemoryMax=4G -p MemorySwapMax=0 $PY $B --variant c --cache_frac $f \
      --ppl subset --tag mem4G
done
echo "=== $(date '+%T') SERIES DONE"
