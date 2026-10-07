# AMD Developer Cloud

## Time and GPU-hour budget

The full plan fits the provisional budget **on one allocated GPU**: the scenario below
totals **8.900–17.349 GPU-hours**, against **20 usable GPU-hours** after reserving
**5 of the requested 25 GPU-hours**. This is a planning scenario, not measured MI300X
performance or a confidence interval. Actual MI300X throughput, installation time,
storage speed and the combined BE model remain unmeasured. Recalculate after the
technical preflight; do not treat the upper scenario as a runtime guarantee.

The historical evidence comes from the main checkout's read-only `runs/` files.
The command below was run from this worktree and reproduces the numerical evidence
and all budget arithmetic. `duration_s` is the recorded whole-process duration;
`train_time_s` excludes evaluation. Queue timestamps include process launch and
monitor shutdown, so their durations differ slightly.

| Historical run | GPU | Whole run (s) | Median training tokens/s | Interpretation |
|---|---|---:|---:|---|
| `cloud/B-16M-s0` | H200 | 10707.1 | 52683.98 | Shared GPU; not a solo speed reference |
| `cloud/B-4M-s0` | H200 | 7700.1 | 77551.24 | Shared GPU; not a solo speed reference |
| `cloud_dense/D-100M-s0` | H100 | 10397.5 | 46548.41 | Shared GPU; not a solo speed reference |
| `cloud_dense/D-200M-s0` | H100 | 12845.6 | 43014.20 | Shared GPU; not a solo speed reference |
| `amd_mi350x/speed_B-16M` | MI350X | 118.1 | 112764.03 | Solo short benchmark |
| `amd_mi350x/speed_B-4M` | MI350X | 60.8 | 119357.75 | Solo short benchmark |
| `amd_mi350x/speed_D-100M` | MI350X | 34.2 | 155897.72 | Solo short benchmark |
| `cloud_dense_preflight/D-100M-s0` | H100 | 95.7 | 208334.28 | Solo short benchmark |
| `cloud_dense_preflight/D-200M-s0` | H100 | 109.0 | 143104.93 | Solo short benchmark |

Across all completed historical full runs, H200 durations ranged from **2.1389 to
2.9742 hours** and H100 durations from **2.1355 to 4.1715 hours**. Those overlapping
runs are useful elapsed-time evidence, but summing them as exclusive GPU time or
using their throughput to compare hardware would be wrong. This matches the
warnings in `REPORT.md`, “Result of the cloud runs” and “Step 1”.

The old MI350X queue recorded **3 seconds** for environment inspection, **642
seconds** for its test step, and **447 seconds** for its speed steps combined.
Pytest itself reported **107 passed, 1 warning in 638.84 seconds**. That older suite
excluded the Qwen test file and predates later additions; it is not a measurement
of today's full suite on gfx942. See `runs/amd_mi350x/queue.log` and
`runs/amd_mi350x/tests/stdout.log`.

For the first four experiments, use the evaluation-work approximation already in
`scripts/run_dense.py`: `500e6 + 50 * (1.48e6 + .25e6) / 3`, giving
**528833333.33 equivalent training tokens**. The scenario applies the existing
**1.15 runtime factor**, adds an explicit **0.1-hour per-run allowance** for
finalization, and assumes MI300X achieves **50–100% of the measured MI350X solo
throughput**. This speed range is an intentionally broad stress assumption, not a
hardware-specification ratio or a measured slowdown.

There is no MI350X D-200M measurement. Its proxy is H100 D-200M throughput multiplied
by the observed D-100M MI350X/H100 ratio: **0.74830567**, giving **107086.23 tokens/s**.
This extrapolation across model sizes is another uncertainty. `REPORT.md` “Step 4”
explicitly excludes BE-1M from the original experiment; there is no historical BE
timing. Its allowance below must be replaced by measured evidence when available.

| Step, in queue priority order | One MI300X planning wall time (hours) | Basis |
|---|---:|---|
| `env` | 0.050 | Explicit allowance; old inspection took seconds |
| `tests` | 0.250–0.500 | Allowance above the old MI350X suite, including compilation |
| `speed` | 0.125–0.250 | Allowance around the old solo benchmark queue |
| `B-16M-s1` | 1.598–3.096 | MI350X B-16M solo throughput and formula above |
| `B-4M-s1` | 1.515–2.931 | MI350X B-4M solo throughput and formula above |
| `D-100M-s1` | 1.184–2.267 | MI350X D-100M solo throughput and formula above |
| `D-200M-s1` | 1.678–3.255 | D-200M proxy and formula above |
| `BE-1M-s0` | 2.000–4.000 | **Provisional allowance only; no measured BE throughput** |
| Setup, upload/check and final packaging/pull | 0.500–1.000 | Explicit combined allowance; network/storage unmeasured |
| **Total, one allocated GPU** | **8.900–17.349** | Sum before rounding |

With **eight allocated GPUs billed for the whole VM**, parallel experiments do
not divide GPU-hour consumption by eight: idle cards still count. The same
scenario gives **2.925–5.800 wall hours**, or **23.400–46.400 GPU-hours**, exceeding
the reserve-preserving target even at the optimistic end. The **20-GPU-hour**
target allows just **2.5 wall hours** on such a VM. Selecting fewer devices in
the queue does not reduce the VM's allocated-card count.

Prefer the single-GPU allocation. If only the full eight-card VM is available,
confirm the actual credit-accounting and GPU-release rules before launching it;
the full plan cannot responsibly be promised within this budget. Recalculate
using the observed preflight throughput and actual elapsed allocation time.
If work must be dropped, drop **BE-1M first**, then **D-200M**, **D-100M**, **B-4M**,
and preserve **B-16M** longest. Dropping BE alone still projects
**20.820–40.441 GPU-hours** on an eight-card allocation, so that alone does not
restore the reserve. Do not consume the reserve waiting for experiment approval;
complete the criteria review before the paid experiment session. Stopping the
queue does not release the VM or stop provider accounting.

### Reproduce the evidence and scenario

This command reads JSON and text only; it does not import PyTorch or run training.
The assumptions are declared separately from the observations in its output.

```bash
/mnt/sandisk/Sparse-Memory-LM/AngryAnt/.venv/bin/python - <<'PY'
from pathlib import Path
from datetime import datetime
import json

root = Path('/mnt/sandisk/Sparse-Memory-LM/AngryAnt/runs')
infos = {}
print('HISTORICAL: group/run GPU duration_s duration_h train_s median_tok_s')
for group in ('cloud', 'cloud_dense', 'cloud_dense_preflight', 'amd_mi350x'):
    group_hours = []
    for path in sorted((root / group).glob('*/run-info.json')):
        info = json.loads(path.read_text())
        infos[group, path.parent.name] = info
        result = info['results']
        hours = info['duration_s'] / 3600
        group_hours.append(hours)
        print(f'{group}/{path.parent.name} {info["hardware"]["gpu"]} '
              f'{info["duration_s"]:.1f} {hours:.4f} {info["train_time_s"]:.1f} '
              f'{result["train_tok_s_median"]:.2f}')
    if group in ('cloud', 'cloud_dense'):
        print(f'{group} full-run duration range_h {min(group_hours):.4f} {max(group_hours):.4f}')

def rate(group, name):
    return infos[group, name]['results']['train_tok_s_median']

lines = (root / 'amd_mi350x/queue.log').read_text().splitlines()
starts, durations = {}, {}
for line in lines:
    stamp, message = datetime.fromisoformat(line[:19]), line[20:]
    if message.startswith('amd queue start:'):
        queue_start = stamp
    elif message.startswith('env:'):
        print('MI350X env elapsed_s', (stamp - queue_start).total_seconds())
    elif message.startswith('start '):
        starts[message.split()[1].rstrip(':')] = stamp
    elif message.startswith('end '):
        name = message.split()[1]
        durations[name] = (stamp - starts[name]).total_seconds()
print('MI350X queue duration_s', durations)
print('MI350X speed total_s', sum(t for name, t in durations.items() if name.startswith('speed_')))
print('MI350X pytest:', (root / 'amd_mi350x/tests/stdout.log').read_text().splitlines()[-1])

# Policy assumptions, not measured MI300X performance or statistical intervals.
work = 500e6 + 50 * (1.48e6 + .25e6) / 3
ratio = rate('amd_mi350x', 'speed_D-100M') / rate('cloud_dense_preflight', 'D-100M-s0')
proxy = rate('cloud_dense_preflight', 'D-200M-s0') * ratio
speeds = {
    'B-16M-s1': rate('amd_mi350x', 'speed_B-16M'),
    'B-4M-s1': rate('amd_mi350x', 'speed_B-4M'),
    'D-100M-s1': rate('amd_mi350x', 'speed_D-100M'),
    'D-200M-s1': proxy,
}
print('ASSUMPTIONS: MI300X speed fraction 0.5..1.0; runtime factor 1.15; finalization 0.1h/run')
print('work_tokens', work, 'D100_MI350X_H100_ratio', ratio, 'D200_proxy_tok_s', proxy)
estimates = {}
for name, speed in speeds.items():
    estimates[name] = tuple(1.15 * work / (speed * fraction) / 3600 + .1 for fraction in (1., .5))
    print(name, 'planning_h', *(f'{hours:.3f}' for hours in estimates[name]))
estimates['BE-1M-s0'] = (2., 4.)  # Explicit provisional allowance: no BE timing exists.
env, tests, speed = (.05, .05), (.25, .5), (.125, .25)
setup_pack = (.5, 1.)
print('ALLOWANCES_h env', env, 'tests', tests, 'speed', speed, 'setup_plus_pack', setup_pack,
      'BE-1M', estimates['BE-1M-s0'])
preflight = tuple(env[i] + tests[i] + speed[i] for i in (0, 1))
serial = tuple(sum(hours[i] for hours in estimates.values()) + preflight[i] + setup_pack[i] for i in (0, 1))
parallel = tuple(max(hours[i] for hours in estimates.values()) + preflight[i] + setup_pack[i] for i in (0, 1))
without_be = tuple(max(hours[i] for name, hours in estimates.items() if name != 'BE-1M-s0')
                   + preflight[i] + setup_pack[i] for i in (0, 1))
print('ONE allocated GPU total_h', *(f'{hours:.3f}' for hours in serial))
print('EIGHT allocated GPUs wall_h', *(f'{hours:.3f}' for hours in parallel))
print('EIGHT allocated GPUs charged_GPU_h', *(f'{hours * 8:.3f}' for hours in parallel))
print('EIGHT allocated GPUs without_BE_charged_GPU_h', *(f'{hours * 8:.3f}' for hours in without_be))
print('budget_GPU_h', 25, 'reserve_GPU_h', 25 * .2, 'usable_GPU_h', 25 * .8,
      'eight_GPU_wall_limit_h', 25 * .8 / 8)
PY
```

## Setup choice and fixed commands

Use the checkout-local Python venv, with managed Python from `uv` and the official
ROCm wheel. This keeps both supported Ubuntu releases on the same Python version,
avoids modifying the distribution Python, and avoids Docker image drift, device
passthrough, bind mounts and an additional large image transfer. The preinstalled
host ROCm driver is still required: a venv does not fix a broken driver.
`setup_devcloud.sh` pins the wheel recorded as working in the repository's README
and ROCm report, installs `requirements.txt` and the Qwen test dependency from
`requirements-qwen.txt`, and saves the resolved environment with `uv pip freeze`.
The defaults are visible in the setup preview. The wheel source is the
[official ROCm wheel index](https://download.pytorch.org/whl/rocm7.2/torch/).
`TORCH_VERSION` can override the pin, but changing the stack requires a fresh
output directory and another preflight. Setup checks the actual HIP version,
visible architecture, and Qwen imports before launching tmux.

Existing uploaded checkouts are used as-is. An explicit missing `REPO_DIR` is
cloned from the public repository; no existing checkout is pulled, reset or
reconfigured. Until these changes have been reviewed and published, upload this
worktree: the public clone will not contain the new entry points yet. The upload
excludes the worktree's `.git` pointer, local environments, caches and historical
results. Token files are uploaded separately and read-only during verification
and training; the old storage-box downloader is never invoked.

The queue clears inherited PyTorch allocator settings, architecture overrides,
Triton interpreter mode and conflicting GPU masks. It uses physical
`ROCR_VISIBLE_DEVICES` ordinals per subprocess and a checkout-local Triton cache.
The models still see their assigned GPU as local device zero. The environment
step records the full inventory. Tests and speed benchmarks run sequentially on
the first selected card. Experiments run in the specified priority order, with
one active experiment per selected card; a free card takes the next pending job.
There is no distributed training or shared-card experiment scheduling.

The B runs exactly reproduce `run_cloud.py` arguments with the requested init seed
changed; the dense runs do the same for `run_dense.py`. Output paths necessarily
differ. The speed group reproduces every `SPEED` entry and `SPEED_ARGS` from
`run_amd.py`; it excludes the old crosscheck, which was not requested here.
BE has no original combined run. Its **proposed, unapproved** configuration keeps
the same Wikipedia budget, validation schedule and optimizer defaults, uses the
requested combined preset and seed, keeps the product-key learning rate from B,
and sets the Engram learning rate separately to the Step-4 value. The exact
arguments are printed by `--dry_run`; review them as a new experiment.

## Before approval

Read the budget section and settle the provider's allocated-GPU billing rules.
Review this worktree and prepare the criteria before using the experiment credit.
Preview both entry points locally; these commands do not start training, probe a
GPU, download packages, create result directories or send notifications:

```bash
bash cloud/setup_devcloud.sh --dry_run --only env,tests,speed
/mnt/sandisk/Sparse-Memory-LM/AngryAnt/.venv/bin/python scripts/run_devcloud.py \
  --dry_run --only all --gpus 0,1,2,3,4,5,6,7
bash cloud/setup_devcloud.sh --transfer_help
```

Before allocating the VM, follow the tokenizer-cache preparation block printed
by `--transfer_help`. It copies hash-verified GPT-2 assets from the existing PC
cache into this checkout's `.cache/tiktoken` and uploads it separately. If the
cache is missing, populate it on the PC using the printed command and retry.
The VM checks those assets and initializes the tokenizer with downloads disabled;
it does not need access to the tokenizer's default Azure Blob Storage endpoint.

`--only` accepts individual step names, `speed`, `experiments` or `all`. Input order
does not change priority. Without `--only`, only technical preflight is selected.
Dry-run mode prints the experiment commands even without an approval file;
execution requires approval. `--gpus auto` discovers cards only during execution;
a dry-run preview assumes the first card unless an explicit list is supplied.

Prepare a criteria document outside the protected reports. For the second seeds,
a sensible draft is to report the per-seed Wikipedia and WikiText results, the
seed spread, and the dense-equivalent interpolation using the existing Step-1
method, with no tuning after results are seen. For BE, decide in writing whether
the purpose is descriptive or has a success threshold, which B/E baselines it
uses, and how its larger total table and single seed limit the conclusion.
**This runbook neither approves those criteria nor invents a BE success threshold.**

After the user has approved the actual written criteria, create an approval JSON
whose fields are:

- `approved`: literal `true`, entered only after approval.
- `criteria_file`: path to the criteria text, relative to the approval JSON or absolute.
- `criteria_sha256`: the output hash from `sha256sum` on that exact text.
- `steps`: the explicitly approved experiment names, for example `B-16M-s1`,
  `B-4M-s1`, `D-100M-s1`, `D-200M-s1`, `BE-1M-s0` as appropriate.

There is deliberately no ready-made approved file. Missing approval, changed
criteria or unlisted experiments stop execution. The queue copies the approval
and criteria text into its result archive. This records authorization; it is not
a cryptographic identity check. Approval of B/D does not implicitly approve BE.

## Day 1

Run `--transfer_help` on the PC and use its exact upload commands after replacing
`VM_HOST`. It prints the local worktree, the original read-only data source,
remote destination and pull commands. Upload both prepared dataset directories;
there is no need to fetch raw Wikipedia or rebuild tokens. Setup verifies every
prepared-file entry in `cloud/data_sha256.txt` before each new queue startup.

Check free disk space before starting. Results, an uncompressed tar, any previous
tar being atomically replaced, retained failed attempts, and environment/cache
files coexist. Allocate space for those copies, not just the final checkpoints.
The original B checkpoints are already large; the exact inspection command is
recorded in `CODEX_NOTES.md`. Upload time and disk speed remain unmeasured here.

On the VM, from the uploaded checkout:

```bash
bash cloud/setup_devcloud.sh --only env,tests,speed --gpus 0
tmux attach -t devcloud
# In another SSH session:
tail -f runs/devcloud/queue.log
```

The environment must report gfx942 and the expected HIP stack. Inspect
`runs/devcloud/tests/stdout.log`: this runs `pytest -q -rs tests` with no ignored
test modules. Setup installs the Qwen dependency, and queue discovery checks its
imports, so missing transformers cannot silently pass as an expected skip.
Unexpected skips, kernel errors, OOMs or a failed benchmark must be understood
before any experiment. Check `run-info.json` for each speed step, including
throughput, memory peak and inference results. The first run may compile kernels.

Recalculate the budget from those measurements and actual VM elapsed time. Once
criteria and the selected experiments are approved, upload the approval JSON and
its referenced criteria document to the VM. Then, after the preflight tmux
session has exited, start the approved subset, for example:

```bash
bash cloud/setup_devcloud.sh --only B-16M-s1,B-4M-s1,D-100M-s1,D-200M-s1 \
  --approval approval.json --gpus 0
# Only if the allocation/budget and all experiment criteria have been approved:
bash cloud/setup_devcloud.sh --only experiments --approval approval.json \
  --gpus 0,1,2,3,4,5,6,7
```

Optional `NTFY_TOPIC` is passed to the queue for a final status notification.
No credential file or storage-box configuration is needed. Setup does not launch
a second queue when the tmux session already exists. A filesystem lock prevents
concurrent queues writing the same output directory.

A subset can reuse completed prerequisites. For example, `--only experiments`
requires all preflight markers already present in the same output directory;
it does not silently skip tests. A prerequisite failure stops later stages.
Independent experiment failures are recorded while other selected experiments
continue. `selection.json` says whether **the selected subset** finished; an
env/tests/speed-only success never means the experiments finished.

## Resume and recovery

Run the same setup command again, or invoke the queue with
`.venv-devcloud/bin/python scripts/run_devcloud.py` and the same arguments.
Completed steps require a matching command/source signature and valid artifacts;
experiments also require a nonempty final `model.pt` and `run-info.json` marked
done. A corrupt or stale completion marker is refused for manual inspection;
use a fresh output directory after intentional code or environment changes.
The queue records package versions and rejects environment drift when resuming.

An incomplete attempt without a completion marker is renamed under `attempts/`
and restarted from the beginning. This consumes the full experiment budget
again; no mid-run checkpoint continuation is claimed. There are no automatic
retries within one invocation. Interrupted processes receive a process-group
termination with a kill fallback; a still-existing PID after an abrupt parent
crash blocks retry for inspection. Do not kill unrelated workloads to clear it.
A hard kill or VM loss cannot guarantee packaging; recover the local files first.

Timeout and stall outcomes are failures, never completion. The stall limit is
configurable with `--stall_min`; step deadlines are visible in `plan()`.
These are process safeguards, not a provider credit cap. The queue never releases
the VM. Track the allocation deadline and release it through the provider after
pulling verified results.

## Pulling results and what to check

The queue writes `runs/devcloud/SHA256SUMS` over results, retained attempts and
approval records, then `runs/devcloud.tar` plus `runs/devcloud.tar.sha256`.
The tar is intentionally uncompressed to avoid spending rental time compressing
floating-point weights. Its top-level directory is `devcloud/`. The queue log
and selection status are finalized before the manifest is made. A failed
selection still gets an archive, and the process exits unsuccessfully.

Use the exact PC commands printed by `--transfer_help`. Directory rsync supports
partial transfers and avoids pulling the large archive again for a small update;
the tar provides a standalone snapshot. You may pull either, or both. Verify
`SHA256SUMS` from inside the pulled result directory, or verify the tar sidecar
before extracting it and then verify the inner manifest. Also pull setup logs
and `requirements.lock.txt`, which live outside the result directory.

To regenerate an interrupted archive without running any GPU work:

```bash
.venv-devcloud/bin/python scripts/run_devcloud.py --pack_only
```

Before releasing the VM, check the pull hashes, `selection.json`, each requested
step's `DONE`, the final checkpoints, test summary and speed records. Inspect
failed attempts instead of treating an archive's existence as success. Use a
new output location for intentionally changed experiments and repeat preflight;
never relabel partial runs as completed seeds.
