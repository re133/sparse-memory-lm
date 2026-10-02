"""Run a phase of the stage-1 experiment sequentially (skips runs whose run-info.json says 'done').

  probe : A, B, C (init seed 0), 20M tokens
  ep1   : A s0/s1, B s0/s1, C s0, one epoch of WikiText-103 (3600 steps)
  ep3   : same five runs, three epochs, own cosine schedule over the full length
  v2    : B-v2a (no weight decay on sub-keys) and B-v2b (v2a + learned per-head score scale), seeds 0/1,
          one epoch with exactly the ep1 settings -> compared against runs/ep1/{A,B,C}-*
  v2b_ep3: B-v2b only, seeds 0/1, three epochs with exactly the ep3 settings -> against runs/ep3/*
  s1b   : stage-1b quick test on 500 M fresh Wikipedia tokens: B-1M (3 memory layers, shared 1M table,
          value LR 4 x base = 2.4e-3) and A, seed 0 each; WikiText-103 validation as second val set
Run entries are (model, seed) or (model, seed, [extra args]).
"""
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable          # the interpreter running this script (works from any worktree)

FIVE = [("A", 0), ("B", 0), ("C", 0), ("A", 1), ("B", 1)]
PHASES = {
    "probe": dict(runs=[("A", 0), ("B", 0), ("C", 0)], args=["--tokens", "20e6", "--eval_every_tokens", "2e6"]),
    "ep1": dict(runs=FIVE, args=["--epochs", "1", "--eval_every_tokens", "4e6"]),
    "ep3": dict(runs=FIVE, args=["--epochs", "3", "--eval_every_tokens", "8e6"]),
    "v2": dict(runs=[("B-v2a", 0), ("B-v2b", 0), ("B-v2a", 1), ("B-v2b", 1)],
               args=["--epochs", "1", "--eval_every_tokens", "4e6"]),
    # v2b only, exactly the ep3 settings and the B seeds -> paired comparison with runs/ep3/B-s{0,1}
    "v2b_ep3": dict(runs=[("B-v2b", 0), ("B-v2b", 1)], args=["--epochs", "3", "--eval_every_tokens", "8e6"]),
    "s1b": dict(runs=[("B-1M", 0, ["--value_lr", "2.4e-3"]), ("A", 0)],
                args=["--data", "wikipedia", "--extra_val", "wikitext103", "--tokens", "500e6",
                      "--eval_every_tokens", "10e6"]),
}


def main():
    for phase in sys.argv[1:]:
        spec = PHASES[phase]
        for model, seed, *extra in spec["runs"]:
            extra = extra[0] if extra else []
            out = os.path.join(ROOT, "runs", phase, f"{model}-s{seed}")
            info = os.path.join(out, "run-info.json")
            if os.path.exists(info) and json.load(open(info)).get("status") == "done":
                print(f"skip {phase}/{model}-s{seed} (done)", flush=True)
                continue
            os.makedirs(out, exist_ok=True)
            cmd = [PY, "-m", "smlm.train", "--model", model, "--seed", str(seed), "--out_dir", out, *spec["args"],
                   *extra]
            print(time.strftime("%H:%M:%S"), "start", " ".join(cmd), flush=True)
            env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
            with open(os.path.join(out, "stdout.log"), "w") as log:
                rc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
            print(time.strftime("%H:%M:%S"), f"end {phase}/{model}-s{seed} rc={rc}", flush=True)


if __name__ == "__main__":
    main()
