"""Step 10 (FACTK) at home (RX 9070): B-1M and D-50M on the Wikipedia data with made-up people (REPORT.md, "Step 10").

  python scripts/run_factk.py [--deadline 2026-10-10T11:00]   (as a systemd unit; resumable, a finished run is skipped)
  python scripts/run_factk.py --smoke DIR                       (both runs on 2M tokens into DIR, no ntfy)

Same arguments as the original runs, only --data differs: B-1M-sparse as in runs/hampter (table LR 2.4e-3, Triton
kernels), D-50M as in runs/cloud_dense. D-100M runs on a Runpod GPU (cloud/setup_dense.sh with SMLM_DENSE_RUNS,
SMLM_DENSE_DATA, SMLM_DENSE_OUT). A run is only started if its estimated end (EST_H) lies before --deadline. Output
runs/factk/<name>/ with GPU temperature / power every 10 s. ntfy message after every run. Nothing is pushed.
"""
import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import run_shape  # noqa: E402

run_shape.OUT = os.path.join(ROOT, "runs", "factk")
run_shape.TITLE = "SMLM FACTK"
run_shape.LOG = "queue_home.log"                   # queue.log is the Runpod queue of D-100M
run_shape.COMMON = ["--data", "wikipedia_factk", "--tokens", "500e6", "--extra_val", "wikitext103",
                    "--eval_every_tokens", "10e6", "--seed", "0", "--data_seed", "1234"]
run_shape.RUNS = [
    ("B-1M-factk-s0", ["--model", "B-1M-sparse", "--value_lr", "2.4e-3", "--mem_impl", "triton"]),
    ("D-50M-factk-s0", ["--model", "D-50M"]),
]
run_shape.EST_H = {"B-1M-factk-s0": 2.6, "D-50M-factk-s0": 2.6}  # smoke test at home: 61.0k / 61.4k tok/s


def main():
    argv, deadline = sys.argv[1:], None
    if len(argv) == 2 and argv[0] == "--deadline":
        deadline = datetime.fromisoformat(argv[1])
    elif len(argv) == 2 and argv[0] == "--smoke":
        run_shape.OUT = run_shape.SMOKE = os.path.abspath(argv[1])
    elif argv:
        sys.exit(__doc__)
    os.makedirs(run_shape.OUT, exist_ok=True)
    run_shape.log(f"FACTK queue start: {[n for n, _ in run_shape.RUNS]}"
                  + (f", deadline {deadline:%Y-%m-%d %H:%M}" if deadline else ""))
    results = {name: run_shape.run(name, args, deadline) for name, args in run_shape.RUNS}
    failed = [n for n, r in results.items() if r != "done"]
    ppls = {n: run_shape.info(os.path.join(run_shape.OUT, n)).get("results", {}).get("val_ppl") for n, _ in run_shape.RUNS}
    summary = ", ".join(f"{n} {p:.3f}" for n, p in ppls.items() if p)
    run_shape.log("QUEUE DONE" + (f", NOT finished: {failed}" if failed else "") + f" | {summary}")
    run_shape.notify(("FACTK-Nacht fertig" if not failed else f"FACTK UNVOLLSTÄNDIG ({', '.join(failed)})")
                     + f": {summary}. Vergleich ohne Fakten: B-1M 21.837, D-50M 22.452.")


if __name__ == "__main__":
    main()
