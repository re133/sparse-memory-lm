"""Step 12 (LOOP) at home (RX 9070): B-1M and B-1M-r2 (a second read of the table in layer 10) on the step 11 data
plus 200 demonstrated pairs (REPORT.md, "Step 12").

  python scripts/run_hop2d.py [--deadline 2026-10-12T10:00]  (as a systemd unit; resumable, a finished run is skipped)
  python scripts/run_hop2d.py --smoke DIR                      (both runs on 2M tokens into DIR, no ntfy)

Same arguments as step 11's B-1M run, only --data (scripts/make_hop2.py --demo) and, for r2, --model differ. Output
runs/hop2d/<name>/ with GPU temperature / power every 10 s. ntfy message after every run. Nothing is pushed.
"""
import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import run_shape  # noqa: E402

run_shape.OUT = os.path.join(ROOT, "runs", "hop2d")
run_shape.TITLE = "SMLM LOOP"
run_shape.LOG = "queue_home.log"
run_shape.COMMON = ["--data", "wikipedia_hop2d", "--tokens", "500e6", "--extra_val", "wikitext103",
                    "--eval_every_tokens", "10e6", "--seed", "0", "--data_seed", "1234"]
run_shape.RUNS = [
    ("B-1M-hop2d-s0", ["--model", "B-1M-sparse", "--value_lr", "2.4e-3", "--mem_impl", "triton"]),
    ("B-1M-r2-hop2d-s0", ["--model", "B-1M-r2-sparse", "--value_lr", "2.4e-3", "--mem_impl", "triton"]),
]
run_shape.EST_H = {"B-1M-hop2d-s0": 2.6, "B-1M-r2-hop2d-s0": 3.2}   # step 11: 2.61 h; r2 from the smoke test


def main():
    argv, deadline = sys.argv[1:], None
    if len(argv) == 2 and argv[0] == "--deadline":
        deadline = datetime.fromisoformat(argv[1])
    elif len(argv) == 2 and argv[0] == "--smoke":
        run_shape.OUT = run_shape.SMOKE = os.path.abspath(argv[1])
    elif argv:
        sys.exit(__doc__)
    os.makedirs(run_shape.OUT, exist_ok=True)
    run_shape.log(f"LOOP queue start: {[n for n, _ in run_shape.RUNS]}"
                  + (f", deadline {deadline:%Y-%m-%d %H:%M}" if deadline else ""))
    results = {name: run_shape.run(name, args, deadline) for name, args in run_shape.RUNS}
    failed = [n for n, r in results.items() if r != "done"]
    ppls = {n: run_shape.info(os.path.join(run_shape.OUT, n)).get("results", {}).get("val_ppl") for n, _ in run_shape.RUNS}
    summary = ", ".join(f"{n} {p:.3f}" for n, p in ppls.items() if p)
    run_shape.log("QUEUE DONE" + (f", NOT finished: {failed}" if failed else "") + f" | {summary}")
    run_shape.notify(("LOOP-Nacht fertig" if not failed else f"LOOP UNVOLLSTÄNDIG ({', '.join(failed)})")
                     + f": {summary}. Vergleich: B-1M ohne Paare 21.837, mit HOP2-Paaren 21.920.")


if __name__ == "__main__":
    main()
