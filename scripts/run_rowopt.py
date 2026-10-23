"""Step 18 (HEIM 2) at home (RX 9070): one optimizer state number per table row instead of Adam's two moments per
value (REPORT.md, "Step 18").

  python scripts/run_rowopt.py screen [--deadline 2026-10-23T07:00]   part A: B-1M on 50 M tokens, one run after another
  python scripts/run_rowopt.py big VALUE_OPTIM VALUE_LR [STOP]        part B: B-16M, table in RAM, BIG's arguments
  python scripts/run_rowopt.py --smoke DIR screen                     part A on 1 M tokens into DIR, no ntfy

Part A: B-1M-sparse as in runs/hampter (Triton, table LR 2.4e-3 for Adam), but a full schedule over 50 M tokens, no
model.pt. Adam twice (seeds 0 and 1, for the spread), rowrms and rowadagrad at three table LRs each (seed 0). The runs
with a row-wise optimizer stop at 20 M tokens if they are more than 5 % worse than Adam seed 0 there (--abort_ref).
Part B: exactly the arguments of runs/big_home/B-16M-host-s0 (step 8) with --value_optim and --value_lr from part A;
STOP (default 99.9e6) ends it after that evaluation. Output runs/rowopt/<name>/ with GPU temperature / power every
10 s. A run counts as done with run-info.json status "done" (or "aborted" by the abort rule). A run is only started if
its estimated end lies before --deadline. ntfy message after every run. Nothing is pushed.
"""
import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import run_shape  # noqa: E402

run_shape.OUT = os.path.join(ROOT, "runs", "rowopt")
run_shape.TITLE = "SMLM Schritt 18"
SCREEN = ["--model", "B-1M-sparse", "--mem_impl", "triton", "--data", "wikipedia", "--tokens", "50e6",
          "--extra_val", "wikitext103", "--eval_every_tokens", "10e6", "--data_seed", "1234", "--seq_len", "1024",
          "--batch_seqs", "32", "--lr", "6e-4", "--weight_decay", "0.1", "--no_save"]
ABORT = ["--abort_ref", os.path.join(run_shape.OUT, "B-1M-adam-s0", "metrics.csv"), "--abort_at_tokens", "20e6",
         "--abort_max_rel", "0.05"]
RUNS = [("B-1M-adam-s0", ["--value_optim", "adam", "--value_lr", "2.4e-3", "--seed", "0"])]
RUNS += [(f"B-1M-rowrms-{lr}-s0", ["--value_optim", "rowrms", "--value_lr", lr, "--seed", "0", *ABORT])
         for lr in ("1.2e-3", "2.4e-3", "0.6e-3")]
RUNS += [("B-1M-adam-s1", ["--value_optim", "adam", "--value_lr", "2.4e-3", "--seed", "1"])]
RUNS += [(f"B-1M-rowadagrad-{lr}-s0", ["--value_optim", "rowadagrad", "--value_lr", lr, "--seed", "0", *ABORT])
         for lr in ("0.03", "0.1", "0.01")]
BIG = ["--model", "B-16M-sparse", "--value_device", "host", "--value_state", "fp32", "--mem_impl", "triton",
       "--seed", "0", "--data", "wikipedia", "--tokens", "500e6", "--extra_val", "wikitext103",
       "--eval_every_tokens", "10e6", "--data_seed", "1234", "--seq_len", "1024", "--batch_seqs", "32", "--lr", "6e-4",
       "--weight_decay", "0.1", "--no_save"]
BIG_ADAM = {1525: 58.056, 3050: 37.433}            # runs/big_home/B-16M-host-s0 (Adam, table in RAM), val PPL


def done(d):
    return run_shape.info(d).get("status") in ("done", "aborted")


def main():
    argv, deadline = sys.argv[1:], None
    if len(argv) >= 2 and argv[0] == "--smoke":
        run_shape.OUT = run_shape.SMOKE = os.path.abspath(argv[1])
        argv = argv[2:]
    if len(argv) == 3 and argv[0] == "screen" and argv[1] == "--deadline":
        deadline, argv = datetime.fromisoformat(argv[2]), argv[:1]
    run_shape.done = done
    os.makedirs(run_shape.OUT, exist_ok=True)
    if argv == ["screen"]:
        common = SCREEN
        if run_shape.SMOKE:
            common = [("1e6" if a == "50e6" else "0.5e6" if a == "10e6" else a) for a in SCREEN]
        run_shape.COMMON = common
        runs = [(n, [a.replace(os.path.join(ROOT, "runs", "rowopt"), run_shape.OUT) for a in args])
                for n, args in RUNS]
        run_shape.EST_H = {n: 0.32 for n, _ in runs}        # smoke test: ~58k tok/s, 5 evaluations, start-up
        run_shape.log(f"step 18 screen start: {[n for n, _ in runs]}"
                      + (f", deadline {deadline:%Y-%m-%d %H:%M}" if deadline else ""))
        results = {n: run_shape.run(n, a, deadline) for n, a in runs}
        ppls = {n: run_shape.info(os.path.join(run_shape.OUT, n)).get("results", {}).get("val_ppl") for n, _ in runs}
        summary = ", ".join(f"{n[5:]} {p:.2f}" for n, p in ppls.items() if p)
        failed = [n for n, r in results.items() if r != "done"]
        run_shape.log("QUEUE DONE" + (f", not done: {failed}" if failed else "") + f" | {summary}")
        run_shape.notify("Schritt 18 Teil A fertig: " + summary)
    elif len(argv) in (3, 4) and argv[0] == "big":
        optim, lr = argv[1], argv[2]
        stop = argv[3] if len(argv) == 4 else "99.9e6"
        run_shape.COMMON = BIG + ["--stop_after_tokens", "1e6" if run_shape.SMOKE else stop]
        name = f"B-16M-host-{optim}-{lr}-s0"
        run_shape.EST_H = {name: 4.5}
        run_shape.log(f"step 18 part B start: {name}, stop after {stop} tokens")
        result = run_shape.run(name, ["--value_optim", optim, "--value_lr", lr], None)
        run_shape.log(f"QUEUE DONE | {name} {result}")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
