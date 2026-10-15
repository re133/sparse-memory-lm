"""Step 13, checked before the runs: resuming on a real GPU with the real model (B-1M, Triton kernels).

  python scripts/check_resume.py [OUT]        (default runs/resume_check; ~25 min on the RX 9070)

1. curve: R1 and R2 run 12 M tokens without a stop (their difference = GPU noise, kernel sums aren't ordered). K is
   killed with SIGKILL after its checkpoint at 4 M tokens, resumed, killed again while it writes the checkpoint at 8 M
   (the half-written file must be cleaned up, the run continues from 4 M), resumed to the end. Rule in the spirit of
   step 8 (resumed no further off than a second run): the largest relative distance of K's val loss from R1's over
   all evaluations is at most twice that of R2 from R1, plus 1e-4 (K is a noisy run itself, so not per evaluation).
2. preflight: the two-part preflight of scripts/run_dense.py (stop at 1.5 M with a checkpoint, resume to 3 M), with
   exactly its arguments.
3. speed: B-1M and A for 3 M tokens with and without --skip_nonfinite (one GPU sync per step), median tok/s.
Result: OUT/check_resume.json, plus a line per check on stdout.
"""
import csv
import glob
import json
import os
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(ROOT, "runs", "resume_check")
PY = sys.executable
B1M = ["--model", "B-1M-sparse", "--mem_impl", "triton", "--value_lr", "2.4e-3"]
COMMON = ["--data", "wikipedia", "--extra_val", "wikitext103", "--seed", "0", "--data_seed", "1234", "--skip_nonfinite"]
CURVE = ["--tokens", "12e6", "--eval_every_tokens", "2e6", "--ckpt_every_tokens", "4e6", "--resume"]


def cmd(name, args):
    return [PY, "-m", "smlm.train", "--out_dir", os.path.join(OUT, name), *args]


def run(name, args):
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "stdout.log"), "a") as f:
        r = subprocess.run(cmd(name, args), cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    return r.returncode


def info(name):
    return json.load(open(os.path.join(OUT, name, "run-info.json")))


def metrics(name):
    with open(os.path.join(OUT, name, "metrics.csv")) as f:
        return {int(r["step"]): float(r["val_loss"]) for r in csv.DictReader(f)}


def run_until(name, args, until):
    """Start the run and SIGKILL it as soon as until(dir) is true; returns what was seen."""
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "stdout.log"), "a") as f:
        p = subprocess.Popen(cmd(name, args), cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        while p.poll() is None:
            seen = until(d)
            if seen:
                p.send_signal(signal.SIGKILL)
                p.wait()
                return seen
            time.sleep(0.05)
    raise SystemExit(f"{name} ended (rc {p.returncode}) before it could be killed")


def ckpt_step(d):
    try:
        return (json.load(open(os.path.join(d, "run-info.json"))).get("checkpoint") or {}).get("step")
    except (OSError, ValueError):
        return None


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {}
    # 1. curve
    for name in ("R1", "R2"):
        if not os.path.exists(os.path.join(OUT, name, "model.pt")):
            assert run(name, B1M + COMMON + CURVE) == 0, name
    if not os.path.exists(os.path.join(OUT, "K", "model.pt")):
        first = run_until("K", B1M + COMMON + CURVE, lambda d: ckpt_step(d) == 122)             # 4 M tokens
        tmp = run_until("K", B1M + COMMON + CURVE,
                        lambda d: glob.glob(os.path.join(d, "ckpt.pt.tmp*")) and "writing ckpt at 8M")
        k_mid = {"after_first_kill_ckpt_step": first, "second_kill": tmp, "ckpt_step_after_second_kill": ckpt_step(
            os.path.join(OUT, "K")), "tmp_files_left": glob.glob(os.path.join(OUT, "K", "ckpt.pt.tmp*"))}
        assert run("K", B1M + COMMON + CURVE) == 0
        res["kill"] = k_mid
    r1, r2, k = metrics("R1"), metrics("R2"), metrics("K")
    rows = []
    for s in sorted(r1):
        noise = abs(r2[s] - r1[s]) / r1[s]
        dk = abs(k[s] - r1[s]) / r1[s]
        rows.append({"step": s, "R1": r1[s], "R2": r2[s], "K": k[s], "noise_rel": noise, "K_rel": dk})
    ik = info("K")
    max_noise, max_k = max(r["noise_rel"] for r in rows), max(r["K_rel"] for r in rows)
    res["curve"] = {"rows": rows, "max_noise_rel": max_noise, "max_K_rel": max_k,
                    "ok": max_k <= 2 * max_noise + 1e-4 and sorted(k) == sorted(r1),
                    "resumes": ik.get("resumes"), "status": ik["status"],
                    "ckpt_left": os.path.exists(os.path.join(OUT, "K", "ckpt.pt")),
                    "final_val_ppl": {n: info(n)["results"]["val_ppl"] for n in ("R1", "R2", "K")}}
    print(f"[curve] ok={res['curve']['ok']} max rel K {max_k:.2e} noise {max_noise:.2e} final ppl {res['curve']['final_val_ppl']} resumes "
          f"{[r['at_step'] for r in ik.get('resumes', [])]}", flush=True)
    # 2. preflight exactly as run_dense.py builds it
    pre = B1M + ["--data", "wikipedia", "--tokens", "3e6", "--extra_val", "wikitext103", "--eval_every_tokens", "1.5e6",
                 "--seed", "0", "--data_seed", "1234", "--skip_nonfinite"]
    if not os.path.exists(os.path.join(OUT, "P", "model.pt")) or not info("P").get("resumes"):
        assert run("P", pre + ["--ckpt_every_tokens", "1.5e6", "--stop_after_tokens", "1.5e6", "--keep_ckpt"]) == 0
        mid = {"status": info("P")["status"], "ckpt": os.path.exists(os.path.join(OUT, "P", "ckpt.pt"))}
        assert run("P", pre + ["--ckpt_every_tokens", "1.5e6", "--resume"]) == 0
        res["preflight_first_part"] = mid
    ip = info("P")
    res["preflight"] = {"status": ip["status"], "resumes": ip.get("resumes"), "checkpoint": ip.get("checkpoint"),
                        "ckpt_left": os.path.exists(os.path.join(OUT, "P", "ckpt.pt")),
                        "tok_s_median": ip["results"]["train_tok_s_median"]}
    print(f"[preflight] {res['preflight']}", flush=True)
    # 3. speed of the guard
    speed = {}
    for label, model in (("B-1M", B1M), ("A", ["--model", "A"])):
        for skip in (True, False):
            name = f"S-{label}-{'skip' if skip else 'noskip'}"
            args = model + [a for a in COMMON if skip or a != "--skip_nonfinite"] + ["--tokens", "3e6",
                                                                                     "--eval_every_tokens", "3e6",
                                                                                     "--no_save"]
            if not os.path.exists(os.path.join(OUT, name, "run-info.json")) or info(name)["status"] != "done":
                assert run(name, args) == 0, name
            speed[name] = info(name)["results"]["train_tok_s_median"]
    res["speed"] = speed
    print(f"[speed] {speed}", flush=True)
    json.dump(res, open(os.path.join(OUT, "check_resume.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
