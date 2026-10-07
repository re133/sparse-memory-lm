"""Compare training throughput, VRAM and matched-seed loss curves on a GPU.

  python scripts/bench_train_speed.py --out_dir .scratch/spd --mode speed
  python scripts/bench_train_speed.py --out_dir .scratch/spd-curves --mode curves

Each flag / micro-batch combination runs in a fresh process. Speed excludes warmup updates;
curves restart from the same initialization and token order without warmup updates. Changing
the micro-batch changes PKM BatchNorm statistics, so curves also compare to a same-micro baseline.
"""
import argparse
import csv
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_DATA = ROOT.parent / "AngryAnt" / "data" / "wikipedia_en_gpt2"


def make_cases(presets, micro_sizes, batch_seqs, micro_table):
    """The table default is always included; an omitted override means twice the default."""
    cases = []
    for preset in dict.fromkeys(presets):
        default = micro_table[preset]
        sizes = list(dict.fromkeys([default, *(micro_sizes or [2 * default])]))
        for micro in sizes:
            if micro < 1 or batch_seqs % micro:
                raise ValueError(f"{preset}: micro_bs={micro} must be positive and divide batch_seqs={batch_seqs}")
            for fused, compiled in itertools.product((False, True), repeat=2):
                cases.append({"id": f"{preset}-ce{int(fused)}-compile{int(compiled)}-mb{micro}",
                              "preset": preset, "fused_ce": fused, "compile": compiled, "micro_bs": micro,
                              "default_micro_bs": default, "micro_override": micro != default})
    return cases


def curve_schedule(tokens, eval_tokens, tokens_per_step):
    if not math.isfinite(tokens) or not math.isfinite(eval_tokens) or tokens < tokens_per_step or eval_tokens <= 0:
        raise ValueError("curve_tokens must cover one optimizer step; eval_every_tokens must be positive and finite")
    steps = int(tokens) // tokens_per_step
    interval = max(1, int(round(eval_tokens / tokens_per_step)))
    return steps, sorted({0, steps, *range(interval, steps + 1, interval)})


def output_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT) or path == ROOT:
        raise ValueError("out_dir must be a subdirectory of this worktree")
    return path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def comparisons(results):
    """Align on actual token counts, retaining both references when micro-batches differ."""
    baselines = {}
    for result in results:
        case = result["case"]
        if result["status"] == "done" and not case["fused_ce"] and not case["compile"]:
            for row in result.get("curves", []):
                baselines[case["preset"], case["micro_bs"], row["tokens"]] = row["val_loss"]
    rows = []
    for result in results:
        case = result["case"]
        for row in result.get("curves", []):
            reference = baselines.get((case["preset"], case["default_micro_bs"], row["tokens"]))
            same_micro = baselines.get((case["preset"], case["micro_bs"], row["tokens"]))
            rows.append({"case": case["id"], "preset": case["preset"], "tokens": row["tokens"],
                         "val_loss": row["val_loss"], "baseline_val_loss": reference,
                         "delta_default_micro": None if reference is None else row["val_loss"] - reference,
                         "same_micro_baseline_val_loss": same_micro,
                         "delta_same_micro": None if same_micro is None else row["val_loss"] - same_micro,
                         "micro_batch_changes_batchnorm": case["micro_override"]
                         and result.get("model_config", {}).get("mem_query_norm") == "batchnorm"
                         and bool(result.get("model_config", {}).get("mem_layers"))})
    return rows


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(out_dir, results):
    write_json(out_dir / "results.json", results)
    speeds, curves = [], []
    for result in results:
        case = result["case"]
        common = {"case": case["id"], "preset": case["preset"], "fused_ce": case["fused_ce"],
                  "compile": case["compile"], "micro_bs": case["micro_bs"], "status": result["status"]}
        if result["mode"] == "speed":
            speeds.append({**common, **result.get("speed", {}), "error": result.get("error", "")})
        curves.extend({**common, **row} for row in result.get("curves", []))
    write_csv(out_dir / "speed.csv", speeds, ["case", "preset", "fused_ce", "compile", "micro_bs", "status",
              "steps", "tokens", "elapsed_s", "tok_s", "peak_allocated_gib", "peak_reserved_gib",
              "warmup_s", "warmup_peak_allocated_gib", "warmup_peak_reserved_gib", "mean_loss", "error"])
    write_csv(out_dir / "curves.csv", curves, ["case", "preset", "fused_ce", "compile", "micro_bs", "status",
              "step", "tokens", "train_loss", "val_loss", "val_tokens", "lr_mult"])
    write_csv(out_dir / "comparisons.csv", comparisons(results), ["case", "preset", "tokens", "val_loss",
              "baseline_val_loss", "delta_default_micro", "same_micro_baseline_val_loss", "delta_same_micro",
              "micro_batch_changes_batchnorm"])


def run_worker(config, result):
    # Set these before importing torch / smlm.data: neither caches nor token inputs belong to the main repo.
    setup_start = time.perf_counter()
    os.environ["SMLM_DATA_DIR"] = config["data_dir"]
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "4"
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    os.environ["TRITON_CACHE_DIR"] = str(ROOT / ".cache" / "triton")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(ROOT / ".cache" / "inductor")
    import numpy as np
    import torch
    from smlm.compile import compile_dense
    from smlm.data import TrainStream, load_meta
    from smlm.model import ModelConfig, Transformer
    from smlm.optim import build_optimizer, clip_grads, lr_multiplier, set_lr
    from smlm.train import MODELS, evaluate, git_info

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a visible GPU; CPU speed results would not answer the question")
    case = config["case"]
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    stream = TrainStream(config["seq_len"], config["batch_seqs"], config["data_seed"])
    if stream.steps_per_epoch < 1:
        raise ValueError("training split is too short for one effective batch")
    cfg = ModelConfig(max_seq_len=config["seq_len"], **MODELS[case["preset"]])
    cfg.mem_impl, cfg.eng_impl = config["mem_impl"], config["eng_impl"]
    cfg.mem_query_norm = config["mem_query_norm"]
    model = Transformer(cfg).cuda()
    if case["compile"]:
        compile_dense(model)
    opt = build_optimizer(model, config["lr"], config["value_lr"], config["weight_decay"],
                          eng_value_lr=config["eng_value_lr"])
    props = torch.cuda.get_device_properties(0)
    result.update({"model_config": cfg.to_dict(), "git": git_info(), "tokens_per_step": stream.tokens_per_step,
                   "hardware": {"gpu": props.name, "arch": getattr(props, "gcnArchName", None),
                                "total_vram_gib": props.total_memory / 2**30,
                                "torch": torch.__version__, "hip": torch.version.hip},
                   "curves": []})
    mems = model.memory_layers()
    size = mems[0].size if mems else 0
    counts_interval = torch.zeros(size, dtype=torch.int64, device="cuda") if mems else None
    counts_total = torch.zeros(size, dtype=torch.int64, device="cuda") if mems else None
    for mem in mems:
        mem.record = True
    micro, accum = case["micro_bs"], config["batch_seqs"] // case["micro_bs"]
    if config["mode"] == "curves":
        total_steps, checkpoints = curve_schedule(config["curve_tokens"], config["eval_every_tokens"],
                                                  stream.tokens_per_step)
    else:
        total_steps, checkpoints = config["warmup_steps"] + config["steps"], []
    warmup_steps = max(1, int(round(config["warmup_frac"] * total_steps)))
    result["schedule"] = {"total_steps": total_steps, "total_tokens": total_steps * stream.tokens_per_step,
                          "warmup_steps": warmup_steps, "eval_steps": checkpoints}
    last_logits = None                  # match train.py's retained `_` output, including across optimizer steps

    def train_step(step):
        nonlocal last_logits
        set_lr(opt, lr_multiplier(step, total_steps, warmup_steps, config["min_lr_ratio"]))
        batch = torch.from_numpy(stream.batch(step).astype(np.int64)).pin_memory().cuda(non_blocking=True)
        loss_sum = torch.zeros((), device="cuda")
        for i in range(accum):
            mb = batch[i * micro:(i + 1) * micro]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                last_logits, loss = model(mb[:, :-1], mb[:, 1:], fused_ce=case["fused_ce"],
                                          ce_chunk_size=config["ce_chunk_size"])
            (loss / accum).backward()
            loss_sum += loss.detach() / accum
            for mem in mems:
                count = torch.bincount(mem.last_indices.reshape(-1), minlength=size)
                counts_interval.add_(count)
                counts_total.add_(count)
        gn, vgn = clip_grads(model, config["clip"])
        opt.step()
        opt.zero_grad(set_to_none=True)
        return torch.stack((loss_sum, gn, vgn if vgn is not None else torch.zeros_like(gn)))

    def check_loss(values):
        if not all(math.isfinite(v) for v in values):
            raise RuntimeError(f"non-finite training loss or gradient norm: {values}")

    def do_eval(step, train_loss):
        meta = load_meta()
        ev = evaluate(model, "validation", config["seq_len"], meta["splits"]["validation"]["n_words"],
                      batch=micro, fused_ce=case["fused_ce"], ce_chunk_size=config["ce_chunk_size"])
        if not math.isfinite(ev["loss"]):
            raise RuntimeError(f"non-finite validation loss at step {step}")
        for mem in mems:
            mem.record = True
        if mems:
            counts_interval.zero_()
        result["curves"].append({"step": step, "tokens": step * stream.tokens_per_step,
                                 "train_loss": train_loss, "val_loss": ev["loss"], "val_tokens": ev["n_tokens"],
                                 "lr_mult": lr_multiplier(max(0, step - 1), total_steps, warmup_steps,
                                                          config["min_lr_ratio"]) if step else 0.0})
        write_json(config["result_path"], result)
        print(f"{case['id']} step={step} tokens={step * stream.tokens_per_step} val_loss={ev['loss']:.6f}",
              flush=True)

    model.train()
    torch.cuda.synchronize()
    result["setup_s"] = time.perf_counter() - setup_start
    if config["mode"] == "curves":
        do_eval(0, None)
        loss_sum, count = torch.zeros(3, device="cuda"), 0
        for step in range(total_steps):
            loss_sum += train_step(step)
            count += 1
            if (step + 1) % config["log_every"] == 0:
                check_loss(loss_sum.tolist())
            if step + 1 in checkpoints:
                values = loss_sum.tolist()
                check_loss(values)
                do_eval(step + 1, values[0] / count)
                loss_sum.zero_()
                count = 0
        return

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for step in range(config["warmup_steps"]):
        check_loss(train_step(step).tolist())
    torch.cuda.synchronize()
    warmup_s = time.perf_counter() - start
    warmup_allocated = torch.cuda.max_memory_allocated() / 2**30
    warmup_reserved = torch.cuda.max_memory_reserved() / 2**30
    torch.cuda.reset_peak_memory_stats()
    loss_sum = torch.zeros(3, device="cuda")
    start = time.perf_counter()
    for step in range(config["steps"]):
        loss_sum += train_step(step + config["warmup_steps"])
        if (step + 1) % config["log_every"] == 0:
            check_loss(loss_sum.tolist())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    values = loss_sum.tolist()
    check_loss(values)
    result["speed"] = {"steps": config["steps"], "tokens": config["steps"] * stream.tokens_per_step,
                       "elapsed_s": elapsed, "tok_s": config["steps"] * stream.tokens_per_step / elapsed,
                       "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                       "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                       "warmup_s": warmup_s, "warmup_peak_allocated_gib": warmup_allocated,
                       "warmup_peak_reserved_gib": warmup_reserved, "mean_loss": values[0] / config["steps"]}


def worker(path):
    config = json.loads(Path(path).read_text())
    output_path(config["result_path"])
    result = {"case": config["case"], "mode": config["mode"], "status": "running", "config": config}
    start = time.perf_counter()
    try:
        run_worker(config, result)
        result["status"] = "done"
    except Exception as error:
        import torch
        result["status"] = "oom" if isinstance(error, torch.cuda.OutOfMemoryError) else "failed"
        result["error"] = str(error)
        result["traceback"] = traceback.format_exc()
        traceback.print_exc()
    result["worker_elapsed_s"] = time.perf_counter() - start
    write_json(config["result_path"], result)
    return 0 if result["status"] == "done" else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", help="new or empty output directory inside this worktree")
    ap.add_argument("--presets", nargs="+", default=["A", "B-1M-sparse", "E-1M"])
    ap.add_argument("--mode", choices=["speed", "curves", "all"], default="all")
    ap.add_argument("--micro_bs", type=int, nargs="+", help="additional micro-batches; default: twice each table value")
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA), help="absolute read-only token dataset directory")
    ap.add_argument("--steps", type=int, default=200, help="timed optimizer steps, excluding warmup")
    ap.add_argument("--warmup_steps", type=int, default=10, help="speed-only updates, including compilation")
    ap.add_argument("--curve_tokens", type=float, default=20e6)
    ap.add_argument("--eval_every_tokens", type=float, default=2e6)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--batch_seqs", type=int, default=32)
    ap.add_argument("--ce_chunk_size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_seed", type=int, default=1234)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--value_lr", type=float, default=1e-3)
    ap.add_argument("--eng_value_lr", type=float)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--min_lr_ratio", type=float, default=0.1)
    ap.add_argument("--mem_impl", choices=["torch", "triton"], default="triton")
    ap.add_argument("--eng_impl", choices=["torch", "triton"], default="triton")
    ap.add_argument("--mem_query_norm", choices=["batchnorm", "none"], default="batchnorm")
    ap.add_argument("--log_every", type=int, default=10, help="synchronization cadence, matching train.py")
    ap.add_argument("--dry_run", action="store_true", help="print the complete plan without GPU access or writes")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.worker:
        return worker(args.worker)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "4"
    from smlm.train import MICRO_BS, MODELS
    try:
        if not args.out_dir:
            raise ValueError("--out_dir is required")
        out_dir = output_path(args.out_dir)
        for name in ("steps", "seq_len", "batch_seqs", "ce_chunk_size", "log_every"):
            if getattr(args, name) < 1:
                raise ValueError(f"--{name} must be positive")
        if args.warmup_steps < 1:
            raise ValueError("--warmup_steps must be positive so compilation is excluded from timing")
        if not set(args.presets).issubset(MODELS):
            raise ValueError(f"unknown presets: {sorted(set(args.presets) - MODELS.keys())}")
        if not Path(args.data_dir).is_absolute():
            raise ValueError("--data_dir must be absolute")
        cases = make_cases(args.presets, args.micro_bs, args.batch_seqs, MICRO_BS)
        steps, checkpoints = curve_schedule(args.curve_tokens, args.eval_every_tokens, args.seq_len * args.batch_seqs)
    except ValueError as error:
        ap.error(str(error))
    settings = {k: v for k, v in vars(args).items() if k not in ("worker", "dry_run", "presets", "out_dir", "micro_bs")}
    plan = {"settings": settings, "cases": cases, "curve_steps": steps, "curve_eval_steps": checkpoints,
            "curve_actual_tokens": steps * args.seq_len * args.batch_seqs,
            "measurement": "host input transfer, forward/backward, PKM usage bincounts, gradient clipping, Adam; "
                           "validation, disk logging and checkpoint writes excluded from speed",
            "compilation": "fresh workers share disk compiler caches; warmup includes compilation/cache lookup "
                           "but is not a guaranteed cold compile measurement",
            "caution": "PKM BatchNorm depends on micro_bs; use delta_same_micro to isolate fused_ce / compile. "
                       "Fixed seed is not a guarantee of bitwise GPU determinism. Run on an otherwise idle GPU."}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    for name in ("train.bin", "validation.bin", "meta.json"):
        if not (Path(args.data_dir) / name).is_file():
            ap.error(f"missing input: {Path(args.data_dir) / name}")
    if out_dir.exists() and any(out_dir.iterdir()):
        ap.error("out_dir must be empty; previous measurements will not be overwritten")
    import torch
    if not torch.cuda.is_available():
        ap.error("a visible GPU is required; use --dry_run to inspect the plan on CPU")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "plan.json", plan)
    results = []
    modes = ["speed", "curves"] if args.mode == "all" else [args.mode]
    for mode in modes:
        for case in cases:
            prefix = out_dir / f"{mode}-{case['id']}"
            result_path = str(prefix) + ".json"
            config_path = str(prefix) + "-config.json"
            config = {**settings, "case": case, "mode": mode, "result_path": result_path}
            write_json(config_path, config)
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", config_path]
            print(f"[{mode}] {case['id']} (log: {prefix}.log)", flush=True)
            env = {**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
                   "NUMEXPR_NUM_THREADS": "4", "PYTHONDONTWRITEBYTECODE": "1"}
            with open(str(prefix) + ".log", "w") as log:
                proc = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            if Path(result_path).exists():
                result = json.loads(Path(result_path).read_text())
            else:
                result = {"case": case, "mode": mode, "status": "failed", "error": "worker exited without results"}
            if proc.returncode and result["status"] in ("running", "done"):
                result.update({"status": "failed", "error": "worker did not finish normally"})
            result["returncode"], result["command"] = proc.returncode, command
            results.append(result)
            summarize(out_dir, results)
            print(f"  {result['status']}" + (f" {result['speed']['tok_s']:.0f} tok/s" if "speed" in result else ""),
                  flush=True)
    return 0 if all(r["status"] == "done" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
