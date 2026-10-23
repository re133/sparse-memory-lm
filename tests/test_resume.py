"""Checkpoints and resuming (smlm/train.py --ckpt_every_tokens / --resume, LazyRowAdam.state_dict).

Optimizer level: a lazy Adam saved after a few steps and loaded into a fresh one continues bit for bit like the
original (every moment storage, CPU and GPU paths). Training level (CPU, tiny models, the real train.main): a run
that crashes after a checkpoint and is resumed ends with the same weights, metrics and logs as a run that never
stopped. Also: a checkpoint of another run is refused, half-written files are cleaned up, and the guard against
non-finite gradients skips the step instead of poisoning the table.
"""
import csv
import io
import json
import os
import sys

import numpy as np
import pytest
import torch

from smlm import data as smlm_data
from smlm import train
from smlm.host_optim import HostLazyRowAdam
from smlm.sparse_values import LazyRowAdam, row_store

GPU = torch.cuda.is_available()
CASES = [("cpu", "torch", "fp32"), ("cpu", "torch", "bf16"), ("cpu", "torch", "int8"), ("cpu", "host", "fp32"),
         ("cpu", "host", "bf16")] + ([("cuda", impl, sd) for impl in ("torch", "triton")
                                      for sd in ("fp32", "bf16", "int8")] if GPU else [])


def lazy(table, impl, state_dtype, lr=1e-2):
    if impl == "host":
        return HostLazyRowAdam([table], lr=lr, state_dtype=state_dtype)
    return LazyRowAdam([table], lr=lr, impl=impl, state_dtype=state_dtype)


def fake_step(opt, table, gen, scale):
    """Accumulate a row-sparse gradient on a random third of the rows (as a backward would), then step."""
    st = row_store(table)
    rows = torch.randperm(table.shape[0], generator=gen)[: table.shape[0] // 3].to(table.device)
    g = torch.randn(rows.numel(), table.shape[1], generator=gen).to(table.device)
    st.acc.index_put_((rows,), g, accumulate=True)
    st.touched[rows] = True
    st.grad_scale = scale
    opt.step()


def roundtrip(sd):
    buf = io.BytesIO()
    torch.save(sd, buf)
    buf.seek(0)
    return torch.load(buf, weights_only=True)


@pytest.mark.parametrize("device,impl,state_dtype", CASES)
@pytest.mark.parametrize("steps_before", [0, 3])
def test_lazy_adam_state_continues_exactly(device, impl, state_dtype, steps_before):
    torch.manual_seed(0)
    a = torch.nn.Parameter(torch.randn(48, 16, device=device))
    opt_a = lazy(a, impl, state_dtype)
    gen_a = torch.Generator().manual_seed(1)
    for i in range(steps_before):
        fake_step(opt_a, a, gen_a, 1.0 if i % 2 else 0.5)
    b = torch.nn.Parameter(a.detach().clone())
    opt_b = lazy(b, impl, state_dtype, lr=0.5)          # other lr: must come back from the state dict
    opt_b.load_state_dict(roundtrip(opt_a.state_dict()))
    assert opt_b.param_groups[0]["lr"] == opt_a.param_groups[0]["lr"]
    gen_b = torch.Generator().manual_seed(2)
    gen_a.manual_seed(2)
    for i in range(4):
        fake_step(opt_a, a, gen_a, 0.7)
        fake_step(opt_b, b, gen_b, 0.7)
        assert torch.equal(a, b), f"step {i}"
    sa, sb = opt_a.state_dict()["state"][0], opt_b.state_dict()["state"][0]
    assert sa.keys() == sb.keys()
    for k in sa:
        assert torch.equal(sa[k].cpu(), sb[k].cpu()), k


def test_lazy_adam_refuses_other_tables():
    t = torch.nn.Parameter(torch.randn(8, 4))
    opt = LazyRowAdam([t], lr=1e-2)
    fake_step(opt, t, torch.Generator().manual_seed(0), 1.0)
    sd = opt.state_dict()
    with pytest.raises(ValueError, match="shapes"):
        LazyRowAdam([torch.nn.Parameter(torch.randn(9, 4))], lr=1e-2).load_state_dict(sd)
    with pytest.raises(ValueError, match="bf16|fp32"):
        LazyRowAdam([torch.nn.Parameter(torch.randn(8, 4))], lr=1e-2, state_dtype="bf16").load_state_dict(sd)


# ---- training level ---------------------------------------------------------------------------------------------

TINY = {
    "T-mem": dict(vocab_size=128, d_model=32, n_layers=4, n_heads=4, ffn_hidden=64, mem_layers=[1, 3], mem_n_keys=8,
                  mem_heads=2, mem_knn=4, mem_k_dim=16, mem_share_values=True, mem_value_grad="row_sparse"),
    "T-dense": dict(vocab_size=128, d_model=32, n_layers=2, n_heads=4, ffn_hidden=64),
}


@pytest.fixture
def tiny_setup(tmp_path, monkeypatch):
    d = tmp_path / "data"
    d.mkdir()
    rng = np.random.default_rng(0)
    # a little structure (repeated phrases), so the loss actually moves
    base = rng.integers(0, 128, 400)
    train_tok = np.concatenate([base[rng.integers(0, 380):][:20] for _ in range(400)]).astype(np.uint16)
    train_tok.tofile(d / "train.bin")
    train_tok[:600].tofile(d / "validation.bin")
    (d / "meta.json").write_text(json.dumps({"splits": {"validation": {"n_words": 400}, "train": {}}}))
    monkeypatch.setitem(smlm_data.DATASETS, "tiny_test", str(d))
    for name, cfg in TINY.items():
        monkeypatch.setitem(train.MODELS, name, cfg)
        monkeypatch.setitem(train.MICRO_BS, name, 2)
    monkeypatch.setattr(train, "DEVICE", "cpu")
    monkeypatch.setattr(train, "CUDA", False)
    monkeypatch.setattr(train, "inference_benchmark", lambda *a, **k: {})
    monkeypatch.setattr(train, "hardware_info", lambda: {"gpu": None, "torch": torch.__version__})
    return tmp_path


def run(monkeypatch, out, model="T-mem", extra=()):
    argv = ["train", "--model", model, "--out_dir", str(out), "--data", "tiny_test", "--tokens", str(12 * 64),
            "--seq_len", "16", "--batch_seqs", "4", "--eval_every_tokens", str(3 * 64), "--log_every", "4",
            "--ckpt_every_tokens", str(6 * 64), "--resume", "--sample_windows", "2", "--value_lr", "2.4e-3", *extra]
    monkeypatch.setattr(sys, "argv", argv)
    train.main()


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


TIMING = {"train_tok_s", "train_time_s", "wall_time_s", "tok_s"}


@pytest.mark.parametrize("model", ["T-mem", "T-dense"])
def test_crash_and_resume_equals_uninterrupted(tiny_setup, monkeypatch, model):
    a, b = tiny_setup / "a", tiny_setup / "b"
    run(monkeypatch, a, model)
    real_batch = smlm_data.TrainStream.batch

    def crashing(self, step):
        if step == 9:                                  # after the checkpoint at step 6 and the log row at step 8
            raise RuntimeError("simulated crash")
        return real_batch(self, step)
    monkeypatch.setattr(smlm_data.TrainStream, "batch", crashing)
    with pytest.raises(RuntimeError, match="simulated crash"):
        run(monkeypatch, b, model)
    info = json.loads((b / "run-info.json").read_text())
    assert info["status"] == "running" and info["checkpoint"]["step"] == 6 and (b / train.CKPT).exists()
    assert [r["step"] for r in rows(b / "train_log.csv")] == ["4", "8"]        # row 8 lies after the checkpoint
    # a half-written file of the killed process must not survive the restart
    (b / "ckpt.pt.tmp12345").write_bytes(b"x")
    monkeypatch.setattr(smlm_data.TrainStream, "batch", real_batch)
    run(monkeypatch, b, model)

    info_a, info_b = (json.loads((x / "run-info.json").read_text()) for x in (a, b))
    assert info_b["status"] == "done" and info_b["resumes"][0]["at_step"] == 6
    assert info_b["started"] <= info_b["resumes"][0]["time"]           # the first start is kept
    assert not (a / train.CKPT).exists() and not (b / train.CKPT).exists() and not (b / "ckpt.pt.tmp12345").exists()
    assert info_a["results"]["val_loss"] == info_b["results"]["val_loss"]
    sa = torch.load(a / "model.pt", weights_only=True)["state_dict"]
    sb = torch.load(b / "model.pt", weights_only=True)["state_dict"]
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    for f in ("metrics.csv", "train_log.csv"):
        ra, rb = rows(a / f), rows(b / f)
        assert [r["step"] for r in ra] == [r["step"] for r in rb]
        assert [{k: v for k, v in r.items() if k not in TIMING} for r in ra] == \
               [{k: v for k, v in r.items() if k not in TIMING} for r in rb], f
    if model == "T-mem":
        assert np.array_equal(np.load(a / "mem_access_train.npy"), np.load(b / "mem_access_train.npy"))


def test_resume_refuses_other_run(tiny_setup, monkeypatch):
    out = tiny_setup / "c"
    run(monkeypatch, out, extra=["--keep_ckpt"])
    assert (out / train.CKPT).exists()
    with pytest.raises(SystemExit, match="another run"):
        run(monkeypatch, out, extra=["--lr", "1e-3"])


def test_nonfinite_gradient_step_is_skipped(tiny_setup, monkeypatch):
    out = tiny_setup / "d"
    real = train.clip_grads
    calls = {"n": 0}

    def clip_with_spike(model, max_norm):
        calls["n"] += 1
        gn, vgn = real(model, max_norm)
        return (torch.tensor(float("inf")) if calls["n"] == 5 else gn), vgn
    monkeypatch.setattr(train, "clip_grads", clip_with_spike)
    run(monkeypatch, out, extra=["--skip_nonfinite"])
    info = json.loads((out / "run-info.json").read_text())
    assert info["status"] == "done" and info["n_skipped_steps"] == 1 and info["skipped_steps"][0]["step"] == 5
    sd = torch.load(out / "model.pt", weights_only=True)["state_dict"]
    assert all(torch.isfinite(v).all() for v in sd.values() if v.is_floating_point())


def test_too_many_nonfinite_steps_stop_as_diverged(tiny_setup, monkeypatch):
    out = tiny_setup / "e"
    real = train.clip_grads
    monkeypatch.setattr(train, "clip_grads", lambda m, c: (torch.tensor(float("nan")), real(m, c)[1]))
    with pytest.raises(SystemExit, match="non-finite"):
        run(monkeypatch, out, extra=["--skip_nonfinite", "--max_skipped_steps", "2"])
    assert json.loads((out / "run-info.json").read_text())["status"] == "diverged"


def test_tokens_or_epochs_required(tiny_setup, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train", "--model", "T-dense", "--out_dir", str(tiny_setup / "f"),
                                      "--data", "tiny_test"])
    with pytest.raises(SystemExit):
        train.main()
    assert not os.path.exists(tiny_setup / "f" / "run-info.json")


def test_weights_trajectory(tiny_setup, monkeypatch):
    """--weights_every_tokens keeps numbered weight files at evaluation steps; the last equals model.pt."""
    out = tiny_setup / "w"
    run(monkeypatch, out, model="T-dense", extra=("--weights_every_tokens", str(3 * 64)))
    files = sorted(os.listdir(out / "weights"), key=lambda f: int(f[:-3]))
    assert [int(f[:-3]) for f in files] == [3 * 64, 6 * 64, 9 * 64, 12 * 64]
    last = torch.load(out / "weights" / files[-1], weights_only=False)["state_dict"]
    final = torch.load(out / "model.pt", weights_only=False)["state_dict"]
    assert all(torch.equal(last[k], final[k]) for k in final)


@pytest.mark.parametrize("extra", [(), ("--value_optim", "rowrms"), ("--value_device", "host"),
                                   ("--value_device", "host", "--value_optim", "rowrms"), ("--ckpt_dir", "CKPT")])
def test_pause_and_resume_equals_uninterrupted(tiny_setup, monkeypatch, extra):
    """SIGUSR1 (scripts/heim.py): checkpoint at the next log row, exit 75; --resume ends like a run that never paused,
    also between two checkpoints of --ckpt_every_tokens and with a row-wise table optimizer."""
    import signal
    a, b = tiny_setup / "a", tiny_setup / "b"
    ckpt_dir = tiny_setup / "shm"
    extra = tuple(str(ckpt_dir) if x == "CKPT" else x for x in extra)
    run(monkeypatch, a, extra=extra)
    real_batch = smlm_data.TrainStream.batch

    def pausing(self, step):
        if step == 4:                                 # requested during step 5, saved at the log row of step 8
            os.kill(os.getpid(), signal.SIGUSR1)
        return real_batch(self, step)
    monkeypatch.setattr(smlm_data.TrainStream, "batch", pausing)
    with pytest.raises(SystemExit) as e:
        run(monkeypatch, b, extra=extra)
    assert e.value.code == train.PAUSED
    info = json.loads((b / "run-info.json").read_text())
    assert info["status"] == "paused" and info["checkpoint"]["step"] == 8 and info["pauses"][0]["at_step"] == 8
    if "--ckpt_dir" in extra:                         # the checkpoint is there, not in the run's folder
        assert (ckpt_dir / train.CKPT).exists() and not (b / train.CKPT).exists()
    monkeypatch.setattr(smlm_data.TrainStream, "batch", real_batch)
    run(monkeypatch, b, extra=extra)
    info_a, info_b = (json.loads((x / "run-info.json").read_text()) for x in (a, b))
    assert info_b["status"] == "done" and info_b["resumes"][0]["at_step"] == 8 and len(info_b["pauses"]) == 1
    assert not (ckpt_dir / train.CKPT).exists()
    assert info_a["results"]["val_loss"] == info_b["results"]["val_loss"]
    sa = torch.load(a / "model.pt", weights_only=True)["state_dict"]
    sb = torch.load(b / "model.pt", weights_only=True)["state_dict"]
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    for f in ("metrics.csv", "train_log.csv"):
        assert [{k: v for k, v in r.items() if k not in TIMING} for r in rows(a / f)] == \
               [{k: v for k, v in r.items() if k not in TIMING} for r in rows(b / f)], f


@pytest.mark.parametrize("fast", [("--hot_frac", "0.25"), ("--hot_frac", "0.25", "--step_rows", "64"),
                                  ("--hot_frac", "0", "--step_rows", "64"), ("--hot_frac", "0.25", "--step_rows", "8")])
def test_hot_rows_train_and_resume_like_rows_in_ram(tiny_setup, monkeypatch, fast):
    """--hot_rows / --step_rows: same run within float rounding as all rows in RAM; paused and resumed, bit for bit
    as unpaused."""
    import signal
    reads = tiny_setup / "reads.npy"
    np.save(reads, np.random.default_rng(0).integers(0, 100, 64))
    host = ("--value_device", "host", "--value_optim", "rowrms")
    hot = host + ("--hot_rows", str(reads)) + fast
    run(monkeypatch, tiny_setup / "ram", extra=host)
    run(monkeypatch, tiny_setup / "hot", extra=hot)
    if "8" in fast:                                     # 8 slots: most steps overflow into RAM
        assert json.loads((tiny_setup / "hot" / "run-info.json").read_text())["step_rows_spilled_steps"] > 0
    sa = torch.load(tiny_setup / "ram" / "model.pt", weights_only=True)["state_dict"]
    sb = torch.load(tiny_setup / "hot" / "model.pt", weights_only=True)["state_dict"]
    for k in sa:
        torch.testing.assert_close(sb[k], sa[k], rtol=1e-4, atol=1e-5)
    real_batch = smlm_data.TrainStream.batch

    def pausing(self, step):
        if step == 4:
            os.kill(os.getpid(), signal.SIGUSR1)
        return real_batch(self, step)
    monkeypatch.setattr(smlm_data.TrainStream, "batch", pausing)
    with pytest.raises(SystemExit):
        run(monkeypatch, tiny_setup / "hot2", extra=hot)
    monkeypatch.setattr(smlm_data.TrainStream, "batch", real_batch)
    run(monkeypatch, tiny_setup / "hot2", extra=hot)
    sc = torch.load(tiny_setup / "hot2" / "model.pt", weights_only=True)["state_dict"]
    assert all(torch.equal(sb[k], sc[k]) for k in sb)
