"""Step 1 "Gegenwert der Tabelle": dense presets D-50M ... D-400M, the budget gate of scripts/run_dense.py and the end
of its queue on a Runpod Pod and on a plain VM (IONOS)."""
import json
import os
import sys
import time

import pytest
import torch

from smlm.model import ModelConfig, Transformer
from smlm.train import MICRO_BS, MODELS

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import run_dense  # noqa: E402

DENSE = {"D-50M": 50e6, "D-100M": 100e6, "D-200M": 200e6, "D-400M": 400e6}


@pytest.mark.parametrize("name", list(DENSE))
def test_dense_preset_shape(name):
    cfg = ModelConfig(**MODELS[name])
    assert not cfg.mem_layers
    assert cfg.d_model // cfg.n_heads == 64 and cfg.d_model % cfg.n_heads == 0
    assert cfg.ffn_hidden % 64 == 0 and abs(cfg.ffn_hidden / cfg.d_model - 8 / 3) < 0.1
    a = ModelConfig(**MODELS["A"])
    assert (cfg.vocab_size, cfg.max_seq_len, cfg.rope_theta, cfg.norm_eps, cfg.init_std) == \
        (a.vocab_size, a.max_seq_len, a.rope_theta, a.norm_eps, a.init_std)
    with torch.device("meta"):
        m = Transformer(cfg)
    pc = m.param_counts()
    assert abs(pc["non_embedding"] / DENSE[name] - 1) < 0.02
    assert pc["memory_values"] == 0
    assert 32 % MICRO_BS[name] == 0


def _tiny(seed=0):
    torch.manual_seed(seed)
    return Transformer(ModelConfig(vocab_size=97, d_model=32, n_layers=2, n_heads=2, ffn_hidden=64,
                                   max_seq_len=32)).double()


def test_micro_batch_only_changes_memory():
    """Gradient accumulation over 2 x 4 sequences == one pass over 8 (train.py: (loss / accum).backward())."""
    x = torch.randint(0, 97, (8, 17))
    grads = []
    for micro in (8, 4, 2):
        m = _tiny()
        accum = 8 // micro
        for i in range(accum):
            mb = x[i * micro:(i + 1) * micro]
            _, loss = m(mb[:, :-1], mb[:, 1:])
            (loss / accum).backward()
        grads.append(torch.cat([p.grad.flatten() for p in m.parameters()]))
    torch.testing.assert_close(grads[0], grads[1], rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(grads[0], grads[2], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("name", list(DENSE))
def test_dense_forward_backward(name):
    if name != "D-50M" and not torch.cuda.is_available():
        pytest.skip("large presets only on the GPU")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    m = Transformer(ModelConfig(max_seq_len=64, **MODELS[name])).to(dev)
    x = torch.randint(0, 50257, (2, 65), device=dev)
    with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
        _, loss = m(x[:, :-1], x[:, 1:])
    loss.backward()
    assert torch.isfinite(loss) and abs(float(loss) - 10.83) < 0.5      # ~ ln(50304) at initialisation
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_budget_gate():
    order = ["D-50M-s0", "D-100M-s0", "D-200M-s0", "D-400M-s0"]
    fast = dict(zip(order, [2e6, 1e6, 6e5, 3e5]))
    adm, drop, proj = run_dense.admit(fast, elapsed_h=0.6, cap_h=5.0, order=order)
    assert adm == order and not drop
    work = (500e6 + 50 * run_dense.VAL_TOKENS / 3) * run_dense.FACTOR / 3600
    assert proj == pytest.approx(0.6 + sum(work / s for s in fast.values()) + run_dense.END_H)
    # 400M does not fit: dropped, the smaller ones run
    adm, drop, _ = run_dense.admit(fast, elapsed_h=0.6, cap_h=1.6, order=order)
    assert adm == order[:3] and [d[0] for d in drop] == ["D-400M-s0"]
    # once a size is over budget, the larger ones are dropped too
    slow200 = {**fast, "D-200M-s0": 1e5}
    adm, drop, _ = run_dense.admit(slow200, elapsed_h=0.6, cap_h=1.6, order=order)
    assert adm == order[:2] and [d[0] for d in drop] == order[2:]
    # a failed preflight only drops that size
    failed = {**fast, "D-100M-s0": None}
    adm, drop, _ = run_dense.admit(failed, elapsed_h=0.6, cap_h=5.0, order=order)
    assert adm == ["D-50M-s0", "D-200M-s0", "D-400M-s0"] and drop[0] == ("D-100M-s0", "preflight failed")


@pytest.mark.parametrize("platform", ["runpod", "vm"])
def test_queue_end_per_platform(tmp_path, monkeypatch, platform):
    """The whole queue with fake training: the copy of queue.log on the box ends with the end message. On a VM nothing
    tries to stop the machine, and the end message talks neither about a Pod nor about a push that didn't happen."""
    out = tmp_path / "runs" / "q"
    for name, value in (("FAKE", True), ("OUT", str(out)), ("PRE", str(out) + "_preflight"),
                        ("BOX", str(tmp_path / "box")), ("BASE", str(tmp_path)), ("RUNS", [("D-50M-s0", "D-50M")]),
                        ("EXTRA", {}), ("PLATFORM", platform), ("NO_PUSH", platform == "vm"), ("FALLBACK_MIN", 0)):
        monkeypatch.setattr(run_dense, name, value)
    monkeypatch.setenv("FAKE_S", "0.05")
    sleep = time.sleep
    monkeypatch.setattr(time, "sleep", lambda s: sleep(min(s, 0.05)))
    run_dense.main()
    log = (out / "queue.log").read_text()
    boxed = (tmp_path / "box" / "runs" / "q" / "queue.log").read_text()
    assert (out / "BACKUP_OK").exists() and "QUEUE DONE, backup verified" in log
    end = [ln for ln in log.splitlines() if "ntfy: Fertig" in ln]
    assert len(end) == 1 and boxed.splitlines()[-1] == end[0]
    if platform == "vm":
        assert boxed == log and "would stop" not in log
        assert "Pod" not in end[0] and "gepusht" not in end[0] and "Instanz löschen" in end[0]
    else:
        assert log.startswith(boxed) and "Claude stoppt und löscht jetzt den Pod" in end[0] and "gepusht" in end[0]
        assert "(fake) would stop the Pod: fallback" in log.split(end[0])[1]


def fake_queue(tmp_path, monkeypatch, runs, ckpt=True, crash=(), **extra):
    out = tmp_path / "runs" / "q"
    args = ["--data", "wikipedia", "--tokens", "500e6"] + (["--ckpt_every_tokens", "100e6", "--resume"] if ckpt else [])
    for name, value in (("FAKE", True), ("OUT", str(out)), ("PRE", str(out) + "_preflight"),
                        ("BOX", str(tmp_path / "box")), ("BASE", str(tmp_path)), ("RUNS", runs), ("EXTRA", {}),
                        ("PLATFORM", "vm"), ("NO_PUSH", True), ("CKPT_EVERY", 100e6 if ckpt else 0.0),
                        ("CKPT_ARG", "100e6" if ckpt else "0"), ("ARGS", args), *extra.items()):
        monkeypatch.setattr(run_dense, name, value)
    monkeypatch.setenv("FAKE_S", "0.05")
    monkeypatch.setenv("SMLM_FAKE_CRASH", ",".join(crash))
    sleep = time.sleep
    monkeypatch.setattr(time, "sleep", lambda s: sleep(min(s, 0.05)))
    run_dense.main()
    return out


def test_queue_restarts_a_crashed_run_from_its_checkpoint(tmp_path, monkeypatch):
    runs = [("B-16M-x-s0", "B-16M-sparse"), ("D-50M-x-s0", "D-50M")]
    out = fake_queue(tmp_path, monkeypatch, runs, crash=["B-16M-x-s0"])
    log = (out / "queue.log").read_text()
    info = json.loads((out / "B-16M-x-s0" / "run-info.json").read_text())
    assert info["status"] == "done" and info["resumes"]
    assert "restarting from its checkpoint" in log and "Neustart 1/3 ab checkpoint at 1M tokens" in log
    assert json.loads((out / "restarts.json").read_text()) == {"B-16M-x-s0": 1}
    assert "QUEUE DONE, backup verified" in log and "NOT finished" not in log
    # preflight: stopped halfway with a checkpoint, then resumed; nothing big left behind
    assert "checkpoint 0.0 GiB saved in 0.5 s, resumed ok" in log
    for n, _ in runs:
        assert not (tmp_path / "runs" / "q_preflight" / n / "ckpt.pt").exists()
        assert not (out / n / "ckpt.pt").exists() and not (tmp_path / "box" / "runs" / "q" / n / "ckpt.pt").exists()
    stdout = (out / "B-16M-x-s0" / "stdout.log").read_text()
    assert stdout.count("===== ") == 2                     # both starts kept in the log


def test_queue_gives_up_after_max_restarts(tmp_path, monkeypatch):
    monkeypatch.setattr(run_dense, "FAKE_TRAIN", run_dense.FAKE_TRAIN.replace(
        "if crash and not info.get(\"resumes\"):", "if crash and len(info.get(\"resumes\", [])) < 99:"))
    out = fake_queue(tmp_path, monkeypatch, [("D-50M-y-s0", "D-50M")], crash=["D-50M-y-s0"], MAX_RESTARTS=2)
    log = (out / "queue.log").read_text()
    assert log.count("restarting from its checkpoint") == 2
    assert "NOT finished: D-50M-y-s0 (running)" in log
    # an unfinished run keeps its checkpoint on the box, so it can be continued elsewhere
    assert (tmp_path / "box" / "runs" / "q" / "D-50M-y-s0" / "ckpt.pt").exists()


def test_queue_without_checkpoints_reports_instead_of_restarting(tmp_path, monkeypatch):
    out = fake_queue(tmp_path, monkeypatch, [("D-50M-z-s0", "D-50M")], ckpt=False, crash=["D-50M-z-s0"])
    log = (out / "queue.log").read_text()
    assert "restarting" not in log and "reported, not restarted (no checkpoints in this queue)" in log


def test_budget_gate_counts_only_what_is_left():
    order = ["B-16M-s0", "D-50M-s0"]
    speeds = {"B-16M-s0": 1e5, "D-50M-s0": 2e5}
    _, _, full = run_dense.admit(speeds, elapsed_h=0.0, cap_h=100.0, order=order)
    _, _, half = run_dense.admit(speeds, elapsed_h=0.0, cap_h=100.0, order=order,
                                 left={"B-16M-s0": 0.5, "D-50M-s0": 0.0})
    work = (500e6 + 50 * run_dense.VAL_TOKENS / 3) * run_dense.FACTOR / 3600
    assert full - half == pytest.approx(work / 1e5 * 0.5 + work / 2e5)
