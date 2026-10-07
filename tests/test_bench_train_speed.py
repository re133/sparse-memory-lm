"""Benchmark plans preserve effective batches and compare curves at equal token counts, without a GPU."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("bench_train_speed", Path(__file__).resolve().parents[1]
                                            / "scripts" / "bench_train_speed.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def test_plan_covers_all_flags_and_micro_batches():
    cases = bench.make_cases(["A", "B-1M-sparse"], None, 32, {"A": 8, "B-1M-sparse": 4})
    assert len(cases) == 16
    assert len({case["id"] for case in cases}) == 16
    for name, sizes in (("A", {8, 16}), ("B-1M-sparse", {4, 8})):
        selected = [case for case in cases if case["preset"] == name]
        assert {case["micro_bs"] for case in selected} == sizes
        for micro in sizes:
            assert {(case["fused_ce"], case["compile"]) for case in selected if case["micro_bs"] == micro} == {
                (False, False), (False, True), (True, False), (True, True)}
            assert 32 // micro * micro == 32


def test_explicit_micro_batches_deduplicate_but_keep_default():
    cases = bench.make_cases(["A", "A"], [16, 16, 32], 32, {"A": 8})
    assert len(cases) == 12
    assert {case["micro_bs"] for case in cases} == {8, 16, 32}


@pytest.mark.parametrize("micro", [0, -1, 3, 64])
def test_invalid_micro_batch_rejected(micro):
    with pytest.raises(ValueError, match="must be positive and divide"):
        bench.make_cases(["A"], [micro], 32, {"A": 8})


def test_curve_schedule_reports_only_complete_steps():
    steps, checkpoints = bench.curve_schedule(103, 31, 10)
    assert steps == 10
    assert checkpoints == [0, 3, 6, 9, 10]
    assert steps * 10 <= 103


@pytest.mark.parametrize("tokens,eval_tokens", [(0, 10), (9, 10), (100, 0), (float("inf"), 10), (100, float("nan"))])
def test_invalid_curve_schedule_rejected(tokens, eval_tokens):
    with pytest.raises(ValueError):
        bench.curve_schedule(tokens, eval_tokens, 10)


def test_comparisons_use_matching_micro_baseline_and_actual_tokens():
    cases = bench.make_cases(["B"], [8], 32, {"B": 4})
    rows = []
    for case in cases:
        loss = 3.0 + (0.4 if case["micro_override"] else 0) + (0.1 if case["fused_ce"] else 0)
        rows.append({"case": case, "status": "done", "model_config": {"mem_layers": [0],
                     "mem_query_norm": "batchnorm"}, "curves": [{"tokens": 100, "val_loss": loss}]})
    out = bench.comparisons(rows)
    fused_override = next(row for row in out if row["case"] == "B-ce1-compile0-mb8")
    assert fused_override["delta_default_micro"] == pytest.approx(0.5)
    assert fused_override["delta_same_micro"] == pytest.approx(0.1)
    assert fused_override["micro_batch_changes_batchnorm"]
    rows[0]["curves"][0]["tokens"] = 99
    out = bench.comparisons(rows)
    assert next(row for row in out if row["case"] == "B-ce1-compile0-mb8")["baseline_val_loss"] is None


def test_failed_baseline_does_not_claim_comparison():
    case = bench.make_cases(["A"], [8], 32, {"A": 8})[0]
    out = bench.comparisons([{"case": case, "status": "failed", "curves": [{"tokens": 10, "val_loss": 3.0}]}])
    assert out[0]["baseline_val_loss"] is None
    assert out[0]["delta_same_micro"] is None


def test_output_cannot_escape_worktree():
    assert bench.output_path(bench.ROOT / ".scratch" / "speed").is_relative_to(bench.ROOT)
    for path in (bench.ROOT, bench.ROOT.parent / "AngryAnt" / "runs", bench.ROOT / ".." / "escaped"):
        with pytest.raises(ValueError, match="subdirectory"):
            bench.output_path(path)


def test_queue_uses_isolated_workers_and_keeps_failures(tmp_path, monkeypatch):
    import torch
    monkeypatch.setattr(bench, "ROOT", tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    data = tmp_path / "data"
    data.mkdir()
    for name in ("train.bin", "validation.bin", "meta.json"):
        (data / name).write_text("")
    configs = []

    def run(command, **kwargs):
        config = json.loads(Path(command[-1]).read_text())
        configs.append(config)
        assert kwargs["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
        status = "oom" if len(configs) == 2 else "done"
        bench.write_json(config["result_path"], {"case": config["case"], "mode": config["mode"], "status": status})
        return SimpleNamespace(returncode=int(status != "done"))

    monkeypatch.setattr(bench.subprocess, "run", run)
    out = tmp_path / "out"
    status = bench.main(["--out_dir", str(out), "--data_dir", str(data), "--presets", "A", "--micro_bs", "8",
                         "--mode", "speed", "--steps", "1", "--warmup_steps", "1"])
    assert status == 1
    assert len(configs) == 4
    assert len({config["result_path"] for config in configs}) == 4
    assert {config["seed"] for config in configs} == {0}
    assert {config["data_seed"] for config in configs} == {1234}
    assert {config["batch_seqs"] for config in configs} == {32}
    assert [result["status"] for result in json.loads((out / "results.json").read_text())] == [
        "done", "oom", "done", "done"]
