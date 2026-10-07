"""Developer Cloud queue without a GPU, network or training.

  python -m pytest -q tests/test_devcloud_queue.py
"""
import ast
import builtins
import dataclasses
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


@pytest.fixture
def dc(monkeypatch):
    import run_devcloud as dc
    monkeypatch.setattr(dc, "PY", sys.executable)
    monkeypatch.delenv("NTFY_TOPIC", raising=False)
    sleep = time.sleep
    monkeypatch.setattr(dc.time, "sleep", lambda seconds: sleep(min(seconds, .01)))
    return dc


def dummy_step(dc, out, name="B-16M-s1", code=None, kind="experiment"):
    directory = out / name
    if code is None:
        code = ("import json, os; from pathlib import Path; "
                f"p = Path({str(directory)!r}); "
                "(p / 'run-info.json').write_text(json.dumps({'status': 'done'})); "
                "(p / 'model.pt').write_bytes(b'checkpoint'); "
                "(p / 'gpu.txt').write_text(os.environ['ROCR_VISIBLE_DEVICES'])")
    return dc.Step(name, [sys.executable, "-c", code], kind, 1)


def write_done(dc, step, out, source="source"):
    directory = out / step.name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "run-info.json").write_text('{"status": "done"}')
    (directory / "model.pt").write_bytes(b"checkpoint")
    dc.write_json(directory / "DONE", {"signature": dc.signature(step, source)})


def approval_file(dc, tmp_path, steps):
    criteria = tmp_path / "criteria.md"
    criteria.write_text("Approved evaluation criteria.\n")
    approval = tmp_path / "approval.json"
    record = {"approved": True, "steps": [s.name for s in steps if s.kind == "experiment"],
              "criteria_file": criteria.name, "criteria_sha256": dc.sha256(criteria)}
    dc.write_json(approval, record)
    return approval, criteria, record


def test_priority_order_is_independent_of_selection_order(dc, tmp_path):
    steps = dc.plan(tmp_path)
    assert [s.name for s in steps[:2]] == ["env", "tests"]
    assert [s.name for s in steps if s.kind == "experiment"] == [
        "B-16M-s1", "B-4M-s1", "D-100M-s1", "D-200M-s1", "BE-1M-s0"]
    selected = dc.select_steps(steps, "BE-1M-s0,speed,tests,B-16M-s1,env,tests")
    assert selected == [s for s in steps if s.name in {x.name for x in selected}]
    assert len(selected) == len({s.name for s in selected})
    assert dc.select_steps(steps, "all") == steps
    with pytest.raises(ValueError, match="unknown"):
        dc.select_steps(steps, "typo")


def test_resume_skips_completed_real_subprocess(dc, tmp_path, monkeypatch):
    step = dummy_step(dc, tmp_path)
    assert dc.execute([step], [3], tmp_path, "source") == {step.name: "done"}
    assert (tmp_path / step.name / "gpu.txt").read_text() == "3"
    assert dc.done(step, tmp_path, "source")

    def forbidden(*args, **kwargs):
        pytest.fail("a completed step started another process")

    monkeypatch.setattr(dc.subprocess, "Popen", forbidden)
    assert dc.execute([step], [3], tmp_path, "source") == {step.name: "done"}
    assert "skip B-16M-s1 (done)" in (tmp_path / "queue.log").read_text()


def test_failed_step_restarts_and_preserves_previous_attempt(dc, tmp_path):
    failed = dummy_step(dc, tmp_path, code="print('old attempt'); raise SystemExit(3)")
    assert dc.execute([failed], [0], tmp_path, "source") == {failed.name: "failed"}
    assert not (tmp_path / failed.name / "DONE").exists()
    resumed = dummy_step(dc, tmp_path)
    assert dc.execute([resumed], [0], tmp_path, "source") == {resumed.name: "done"}
    previous = list((tmp_path / "attempts").iterdir())
    assert len(previous) == 1
    assert "old attempt" in (previous[0] / "stdout.log").read_text()
    assert dc.read_json(previous[0] / "attempt.json")["returncode"] == 3


@pytest.mark.parametrize("reason", ["timeout", "stalled"])
def test_expired_job_stops_child_and_records_failure(dc, tmp_path, reason):
    step = dummy_step(dc, tmp_path, code="import time; time.sleep(30)")
    if reason == "timeout":
        step = dataclasses.replace(step, timeout_min=.0002)
    job = dc.Job(step, 0, tmp_path, "source", stall_min=.001)
    job.start()
    try:
        if reason == "stalled":
            old = time.time() - 60
            os.utime(tmp_path / step.name / "stdout.log", (old, old))
        result = None
        deadline = time.monotonic() + 2
        while result is None and time.monotonic() < deadline:
            result = job.poll()
            if result is None:
                time.sleep(.01)
        assert result == reason
        assert job.proc.poll() is not None and job.proc.returncode < 0
        record = dc.read_json(tmp_path / step.name / "attempt.json")
        assert record["status"] == reason and record["returncode"] == job.proc.returncode
        assert not (tmp_path / step.name / "DONE").exists()
    finally:
        job.stop()
        if job.fh:
            job.fh.close()


@pytest.mark.parametrize("damage", ["missing_model", "empty_model", "incomplete_info", "invalid_info",
                                    "changed_source", "changed_command"])
def test_finished_result_needs_artifacts_and_matching_signature(dc, tmp_path, damage):
    step = dummy_step(dc, tmp_path)
    write_done(dc, step, tmp_path)
    assert dc.done(step, tmp_path, "source")
    source = "source"
    directory = tmp_path / step.name
    if damage == "missing_model":
        (directory / "model.pt").unlink()
    elif damage == "empty_model":
        (directory / "model.pt").write_bytes(b"")
    elif damage == "incomplete_info":
        (directory / "run-info.json").write_text('{"status": "running"}')
    elif damage == "invalid_info":
        (directory / "run-info.json").write_text('{"status":')
    elif damage == "changed_source":
        source = "changed source or environment fingerprint"
    else:
        step = dataclasses.replace(step, command=[*step.command, "changed argument"])
    assert not dc.done(step, tmp_path, source)
    with pytest.raises(ValueError, match="stale/incomplete DONE"):
        dc.execute([step], [0], tmp_path, source)
    assert (directory / "DONE").exists()


@pytest.mark.parametrize("gpus", [[0], list(range(8))])
def test_one_experiment_per_gpu_in_priority_order(dc, tmp_path, monkeypatch, gpus):
    steps = [s for s in dc.plan(tmp_path) if s.kind == "experiment"]
    starts, occupied, max_active = [], set(), [0]

    class FakeJob:
        def __init__(self, step, gpu, out, source, stall_min):
            self.step, self.gpu, self.fh, self.polls = step, gpu, None, 0

        def start(self):
            assert self.gpu not in occupied
            occupied.add(self.gpu)
            starts.append((self.step.name, self.gpu))
            max_active[0] = max(max_active[0], len(occupied))

        def poll(self):
            self.polls += 1
            if self.polls == 1:
                return None
            occupied.remove(self.gpu)
            return "done"

        def stop(self):
            occupied.discard(self.gpu)

    monkeypatch.setattr(dc, "Job", FakeJob)
    monkeypatch.setattr(dc.time, "sleep", lambda _: None)
    assert dc.execute(steps, gpus, tmp_path, "source", parallel=True) == {s.name: "done" for s in steps}
    assert [name for name, _ in starts] == [s.name for s in steps]
    assert [gpu for _, gpu in starts] == [gpus[i % len(gpus)] for i in range(len(steps))]
    assert max_active[0] == min(len(gpus), len(steps))
    assert not occupied


def test_default_subset_and_dry_run_have_no_side_effects(dc, tmp_path, monkeypatch, capsys):
    out = tmp_path / "must-not-exist"

    def forbidden(*args, **kwargs):
        pytest.fail("dry run attempted a process, GPU/data probe, or filesystem write")

    def read_only(original):
        def open_file(file, mode="r", *args, **kwargs):
            if any(flag in mode for flag in "wax+"):
                forbidden()
            return original(file, mode, *args, **kwargs)
        return open_file

    monkeypatch.setattr(builtins, "open", read_only(builtins.open))
    monkeypatch.setattr(io, "open", read_only(io.open))
    for name in ("Popen", "run", "check_output", "check_call", "call"):
        monkeypatch.setattr(dc.subprocess, name, forbidden)
    for name in ("inventory", "verify_data", "write_json", "bundle", "approval_record"):
        monkeypatch.setattr(dc, name, forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    monkeypatch.setattr(Path, "write_bytes", forbidden)
    assert dc.main(["--dry_run", "--out_dir", str(out)]) == 0
    output = capsys.readouterr().out
    assert "RUN env:" in output and "RUN tests:" in output
    assert all(f"RUN {name}:" in output for name, _, _ in dc.SPEED)
    assert all(f"RUN {name}:" not in output for name, _, _ in dc.EXPERIMENTS)
    assert not out.exists()
    assert list(tmp_path.iterdir()) == []


def test_all_dry_run_prints_each_argument_and_initial_gpu(dc, tmp_path, capsys):
    import shlex
    assert dc.main(["--dry_run", "--only", "all", "--gpus", "0,1,2,3,4,5,6,7",
                    "--out_dir", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    for index, step in enumerate(s for s in dc.plan(tmp_path) if s.kind == "experiment"):
        assert f"RUN {step.name}: env ROCR_VISIBLE_DEVICES={index} " in output
        assert shlex.join(step.command) in output
    assert "--pack_only" in output


def test_python_override_is_preview_only(dc, tmp_path, capsys):
    executable = "/not-installed-yet/.venv/bin/python"
    assert dc.main(["--dry_run", "--only", "env,tests", "--python", executable,
                    "--out_dir", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert f"{executable} -m pytest -q -rs tests" in output
    with pytest.raises(ValueError, match="only for --dry_run"):
        dc.main(["--python", executable, "--out_dir", str(tmp_path)])
    assert list(tmp_path.iterdir()) == []


def test_dry_run_rejects_internal_record_env_without_side_effects(dc, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("internal environment recording bypassed dry run")

    monkeypatch.setattr(dc, "inventory", forbidden)
    monkeypatch.setattr(dc, "write_json", forbidden)
    monkeypatch.setattr(dc.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="cannot be combined with --dry_run"):
        dc.main(["--dry_run", "--record_env", str(tmp_path / "env.json")])
    assert list(tmp_path.iterdir()) == []


def test_pack_only_dry_run_prints_only_the_pack_command(dc, tmp_path, monkeypatch, capsys):
    import shlex

    def forbidden(*args, **kwargs):
        pytest.fail("package preview attempted real work")

    monkeypatch.setattr(dc, "inventory", forbidden)
    monkeypatch.setattr(dc, "bundle", forbidden)
    monkeypatch.setattr(dc.subprocess, "run", forbidden)
    assert dc.main(["--pack_only", "--dry_run", "--only", "all", "--out_dir", str(tmp_path)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    assert shlex.split(lines[0]) == [dc.PY, str(Path(dc.__file__).resolve()),
                                     "--out_dir", str(tmp_path), "--pack_only"]
    assert list(tmp_path.iterdir()) == []


def test_inventory_checks_qwen_imports_before_launching_suite(dc, monkeypatch):
    from types import SimpleNamespace
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout='{"devices": []}')

    monkeypatch.setattr(dc.subprocess, "run", run)
    assert dc.inventory() == {"devices": []}
    assert len(calls) == 1
    command, options = calls[0]
    assert command[:2] == [dc.PY, "-c"] and options["check"] is True
    tree = ast.parse(command[2])
    imported = {name.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                and node.module == "transformers" for name in node.names}
    assert {"Qwen3_5ForCausalLM", "Qwen3_5TextConfig"} <= imported


def literal_assignments(path):
    """Read old queue constants without importing their side effects or dependencies."""
    values = {}

    def value(node):
        if isinstance(node, ast.Name):
            return values[node.id]
        if isinstance(node, (ast.List, ast.Tuple)):
            result = []
            for child in node.elts:
                if isinstance(child, ast.Starred):
                    result.extend(value(child.value))
                else:
                    result.append(value(child))
            return tuple(result) if isinstance(node, ast.Tuple) else result
        return ast.literal_eval(node)

    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = value(node.value)
            except (ValueError, TypeError, KeyError):
                pass
    return values


def test_training_commands_match_original_arguments_except_seed_and_output(dc, tmp_path):
    old_cloud = literal_assignments(dc.ROOT / "scripts" / "run_cloud.py")
    old_dense = literal_assignments(dc.ROOT / "scripts" / "run_dense.py")
    old_amd = literal_assignments(dc.ROOT / "scripts" / "run_amd.py")
    steps = {s.name: s for s in dc.plan(tmp_path)}
    for name, model, _ in dc.EXPERIMENTS[:-1]:
        expected = (old_cloud if name.startswith("B-") else old_dense)["ARGS"].copy()
        expected[expected.index("--seed") + 1] = "1"
        assert steps[name].command == [dc.PY, "-m", "smlm.train", "--model", model,
                                       "--out_dir", str(tmp_path / name), *expected]
    for name, model, extra in old_amd["SPEED"]:
        assert steps[name].command == [dc.PY, "-m", "smlm.train", "--model", model,
                                       "--out_dir", str(tmp_path / name), *extra, *old_amd["SPEED_ARGS"]]
    assert steps["tests"].command == [dc.PY, "-m", "pytest", "-q", "-rs", "tests"]
    combined = steps["BE-1M-s0"].command
    assert combined[combined.index("--seed") + 1] == "0"
    assert combined[combined.index("--value_lr") + 1] == "2.4e-3"
    assert combined[combined.index("--eng_value_lr") + 1] == "3e-3"


def test_approval_requires_selected_experiments_and_unchanged_criteria(dc, tmp_path):
    steps = dc.select_steps(dc.plan(tmp_path), "experiments")
    with pytest.raises(ValueError, match="approval"):
        dc.approval_record(None, steps)
    path, criteria, record = approval_file(dc, tmp_path, steps)
    assert dc.approval_record(path, steps)["criteria"] == criteria.read_text()
    criteria.write_text("Changed criteria after approval.\n")
    with pytest.raises(ValueError, match="hash changed"):
        dc.approval_record(path, steps)
    criteria.write_text("Approved evaluation criteria.\n")
    record["steps"] = [steps[0].name]
    dc.write_json(path, record)
    with pytest.raises(ValueError, match="incomplete"):
        dc.approval_record(path, steps)
    record["steps"] = [s.name for s in steps]
    record["approved"] = False
    dc.write_json(path, record)
    with pytest.raises(ValueError, match="approval"):
        dc.approval_record(path, steps)
    assert dc.approval_record(None, dc.select_steps(dc.plan(tmp_path), "env,tests")) is None


@pytest.mark.parametrize("failed_prerequisite", ["env", "tests", "speed_B-16M"])
def test_failed_prerequisite_never_launches_experiments(dc, tmp_path, monkeypatch, failed_prerequisite):
    out = tmp_path / "results"
    steps = dc.plan(out)
    approval, _, _ = approval_file(dc, tmp_path, steps)
    calls = []

    def execute(selected, *args, **kwargs):
        calls.extend(s.name for s in selected)
        return {s.name: "failed" if s.name == failed_prerequisite else "done" for s in selected}

    info = {"hip": "7.2", "torch": "test", "triton": "test", "python": "test",
            "devices": [{"index": 0, "arch": "gfx942", "name": "test", "memory_bytes": 1}]}
    monkeypatch.setattr(dc, "inventory", lambda: info)
    monkeypatch.setattr(dc, "verify_data", lambda: [])
    monkeypatch.setattr(dc, "execute", execute)
    monkeypatch.setattr(dc, "source_hash", lambda: "source")
    assert dc.main(["--only", "all", "--approval", str(approval), "--out_dir", str(out)]) == 1
    assert failed_prerequisite in calls
    assert all(name not in calls for name, _, _ in dc.EXPERIMENTS)
    assert dc.read_json(out / "selection.json")["complete"] is False


def test_experiments_only_cannot_bypass_missing_prerequisites(dc, tmp_path, monkeypatch):
    out = tmp_path / "results"
    approval, _, _ = approval_file(dc, tmp_path, dc.plan(out))
    monkeypatch.setattr(dc, "inventory", lambda: {
        "hip": "7.2", "devices": [{"index": 0, "arch": "gfx942"}]})
    monkeypatch.setattr(dc, "verify_data", lambda: [])
    monkeypatch.setattr(dc, "source_hash", lambda: "source")

    def forbidden(*args, **kwargs):
        pytest.fail("experiments started without completed preflight")

    monkeypatch.setattr(dc, "execute", forbidden)
    assert dc.main(["--only", "experiments", "--approval", str(approval), "--out_dir", str(out)]) == 1
    assert "missing prerequisite env" in (out / "queue.log").read_text()


@pytest.mark.parametrize("changed", ["software", "gpu"])
def test_environment_drift_refuses_resume_before_any_work(dc, tmp_path, monkeypatch, changed):
    out = tmp_path / "results"
    out.mkdir()
    previous = {"torch": "original", "hip": "7.2", "devices": [{"index": 0, "arch": "gfx942"}]}
    dc.write_json(out / "inventory.json", previous)
    current = json.loads(json.dumps(previous))
    if changed == "software":
        current["torch"] = "replacement"
    else:
        current["devices"][0]["arch"] = "gfx942:sramecc+:xnack-"

    def forbidden(*args, **kwargs):
        pytest.fail("environment drift did not stop work")

    monkeypatch.setattr(dc, "inventory", lambda: current)
    monkeypatch.setattr(dc, "execute", forbidden)
    monkeypatch.setattr(dc, "verify_data", forbidden)
    monkeypatch.setattr(dc, "bundle", forbidden)
    with pytest.raises(ValueError, match="GPU/software environment changed"):
        dc.main(["--only", "env,tests", "--out_dir", str(out)])
    assert dc.read_json(out / "inventory.json") == previous
    assert not (out / "selection.json").exists()


def test_successful_main_records_complete_selection_and_bundle(dc, tmp_path, monkeypatch):
    out = tmp_path / "results"
    info = {"hip": "7.2", "devices": [{"index": 0, "arch": "gfx942"}]}
    env_code = ("from pathlib import Path; "
                f"Path({str(out / 'env' / 'env.json')!r}).write_text({json.dumps(info)!r})")
    steps = [dummy_step(dc, out, name="env", code=env_code, kind="env"),
             dummy_step(dc, out, name="tests", code="print('dummy test suite passed')", kind="tests")]
    monkeypatch.setattr(dc, "plan", lambda _: steps)
    monkeypatch.setattr(dc, "inventory", lambda: info)
    monkeypatch.setattr(dc, "source_hash", lambda: "source")

    def forbidden(*args, **kwargs):
        pytest.fail("environment/tests-only selection tried to read training data")

    monkeypatch.setattr(dc, "verify_data", forbidden)
    assert dc.main(["--only", "env,tests", "--out_dir", str(out)]) == 0
    result = dc.read_json(out / "selection.json")
    assert result["complete"] is True
    assert result["selected"] == ["env", "tests"]
    assert result["results"] == {"env": "done", "tests": "done"}
    assert all(dc.done(step, out, "source") for step in steps)
    with tarfile.open(out.with_suffix(".tar")) as archive:
        saved = json.load(archive.extractfile("results/selection.json"))
        assert saved == result
    assert "SELECTED STEPS COMPLETE" in (out / "queue.log").read_text()


def test_bundle_rebuilds_manifest_and_archive_after_results_change(dc, tmp_path):
    out = tmp_path / "results"
    out.mkdir()
    (out / "old.log").write_text("removed before second bundle\n")
    (out / "model.pt").write_bytes(b"first")
    dc.bundle(out)
    (out / "old.log").unlink()
    (out / "model.pt").write_bytes(b"updated")
    (out / "new.log").write_text("new result\n")
    dc.bundle(out)
    manifest = (out / "SHA256SUMS").read_text()
    expected = {"model.pt": b"updated", "new.log": b"new result\n"}
    assert {line.split("  ", 1)[1] for line in manifest.splitlines()} == set(expected)
    for line in manifest.splitlines():
        digest, name = line.split("  ", 1)
        assert digest == hashlib.sha256(expected[name]).hexdigest()
    archive = out.with_suffix(".tar")
    digest, name = archive.with_name(archive.name + ".sha256").read_text().strip().split("  ", 1)
    assert digest == dc.sha256(archive) and name == archive.name
    with tarfile.open(archive) as tar:
        files = {member.name: tar.extractfile(member).read() for member in tar.getmembers() if member.isfile()}
    assert files == {f"results/{name}": content for name, content in expected.items()} | {
        "results/SHA256SUMS": manifest.encode()}
    assert not archive.with_name(archive.name + ".tmp").exists()


def test_bundle_rejects_symlink_outside_results(dc, tmp_path):
    out = tmp_path / "results"
    out.mkdir()
    elsewhere = tmp_path / "unrelated.txt"
    elsewhere.write_text("do not include\n")
    (out / "link").symlink_to(elsewhere)
    with pytest.raises(ValueError, match="symlink"):
        dc.bundle(out)
    assert not out.with_suffix(".tar").exists()


def test_uploaded_data_is_verified_without_modification(dc, tmp_path):
    directory = tmp_path / "data" / "wikipedia_en_gpt2"
    directory.mkdir(parents=True)
    token_file = directory / "train.bin"
    token_file.write_bytes(b"prepared tokens")
    cloud = tmp_path / "cloud"
    cloud.mkdir()
    manifest = cloud / "data_sha256.txt"
    manifest.write_text(f"{dc.sha256(token_file)}  wikipedia_en_gpt2/train.bin\n")
    assert dc.verify_data(tmp_path) == ["wikipedia_en_gpt2/train.bin"]
    token_file.write_bytes(b"damaged tokens")
    with pytest.raises(ValueError, match="missing/corrupt"):
        dc.verify_data(tmp_path)
    assert token_file.read_bytes() == b"damaged tokens"


def test_allocator_flags_and_conflicting_gpu_masks_are_removed(dc, monkeypatch):
    for key in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF", "PYTORCH_ALLOC_CONF",
                "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL",
                "HSA_OVERRIDE_GFX_VERSION", "TRITON_INTERPRET"):
        monkeypatch.setenv(key, "unsafe inherited setting")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "7")
    env = dc.clean_env(2)
    assert env["ROCR_VISIBLE_DEVICES"] == "2"
    assert env["TRITON_CACHE_DIR"] == str(dc.ROOT / ".cache" / "triton")
    assert "HIP_VISIBLE_DEVICES" not in env and "CUDA_VISIBLE_DEVICES" not in env
    assert all(not (key.startswith("PYTORCH_") and key.endswith("ALLOC_CONF")) for key in env)
    assert "HSA_OVERRIDE_GFX_VERSION" not in env and "TRITON_INTERPRET" not in env
