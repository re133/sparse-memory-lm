"""The Qwen cloud queue (scripts/run_qwen.py) without a GPU, a Pod or the network: which results count as done,
what a failed step does, and that the end message says "unvollständig" instead of "fertig" when a step failed."""
import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))


@pytest.fixture
def rq(tmp_path, monkeypatch):
    import run_qwen as rq
    rd = rq.rd
    for mod, name, value in ((rd, "FAKE", True), (rd, "OUT", str(tmp_path)), (rq, "OUT", str(tmp_path)),
                             (rd, "BOX", str(tmp_path / "box")), (rd, "BASE", str(tmp_path)),
                             (rq, "FALLBACK_MIN", 0)):
        monkeypatch.setattr(mod, name, value)
    sleep = time.sleep
    monkeypatch.setattr(time, "sleep", lambda s: sleep(min(s, 0.05)))
    return rq


def test_is_done_only_for_complete_results(rq, tmp_path):
    def f(name, text):
        p = tmp_path / name
        p.write_text(text)
        return str(p)
    assert not rq.is_done(str(tmp_path / "missing.json"))
    assert not rq.is_done(f("empty.json", ""))
    assert not rq.is_done(f("cut.json", '{"summary": {"train'))
    assert not rq.is_done(f("nothing.json", "{}"))
    assert rq.is_done(f("facts.json", '{"summary": {}, "items": [1]}'))
    os.makedirs(tmp_path / "r")
    assert not rq.is_done(f("r/run-info.json", '{"status": "running"}'))
    assert rq.is_done(f("r/run-info.json", '{"status": "done"}'))
    assert not rq.is_done(f("addons.pt", ""))
    assert rq.is_done(f("addons.pt", "x"))


def test_step_failed_and_done(rq, tmp_path):
    out = tmp_path / "QT" / "facts.json"
    os.makedirs(out.parent)
    assert rq.step("QT_facts", [sys.executable, "-c", "raise SystemExit(1)"], str(out)) == "failed"
    write = f"import json; json.dump({{'items': [1]}}, open({str(out)!r}, 'w'))"
    assert rq.step("QT_facts", [sys.executable, "-c", write], str(out)) == "done"
    assert "QT_facts" in rq.step_minutes()
    log = (tmp_path / "queue.log").read_text()
    assert "FEHLGESCHLAGEN" in log and "QT/facts.json" in log.split("would push: Cloud qwen: QT_facts done")[1]
    assert rq.step("QT_facts", [sys.executable, "-c", "raise SystemExit(1)"], str(out)) == "done"   # skipped


@pytest.mark.parametrize("fail", [None, "QT_general"])
def test_end_message_lists_failed_steps(rq, tmp_path, monkeypatch, fail):
    data = tmp_path / "data"
    os.makedirs(data)
    (data / "meta.json").write_text(json.dumps({"splits": {"train_new": {"n_tokens": 1000}}}))
    monkeypatch.setattr(rq, "DATA", str(data))
    monkeypatch.setattr(rq, "tok_s", lambda d: 1e5)
    monkeypatch.setattr(rq, "step", lambda name, cmd, done_file, thermal=False: "failed" if name == fail else "done")
    rq.main()
    log = (tmp_path / "queue.log").read_text()
    assert (tmp_path / "BACKUP_OK").exists()
    if fail:
        assert not (tmp_path / "COMPLETE").exists()
        assert "UNVOLLSTÄNDIG" in log and "QT_general (failed)" in log and "Qwen fertig" not in log
    else:
        assert (tmp_path / "COMPLETE").exists() and "Qwen fertig" in log and "UNVOLLSTÄNDIG" not in log
