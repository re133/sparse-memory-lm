"""Step 13 evaluation (scripts/dense_equiv.py --step13) on made-up runs: gates per model and the verdict rule fixed in
REPORT.md before the runs."""
import json
import os
import shutil
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import dense_equiv as de  # noqa: E402

N = {"A": 21.2e6, "D-50M": 49.6e6, "D-100M": 99.1e6, "D-200M": 202.4e6, "D-400M": 396.5e6,
     "B-1M": 1.9e6, "B-4M": 2.0e6, "B-16M": 2.1e6}
OLD = {"A": 25.665, "D-50M": 22.452, "D-100M": 20.270, "D-200M": 18.758, "D-400M": 17.598,
       "B-1M": 21.837, "B-4M": 20.799, "B-16M": 19.960}


def write(root, rel, ppl, n, status="done", skipped=0):
    d = root / rel
    d.mkdir(parents=True, exist_ok=True)
    (d / "run-info.json").write_text(json.dumps({
        "status": status, "n_skipped_steps": skipped, "train_time_s": 3600, "train_config": {"total_tokens": 1},
        "results": {"val_ppl": ppl}, "macs_per_token": {"total": 1},
        "params": {"non_embedding": n, "memory_values": 0, "active_non_embedding_per_token": n}}))


@pytest.fixture
def root(tmp_path, monkeypatch):
    (tmp_path / "report").mkdir()
    shutil.copy(os.path.join(ROOT, "report", "dense_equiv.json"), tmp_path / "report")
    for name, rel in de.DENSE + de.BS:
        write(tmp_path, rel, OLD[name], N[name])
    monkeypatch.setattr(de, "ROOT", str(tmp_path))
    return tmp_path


def long(root, dense_scale, b_ppl, **kw):
    for name, rel in de.LONG_DENSE:
        if name not in kw.get("missing", ()):
            write(root, rel, OLD[name] * dense_scale.get(name, 0.9), N[name], **kw.get(name, {}))
    for name, rel in de.LONG_BS:
        write(root, rel, b_ppl[name], N[name])
    de.main(step13=True)
    return json.load(open(root / "report" / "long" / "dense_equiv.json"))


def test_step1_reference_ranges_are_the_published_ones():
    prev = json.load(open(os.path.join(ROOT, "report", "dense_equiv.json")))["b"]
    assert [round(prev[b]["n_eq"] / 1e6) for b in ("B-1M", "B-4M", "B-16M")] == [60, 83, 114]


def test_same_relative_gain_holds(root):
    # every model 10% better: N_eq stays where it was
    out = long(root, {}, {b: OLD[b] * 0.9 for b in ("B-1M", "B-4M", "B-16M")})
    assert all(out["b"][b]["verdict"] == "holds" for b in out["b"])
    assert abs(out["b"]["B-16M"]["n_eq"] / 1e6 - 114) < 1


def test_grows_and_shrinks(root):
    out = long(root, {}, {"B-1M": OLD["B-1M"] * 0.9, "B-4M": 18.758 * 0.9 * 0.99, "B-16M": OLD["B-16M"] * 0.95})
    assert out["b"]["B-4M"]["verdict"] == "grows"           # now about as good as D-200M
    assert out["b"]["B-16M"]["verdict"] == "shrinks"        # gained less than the dense models


def test_better_than_largest_counts_as_lower_end(root):
    out = long(root, {}, {"B-1M": OLD["B-1M"] * 0.9, "B-4M": OLD["B-4M"] * 0.9, "B-16M": 15.0},
               missing=("D-400M",))
    assert out["b"]["B-16M"]["status"].startswith(">") and out["b"]["B-16M"]["verdict"] == "grows"


def test_dense_model_failing_a_gate_leaves_the_ladder(root):
    out = long(root, {"D-200M": 1.01}, {b: OLD[b] * 0.9 for b in ("B-1M", "B-4M", "B-16M")},
               **{"D-50M": {"skipped": 21}})
    assert out["gates"]["D-200M"].startswith("gate 2") and out["gates"]["D-50M"].startswith("gate 1")
    assert "D-200M" not in out["dense"] and "D-50M" not in out["dense"]
    assert out["b"]["B-16M"]["verdict"] in ("holds", "grows", "shrinks")


def test_no_verdict_if_dense_points_not_monotone(root):
    out = long(root, {"D-100M": 0.8, "D-200M": 0.89}, {b: OLD[b] * 0.9 for b in ("B-1M", "B-4M", "B-16M")})
    assert out["dense_monotone"] is False and all(b["verdict"] is None for b in out["b"].values())
