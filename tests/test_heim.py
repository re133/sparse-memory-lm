"""scripts/heim.py: pause on a reason, continue after a quiet time, end with the run's exit code (fake run, no GPU).

  python -m pytest -q tests/test_heim.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import heim  # noqa: E402

# counts to a target, one step every 10 ms, remembers its count in a file; SIGUSR1 -> exit 75 after the step
CHILD = """
import os, signal, sys, time
path, target = sys.argv[1], int(sys.argv[2])
n = int(open(path).read()) if os.path.exists(path) else 0
asked = []
signal.signal(signal.SIGUSR1, lambda *_: asked.append(1))
while n < target:
    time.sleep(0.01)
    n += 1
    open(path, "w").write(str(n))
    if asked:
        sys.exit(75)
"""


def test_pause_and_continue(tmp_path):
    script, count = tmp_path / "child.py", tmp_path / "count"
    script.write_text(CHILD)
    calls = []

    def why(own):
        calls.append(len(own))
        n = int(count.read_text()) if count.exists() else 0
        return "a game" if 20 <= n and len(calls) < 40 else None     # a "game" from step 20 for a while

    pushed = []
    code = heim.supervise([sys.executable, str(script), str(count), "100"], str(tmp_path), why, resume_s=0.05,
                          interval=0.01, notify=pushed.append)
    assert code == 0 and count.read_text() == "100"
    log = (tmp_path / "heim.log").read_text()
    assert log.count("pause: a game") == 1 and "continue" in log and "exit code 0" in log
    assert any(m.startswith("pause") for m in pushed) and any(m.startswith("continue") for m in pushed)
    assert max(calls) >= 1                               # the run's own processes are passed to why()


def test_other_exit_code_ends(tmp_path):
    script = tmp_path / "fail.py"
    script.write_text("import sys; sys.exit(3)")
    code = heim.supervise([sys.executable, str(script)], str(tmp_path), lambda own: None, resume_s=0,
                          interval=0.01)
    assert code == 3
