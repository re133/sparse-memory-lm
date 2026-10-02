"""Abort-rule reference lookup (Hampter queue)."""
import math

from smlm.train import reference_ppl


def test_reference_ppl_exact_at_evals_and_log_interpolated(tmp_path):
    p = tmp_path / "metrics.csv"
    p.write_text("step,tokens,val_loss,val_ppl\n0,0,4.0,54.6\n305,9994240,3.0,20.1\n610,19988480,2.0,7.39\n")
    assert reference_ppl(p, 9994240) == math.exp(3.0)
    assert math.isclose(reference_ppl(p, (9994240 + 19988480) / 2), math.exp(2.5))


def test_reference_ppl_of_quicktest_run_at_abort_point():
    """The abort point of the queue (step 3050) is an evaluation of runs/s1b/B-1M-s0: no interpolation."""
    import os
    ref = os.path.join(os.path.dirname(os.path.dirname(__file__)), "runs", "s1b", "B-1M-s0", "metrics.csv")
    if not os.path.exists(ref):
        return
    assert math.isclose(reference_ppl(ref, 3050 * 32768), 39.2928, rel_tol=1e-4)
