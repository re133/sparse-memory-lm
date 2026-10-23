"""Step 17 (PCAT): does training move in few directions? (pre-check for training in a low-dimensional subspace)

  python -m smlm.train --model A --out_dir runs/pcat/A-s0 --tokens 500e6 --stop_after_tokens 100e6 \\
      --weights_every_tokens 1e6 --eval_every_tokens 10e6       (A's 500 M schedule, stopped early)
  python scripts/pcat.py                                         -> report/pcat/summary.json

Subspace: PCA of the weight vectors saved between 30 M and 70 M tokens (all parameters, flattened, centred). Test: how
much of the later movement lies in it: the share of the squared length of w(100 M) − w(70 M) inside the top-k
directions, and the same for each ~1 M step between 70 M and 100 M. Explained variance inside the window is reported
too, but it says little: a pure random walk also puts most of its variance into a few components (Antognini &
Sohl-Dickstein 2018); the prediction of later movement is the test.
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from smlm.atomic import write_json  # noqa: E402

RUN = os.path.join(ROOT, "runs", "pcat", "A-s0", "weights")
OUT = os.path.join(ROOT, "report", "pcat")
FIT, TEST = (30e6, 70e6), (70e6, 100e6)


def flat(path, keys=None):
    sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    if keys is None:                                     # each parameter once: the tied output weight is the embedding
        seen, keys = set(), []
        for k in sorted(sd):
            if sd[k].is_floating_point() and sd[k].data_ptr() not in seen:
                seen.add(sd[k].data_ptr())
                keys.append(k)
    return torch.cat([sd[k].float().reshape(-1) for k in keys]).numpy(), keys


def share(U, d):
    """Share of the squared length of d inside the span of the orthonormal columns of U."""
    p = U.T @ d.astype(np.float64)
    return float(p @ p / (d.astype(np.float64) @ d))


def main():
    files = sorted((int(f[:-3]), os.path.join(RUN, f)) for f in os.listdir(RUN) if f.endswith(".pt"))
    fit = [p for t, p in files if FIT[0] <= t <= FIT[1]]
    test = [(t, p) for t, p in files if TEST[0] <= t <= TEST[1]]
    w0, keys = flat(fit[0])
    X = np.empty((len(fit), w0.size), dtype=np.float32)
    X[0] = w0
    for i, p in enumerate(fit[1:], 1):
        X[i] = flat(p, keys)[0]
    start_fit, end_fit = X[0].copy(), X[-1].copy()
    X -= X.mean(0)
    lam, V = np.linalg.eigh((X.astype(np.float64) @ X.T.astype(np.float64)))
    lam, V = lam[::-1], V[:, ::-1]
    n = int((lam > lam[0] * 1e-9).sum())                 # at most len(fit) - 1 directions
    U = np.empty((w0.size, n), dtype=np.float64)
    for j in range(n):
        u = X.T.astype(np.float64) @ V[:, j]
        U[:, j] = u / np.linalg.norm(u)
    del X
    ws = [flat(p, keys)[0] for _, p in test]
    net = ws[-1] - ws[0]
    steps = [b - a for a, b in zip(ws, ws[1:])]
    ks = [k for k in (1, 5, 10, 20, n) if k <= n]
    drift = (end_fit - start_fit).astype(np.float64)
    drift /= np.linalg.norm(drift)
    out = {"params": int(w0.size), "fit_points": len(fit),
           "test_points": len(test), "directions": n,
           "explained_variance_in_window": {k: float(lam[:k].sum() / lam[:n].sum()) for k in ks},
           "net_70_100_share": {k: share(U[:, :k], net) for k in ks},
           "steps_70_100_share_mean": {k: float(np.mean([share(U[:, :k], d) for d in steps])) for k in ks},
           "random_subspace_share": {k: k / w0.size for k in ks},
           "net_70_100_share_on_past_drift_30_70": float((drift @ net.astype(np.float64)) ** 2
                                                          / (net.astype(np.float64) @ net))}
    out["fit_tokens"] = [min(t for t, _ in files if FIT[0] <= t), max(t for t, _ in files if t <= FIT[1])]
    out["test_tokens"] = [test[0][0], test[-1][0]]
    s = out["net_70_100_share"][n]
    out["verdict"] = ("low-dimensional enough to try" if s >= 0.5 else "not low-dimensional" if s < 0.1 else
                      "partly low-dimensional")
    os.makedirs(OUT, exist_ok=True)
    write_json(os.path.join(OUT, "summary.json"), out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
