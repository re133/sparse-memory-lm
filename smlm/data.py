"""Token streams for training / evaluation.

Training: the train split is cut into non-overlapping windows of seq_len+1 tokens (inputs + shifted
targets). Each epoch visits every window once in an order fixed by (data_seed, epoch), so all models
trained with the same data_seed see exactly the same token stream in the same order.
Evaluation: the full split as consecutive non-overlapping windows (last window shorter), so every
token except the first is predicted exactly once.
"""
import json
import os

import numpy as np
import torch

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "wikitext103_gpt2")


def load_meta():
    with open(os.path.join(DATA_DIR, "meta.json")) as f:
        return json.load(f)


def load_split(split):
    return np.memmap(os.path.join(DATA_DIR, f"{split}.bin"), dtype=np.uint16, mode="r")


class TrainStream:
    def __init__(self, seq_len, batch_seqs, data_seed, split="train"):
        self.tokens = load_split(split)
        self.seq_len, self.batch_seqs, self.data_seed = seq_len, batch_seqs, data_seed
        self.n_windows = (len(self.tokens) - 1) // seq_len
        self.steps_per_epoch = self.n_windows // batch_seqs
        self.tokens_per_step = batch_seqs * seq_len
        self._epoch, self._perm = None, None

    def _order(self, epoch):
        if epoch != self._epoch:
            self._perm = np.random.default_rng([self.data_seed, epoch]).permutation(self.n_windows)
            self._epoch = epoch
        return self._perm

    def batch(self, step):
        """Batch for global step `step` -> uint16 array (batch_seqs, seq_len + 1)."""
        epoch, s = divmod(step, self.steps_per_epoch)
        win = np.sort(self._order(epoch)[s * self.batch_seqs:(s + 1) * self.batch_seqs])
        T = self.seq_len
        return np.stack([self.tokens[w * T:w * T + T + 1] for w in win])


def eval_windows(split, seq_len):
    """Yield (inputs, targets) int64 tensors (1, <=seq_len) covering the whole split."""
    tok = torch.from_numpy(np.asarray(load_split(split), dtype=np.int64))
    n = len(tok) - 1
    for start in range(0, n, seq_len):
        end = min(start + seq_len, n)
        yield tok[start:end].unsqueeze(0), tok[start + 1:end + 1].unsqueeze(0)
