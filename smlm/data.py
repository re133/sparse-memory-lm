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

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DATASETS = {
    "wikitext103": "wikitext103_gpt2",      # stage 1 (scripts/prepare_data.py)
    "wikipedia": "wikipedia_en_gpt2",       # stage 1b, fresh tokens (scripts/prepare_wikipedia.py)
    # Hampter: 1.5 B training tokens (wider article band), same validation set, first 505 M = "wikipedia"
    "wikipedia_1500m": "wikipedia_en_gpt2_1500m",
}
# default dataset directory (stage 1); SMLM_DATA_DIR overrides it
DATA_DIR = os.environ.get("SMLM_DATA_DIR") or os.path.join(DATA_ROOT, DATASETS["wikitext103"])


def data_dir(dataset=None):
    return DATA_DIR if dataset is None else os.path.join(DATA_ROOT, DATASETS[dataset])


def load_meta(dataset=None):
    with open(os.path.join(data_dir(dataset), "meta.json")) as f:
        return json.load(f)


def has_split(split, dataset=None):
    return os.path.exists(os.path.join(data_dir(dataset), f"{split}.bin"))


def load_split(split, dataset=None):
    return np.memmap(os.path.join(data_dir(dataset), f"{split}.bin"), dtype=np.uint16, mode="r")


class TrainStream:
    def __init__(self, seq_len, batch_seqs, data_seed, split="train", dataset=None):
        self.tokens = load_split(split, dataset)
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
