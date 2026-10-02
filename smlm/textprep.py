"""Helpers for building the stage-1b Wikipedia token stream (used by scripts/prepare_wikipedia.py)."""
import hashlib
import re

_WT_ESCAPES = {" @-@ ": "-", " @,@ ": ",", " @.@ ": "."}
_HEADING = re.compile(r"^ = ([^=].*?) = $")


def normalize_title(title):
    """Comparable form of a title: WikiText escapes undone, lower case, only letters and digits."""
    for k, v in _WT_ESCAPES.items():
        title = title.replace(k, v)
    return "".join(ch for ch in title.lower() if ch.isalnum())


def wikitext_article_titles(lines):
    """Article titles in WikiText raw lines: top-level headings ' = Title = ' (sections use '= =')."""
    out = []
    for line in lines:
        m = _HEADING.match(line.rstrip("\n"))
        if m:
            out.append(m.group(1))
    return out


def unit_hash(key, seed):
    """Deterministic pseudo-random number in [0, 1) for an article id (independent of file order)."""
    d = hashlib.blake2b(f"{seed}:{key}".encode(), digest_size=8).digest()
    return int.from_bytes(d, "big") / 2.0 ** 64


def assign_split(u, val_frac, train_frac):
    """'validation' for u < val_frac, 'train' for a disjoint band of width train_frac, else None.
    The training band starts at 0.001, so val_frac must stay below that (no overlap, no shared articles)."""
    assert val_frac < 0.001
    if u < val_frac:
        return "validation"
    if 0.001 <= u < 0.001 + train_frac:
        return "train"
    return None
