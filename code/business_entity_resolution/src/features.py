"""Pair-wise feature computation (rapidfuzz), multiprocess over chunks.

Features (float32, all in [0, 1]):

0. name_token_set_core  token_set_ratio on suffix-stripped core names
1. name_token_sort      token_sort_ratio on full normalised names
2. name_ratio           plain ratio on full normalised names
3. addr_token_sort      token_sort_ratio on normalised addresses (0 if either
                        side has no address)
4. addr_ratio           plain ratio on addresses (0 if either missing)
5. addr_both_present    1.0 when both addresses exist, else 0.0
6. postal_match         1.0 when both postal codes exist and are equal

Memory notes: pair indices are int32 (never stacked into an int64 matrix) and
strings live in plain Python lists — both matter at 10M+ rows on a 16 GB box.
Workers are forked (macOS/Linux) so the arrays are shared copy-on-write rather
than pickled; each task only ships an index range.
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
from rapidfuzz import fuzz

FEATURE_NAMES = [
    "name_token_set_core",
    "name_token_sort",
    "name_ratio",
    "addr_token_sort",
    "addr_ratio",
    "addr_both_present",
    "postal_match",
]
N_FEATURES = len(FEATURE_NAMES)

# per-worker globals (set in the parent just before forking)
_G: dict = {}


def _set_globals(ci, cj, s1_cols, pool_cols) -> None:
    _G["ci"] = ci
    _G["cj"] = cj
    _G["s1"] = s1_cols
    _G["pool"] = pool_cols


def _init_worker() -> None:  # pragma: no cover - runs in child
    pass


def _core(norm: str, k: int) -> str:
    """Rebuild the suffix-stripped core name from norm + trailing-suffix count."""
    if k < 0:
        return ""
    if k == 0:
        return norm
    parts = norm.rsplit(" ", k)
    return parts[0] if len(parts) == k + 1 else ""


def _feat_row(n1, k1, a1, p1, n2, k2, a2, p2):
    # guard: token_set_ratio("", "") == 100 would be a false "perfect" match
    c1 = _core(n1, k1)
    c2 = _core(n2, k2)
    if c1 and c2:
        f0 = fuzz.token_set_ratio(c1, c2)
    else:
        f0 = fuzz.ratio(n1, n2) if (n1 and n2) else 0.0
    f1 = fuzz.token_sort_ratio(n1, n2) if (n1 and n2) else 0.0
    f2 = fuzz.ratio(n1, n2) if (n1 and n2) else 0.0
    if a1 and a2:
        f3 = fuzz.token_sort_ratio(a1, a2)
        f4 = fuzz.ratio(a1, a2)
        f5 = 100.0
    else:
        f3 = f4 = 0.0
        f5 = 0.0
    f6 = 100.0 if (p1 and p2 and p1 == p2) else 0.0
    return (f0, f1, f2, f3, f4, f5, f6)


def _compute_slice(args: Tuple[int, int]) -> np.ndarray:
    start, stop = args
    ci = _G["ci"][start:stop].tolist()
    cj = _G["cj"][start:stop].tolist()
    s1n, s1k, s1a, s1p = _G["s1"]
    pn, pk, pa, pp = _G["pool"]
    n = len(ci)
    out = np.empty((n, N_FEATURES), dtype=np.float32)
    for r in range(n):
        i = ci[r]
        j = cj[r]
        out[r] = _feat_row(
            s1n[i], s1k[i], s1a[i], s1p[i], pn[j], pk[j], pa[j], pp[j]
        )
    np.multiply(out, 1.0 / 100.0, out=out)
    return out


def compute_features(
    ci: np.ndarray,
    cj: np.ndarray,
    s1_cols: Tuple,
    pool_cols: Tuple,
    workers: int | None = None,
    chunk: int = 250_000,
):
    """Yield feature blocks (float32) for the aligned ``ci``/``cj`` pairs."""
    n = ci.shape[0]
    if n == 0:
        return

    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    tasks = [(s, min(s + chunk, n)) for s in range(0, n, chunk)]

    if workers <= 1:
        _set_globals(ci, cj, s1_cols, pool_cols)
        for t in tasks:
            yield _compute_slice(t)
        return

    import multiprocessing as mp

    ctx = mp.get_context("fork")  # share string arrays copy-on-write
    _set_globals(ci, cj, s1_cols, pool_cols)
    with ctx.Pool(processes=workers, initializer=_init_worker) as pool:
        for block in pool.imap(_compute_slice, tasks, chunksize=1):
            yield block
