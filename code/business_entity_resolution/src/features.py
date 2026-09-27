"""Pair-wise feature computation (rapidfuzz), multiprocess over chunks.

Features (float32, all in [0, 1]) — see ``FEATURE_NAMES`` for the authoritative
order. Columns 0-6 are the original legacy block (kept byte-compatible so older
artifacts and the heuristic fallback keep working); 7-17 were added after the
first leaderboard feedback to give the model more ways to separate
abbreviation / word-order / transliteration variants from genuine non-matches:

0. name_token_set_core  token_set_ratio on suffix-stripped core names
1. name_token_sort      token_sort_ratio on full normalised names
2. name_ratio           plain ratio on full normalised names
3. addr_token_sort      token_sort_ratio on normalised addresses (0 if either
                        side has no address)
4. addr_ratio           plain ratio on addresses (0 if either missing)
5. addr_both_present    1.0 when both addresses exist, else 0.0
6. postal_match         1.0 when both postal codes exist and are equal
7. name_wratio_core     WRatio on core names (best-of partial/token/ratio)
8. name_tset_full       token_set_ratio on full names
9. name_exact           1.0 when normalised names are identical
10. core_exact          1.0 when suffix-stripped core names are identical
11. skel_exact          1.0 when consonant skeletons agree (transliteration)
12. addr_wratio         WRatio on addresses
13. addr_tset           token_set_ratio on addresses (component reordering)
14. addr_exact          1.0 when normalised addresses are identical
15. postal_prefix3      1.0 when postal region prefixes agree
16. name_len_ratio      min/max token count of the two names
17. addr_len_ratio      min/max token count of the two addresses

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

from .normalize import name_skeleton

FEATURE_NAMES = [
    # --- legacy block (indices 0-6 keep their original meaning) ---
    "name_token_set_core",   # 0  token_set_ratio on suffix-stripped core names
    "name_token_sort",       # 1  token_sort_ratio on full normalised names
    "name_ratio",            # 2  plain ratio on full normalised names
    "addr_token_sort",       # 3  token_sort_ratio on addresses (0 if missing)
    "addr_ratio",            # 4  plain ratio on addresses (0 if missing)
    "addr_both_present",     # 5  1.0 when both addresses exist
    "postal_match",          # 6  1.0 when both postal codes exist and are equal
    # --- added: name robustness ---
    "name_wratio_core",      # 7  WRatio on core names (best-of combination)
    "name_tset_full",        # 8  token_set_ratio on full names
    "name_exact",            # 9  1.0 when normalised names are identical
    "core_exact",            # 10 1.0 when core names are identical
    "skel_exact",            # 11 1.0 when consonant skeletons agree (translit)
    # --- added: address robustness ---
    "addr_wratio",           # 12 WRatio on normalised addresses
    "addr_tset",             # 13 token_set_ratio (component reordering)
    "addr_exact",            # 14 1.0 when normalised addresses are identical
    # --- added: geo / shape ---
    "postal_prefix3",        # 15 1.0 when postal prefixes (region) agree
    "name_len_ratio",        # 16 min/max token count of the two names
    "addr_len_ratio",        # 17 min/max token count of the two addresses
]
N_FEATURES = len(FEATURE_NAMES)

# explicit indices (the model consumes the matrix by column order)
IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}

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


def _len_ratio(s1: str, s2: str) -> float:
    """min/max token-count ratio of two strings (0.0 when either is empty)."""
    if not s1 or not s2:
        return 0.0
    n1 = s1.count(" ") + 1
    n2 = s2.count(" ") + 1
    return min(n1, n2) / max(n1, n2)


def _skeleton(text: str) -> str:
    """Consonant skeleton of a whole normalised string (transliteration key)."""
    return "".join(name_skeleton(t) for t in text.split())


def _feat_row(n1, k1, a1, p1, n2, k2, a2, p2):
    # guard: token_set_ratio("", "") == 100 would be a false "perfect" match
    c1 = _core(n1, k1)
    c2 = _core(n2, k2)
    have_n = bool(n1 and n2)
    have_c = bool(c1 and c2)
    have_a = bool(a1 and a2)

    # ---- name block -------------------------------------------------------
    if have_c:
        f0 = fuzz.token_set_ratio(c1, c2)
        f7 = fuzz.WRatio(c1, c2)
    elif have_n:
        f0 = fuzz.ratio(n1, n2)
        f7 = fuzz.WRatio(n1, n2)
    else:
        f0 = f7 = 0.0
    if have_n:
        f1 = fuzz.token_sort_ratio(n1, n2)
        f2 = fuzz.ratio(n1, n2)
        f8 = fuzz.token_set_ratio(n1, n2)
    else:
        f1 = f2 = f8 = 0.0
    f9 = 100.0 if (have_n and n1 == n2) else 0.0
    f10 = 100.0 if (have_c and c1 == c2) else 0.0
    f11 = 100.0 if (have_c and _skeleton(c1) == _skeleton(c2)) else 0.0
    f16 = 100.0 * _len_ratio(n1, n2)

    # ---- address block ----------------------------------------------------
    if have_a:
        f3 = fuzz.token_sort_ratio(a1, a2)
        f4 = fuzz.ratio(a1, a2)
        f12 = fuzz.WRatio(a1, a2)
        f13 = fuzz.token_set_ratio(a1, a2)
        f14 = 100.0 if a1 == a2 else 0.0
        f17 = 100.0 * _len_ratio(a1, a2)
        f5 = 100.0
    else:
        f3 = f4 = f12 = f13 = f14 = f17 = 0.0
        f5 = 0.0

    # ---- geo --------------------------------------------------------------
    f6 = 100.0 if (p1 and p2 and p1 == p2) else 0.0
    if p1 and p2 and len(p1) >= 3 and len(p2) >= 3:
        f15 = 100.0 if p1[:3] == p2[:3] else 0.0
    else:
        f15 = 0.0

    return (
        f0, f1, f2, f3, f4, f5, f6,
        f7, f8, f9, f10, f11,
        f12, f13, f14,
        f15, f16, f17,
    )


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


# --------------------------------------------------------------------------
# similarity prefilter (candidate-generation stage)
# --------------------------------------------------------------------------
def _prefilter_slice(t):
    """Keep-mask for one chunk: keep if name WRatio >= min_name OR address
    WRatio >= min_addr (a missing name/address scores 0, never a false pass)."""
    s, e = t
    ci, cj = _G["ci"], _G["cj"]
    n1s, a1s = _G["s1"][0], _G["s1"][2]
    n2s, a2s = _G["pool"][0], _G["pool"][2]
    min_n, min_a = _G["min_name"], _G["min_addr"]
    out = np.empty(e - s, dtype=bool)
    for r in range(s, e):
        p, q = int(ci[r]), int(cj[r])
        if min_n <= 0:
            out[r - s] = True
            continue
        na, nb = n1s[p], n2s[q]
        keep = bool(na and nb and fuzz.WRatio(na, nb) >= min_n)
        if not keep and min_a > 0:
            aa, ab = a1s[p], a2s[q]
            keep = bool(aa and ab and fuzz.WRatio(aa, ab) >= min_a)
        out[r - s] = keep
    return out


def prefilter_pairs(
    ci: np.ndarray,
    cj: np.ndarray,
    s1_cols: Tuple,
    pool_cols: Tuple,
    *,
    min_name: float,
    min_addr: float,
    workers: int | None = None,
    chunk: int = 250_000,
) -> np.ndarray:
    """Boolean keep-mask over ``ci``/``cj`` for the similarity prefilter.

    Runs before feature extraction, so ``candidate_pairs.tsv`` stays exactly
    what the model runs inference over.  Thresholds <= 0 disable that side of
    the rule; both disabled returns all-True (prefilter off).
    """
    n = ci.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)
    if min_name <= 0 and min_addr <= 0:
        return np.ones(n, dtype=bool)

    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    tasks = [(s, min(s + chunk, n)) for s in range(0, n, chunk)]
    _set_globals(ci, cj, s1_cols, pool_cols)
    _G["min_name"] = min_name
    _G["min_addr"] = min_addr

    if workers <= 1:
        parts = [_prefilter_slice(t) for t in tasks]
    else:
        import multiprocessing as mp

        ctx = mp.get_context("fork")  # share string arrays copy-on-write
        with ctx.Pool(processes=workers, initializer=_init_worker) as pool:
            parts = list(pool.imap(_prefilter_slice, tasks, chunksize=1))
    return np.concatenate(parts)
