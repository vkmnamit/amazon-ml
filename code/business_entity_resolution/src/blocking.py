"""Candidate generation (blocking).

Strategy — several complementary keys per record, union of matches, then
popularity pruning and a per-S1 cap:

* ``p`` postal code              (strong, catches address-format noise)
* ``n`` name first-token prefix  (catches typos/suffix noise in lead token)
* ``n2`` name second-token prefix (catches DBA / word-order variants)
* ``s`` sorted name tokens       (catches word-order transpositions exactly)
* ``v`` consonant skeleton       (catches vowel/transliteration variants)
* ``a`` house number + street    (strong address anchor)

Keys are hashed to uint64 in-process (Python's str hash is stable within a
run, which is all we need — keys are never persisted). Blocking executes
country-by-country (country labels are consistent across the three sources)
and, within a country, in chunks of Source-1 records:

1. prune keys by pool-side document frequency and by n_S1(key)*n_pool(key),
2. sort the surviving pool keys once,
3. per S1 chunk: ``searchsorted`` → vectorised range explosion → dedupe pairs
   with counts (number of shared keys) → keep top ``top_k`` per entity.

Chunking keeps peak memory to a few hundred MB regardless of file size (the
entire merge output is never materialised at once), and each S1 entity's keys
lie inside a single chunk, so the per-entity top-k ranking is exact.
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import BlockingConfig
from .normalize import core_name, name_skeleton

S1_CHUNK = 100_000  # S1 records per blocking chunk


def record_keys(
    country: str, name_norm: str, addr_norm: str, postal: str, weak: bool = True
) -> List[str]:
    """Blocking keys for one record (country-scoped).

    ``weak=False`` emits only the four original per-record families
    (``p`` postal, ``n``/``n2`` leading-token prefixes, ``v`` consonant
    skeleton, ``s`` sorted full token set, ``a`` house-number+street). They are
    cheap and information-dense: measured on a 20k-entity train sample they
    beat the extended families at any fixed per-entity candidate budget
    (pair recall 0.734 @ 38 candidates/S1 vs 0.720 @ 58), because the extra
    low-information keys crowd out strong-key candidates in the top-k ranking.

    ``weak=True`` additionally emits the coverage families (``w`` length-ranked
    5-token windows, ``o`` original-order 4-grams, ``u`` unigrams, ``b``
    significant token pairs). They buy recall only at a much larger budget
    (pair recall 0.775 @ 106 candidates/S1) and are therefore opt-in.
    """
    keys: List[str] = []
    if postal:
        keys.append(f"p|{country}|{postal}")

    toks = core_name(name_norm).split() or name_norm.split()
    if toks:
        t0 = toks[0]
        if len(t0) >= 3:
            keys.append(f"n|{country}|{t0[:5]}")
        if len(toks) > 1 and len(toks[1]) >= 3:
            keys.append(f"n2|{country}|{toks[1][:5]}")
        skel = name_skeleton(t0)
        if len(skel) >= 4:
            keys.append(f"v|{country}|{skel[:5]}")
        sig = sorted({t for t in toks if len(t) >= 2})[:8]
        if sig:
            keys.append("s|{}|{}".format(country, " ".join(sig)[:60]))
        if not weak:
            return _address_keys(keys, country, addr_norm)
        # --- coverage families (opt-in, see docstring) ---------------------
        sig_all = sorted({t for t in toks})
        # w1: length-ranked window (keeps long distinctive tokens together —
        # catches one extra/missing token on either side).
        if len(sig_all) > 3:
            Swin = sorted(sig_all, key=len, reverse=True)
            for ws in range(max(len(Swin) - 4, 1)):
                win = sorted(Swin[ws:ws + 5])
                keys.append("w|{}|{}".format(country, "+".join(win)[:90]))
        # w2: ORIGINAL-ORDER sliding 4-grams (word-order/local-phrase robust:
        # an insertion elsewhere in the name doesn't shift the surviving
        # window, unlike the sorted full set).
        if len(toks) >= 4:
            for ws in range(len(toks) - 3):
                win = sorted(toks[ws:ws + 4])
                keys.append("o|{}|{}".format(country, "+".join(win)[:90]))
        # w3: UNIGRAM fallback — every single significant token with len>=4.
        # Threshold at 4 (not 5): the failing true pairs share short stubs
        # like "moore","prabhav","dermatology","green","chapel","center".
        # Common words ("services","limited","private") are pruned away by
        # pool/product frequency caps automatically.
        for t in set(toks):
            if len(t) >= 4:
                keys.append(f"u|{country}|{t[:8]}")
        # SUBSET fallback: every 2-combination of significant tokens.
        # Catches pairs that share only a rare 2-token core (e.g. test-side
        # DBA name vs registry name, missing middle tokens). Bounded: a
        # 6-token core emits 15 short keys; rare keys survive pruning.
        big = [t for t in sig_all if len(t) >= 4]
        if len(big) >= 2:
            for ia in range(len(big)):
                ta = big[ia]
                for tb in big[ia + 1:]:
                    a, b = (ta, tb) if ta < tb else (tb, ta)
                    keys.append(f"b|{country}|{a[:6]}+{b[:6]}")

    return _address_keys(keys, country, addr_norm)


def _address_keys(keys: List[str], country: str, addr_norm: str) -> List[str]:
    """Append the house-number + street-token key for one address."""
    atoks = addr_norm.split()
    for idx, t in enumerate(atoks):
        if t.isdigit():
            for st in atoks[idx + 1:]:
                if len(st) >= 3 and not st.isdigit():
                    keys.append(f"a|{country}|{t}|{st[:5]}")
                    break
            break
    return keys


def _keys_for_frame(df: pd.DataFrame, weak: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Explode a frame into (hashed_keys uint64, owner local-index int32).

    Keys of the same owner are contiguous (built in row order), which the
    chunked merge relies on.
    """
    h_parts: List[np.ndarray] = []
    i_parts: List[np.ndarray] = []
    countries = df["country"].to_numpy(dtype=object)
    norms = df["name_norm"].to_numpy(dtype=object)
    addrs = df["addr_norm"].to_numpy(dtype=object)
    postals = df["postal"].to_numpy(dtype=object)
    n = len(df)
    for start in range(0, n, 500_000):
        stop = min(start + 500_000, n)
        key_buf: List[int] = []
        idx_buf: List[int] = []
        for i in range(start, stop):
            ks = record_keys(countries[i], norms[i], addrs[i], postals[i], weak)
            # mask keeps the hash non-negative so it fits uint64 cleanly
            key_buf.extend([hash(k) & 0x7FFFFFFFFFFFFFFF for k in ks])
            idx_buf.extend([i] * len(ks))
        if key_buf:
            h_parts.append(np.asarray(key_buf, dtype=np.uint64))
            i_parts.append(np.asarray(idx_buf, dtype=np.int32))
    if not h_parts:
        return np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.int32)
    return np.concatenate(h_parts), np.concatenate(i_parts)


def _explode_topk(
    k1: np.ndarray,
    own1: np.ndarray,
    ks_sorted: np.ndarray,
    own_sorted: np.ndarray,
    n_pool: int,
    top_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Chunked merge + dedupe + per-entity top-k.

    Returns (local_s1_idx, local_pool_idx) kept pairs for the S1 records
    whose keys lie in ``k1`` (must all belong to distinct S1 records from
    other chunks — guaranteed by chunking on record boundaries).
    """
    out_i: List[np.ndarray] = []
    out_j: List[np.ndarray] = []
    n = own1.size
    a = 0
    while a < n:
        # sub-chunk on an OWNER boundary so every S1 record's keys stay in
        # one sub-chunk (dedupe + per-entity top-k stay exact)
        b = a + int(np.searchsorted(own1[a:], own1[a] + 1))
        if b - a > S1_CHUNK * 8:  # never trips in practice (≤ ~8 keys/record)
            b = a + int(
                np.searchsorted(own1[a : a + S1_CHUNK * 8], own1[a] + 1)
            )
        kk = k1[a:b]
        lo = np.searchsorted(ks_sorted, kk, side="left")
        hi = np.searchsorted(ks_sorted, kk, side="right")
        counts = (hi - lo).astype(np.int64)
        total = int(counts.sum())
        if total == 0:
            a = b
            continue
        rep_i = np.repeat(own1[a:b].astype(np.int64), counts)
        starts = np.cumsum(counts) - counts
        pos = np.repeat(lo.astype(np.int64), counts) + (
            np.arange(total, dtype=np.int64)
            - np.repeat(starts, counts)
        )
        rep_j = own_sorted[pos].astype(np.int64)
        packed = rep_i * n_pool + rep_j
        del rep_i, rep_j, pos, starts, counts, lo, hi
        uniq, cnt = np.unique(packed, return_counts=True)
        del packed
        li = uniq // n_pool
        lj = uniq % n_pool
        del uniq
        # rank by shared-key count within each entity (stable keeps entity order)
        order = np.argsort(-cnt.astype(np.int32), kind="stable")
        li, lj = li[order], lj[order]
        del order
        if li.size:
            boundary = np.empty(li.size, dtype=bool)
            boundary[0] = True
            np.not_equal(li[1:], li[:-1], out=boundary[1:])
            group_start = np.flatnonzero(boundary)
            group_id = np.cumsum(boundary) - 1
            rank = np.arange(li.size) - group_start[group_id]
            keep = rank < top_k
            li, lj = li[keep], lj[keep]
        out_i.append(li.astype(np.int32, copy=False))
        out_j.append(lj.astype(np.int32, copy=False))
        a = b
    if not out_i:
        empty = np.empty(0, dtype=np.int32)
        return empty, empty
    return np.concatenate(out_i), np.concatenate(out_j)


def build_candidates(
    s1: pd.DataFrame,
    pool: pd.DataFrame,
    cfg: BlockingConfig,
    log=print,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Return (ci, cj, stats): candidate pairs indexing ``s1`` and ``pool``."""
    t0 = time.time()
    s1_country = s1["country"].to_numpy(dtype=object)
    pool_country = pool["country"].to_numpy(dtype=object)
    n_s1 = len(s1)
    n_pool = len(pool)

    countries = sorted(set(s1_country.tolist()))
    log(f"[blocking] countries: {countries}")

    out_ci: List[np.ndarray] = []
    out_cj: List[np.ndarray] = []
    stats = dict.fromkeys(
        (
            "raw_pairs",
            "kept_pairs",
            "keys_used",
            "keys_pruned_pool",
            "keys_pruned_prod",
            "s1_with_candidates",
        ),
        0,
    )

    for country in countries:
        s1_local = np.flatnonzero(s1_country == country).astype(np.int32)
        pool_local = np.flatnonzero(pool_country == country).astype(np.int32)
        if s1_local.size == 0 or pool_local.size == 0:
            log(f"[blocking] {country}: no records on one side, skipping")
            continue

        s1_sub = s1.iloc[s1_local].reset_index(drop=True)
        pool_sub = pool.iloc[pool_local].reset_index(drop=True)

        k1, own1 = _keys_for_frame(s1_sub, cfg.weak_keys)
        kp, ownp = _keys_for_frame(pool_sub, cfg.weak_keys)
        del s1_sub, pool_sub
        if k1.size == 0 or kp.size == 0:
            log(f"[blocking] {country}: no keys, skipping")
            continue

        up, cp = np.unique(kp, return_counts=True)
        u1, c1 = np.unique(k1, return_counts=True)

        # pool-side popularity pruning
        keep_mask = cp <= cfg.max_pool_per_key
        pruned_pool = int((~keep_mask).sum())
        up, cp = up[keep_mask], cp[keep_mask]

        # align S1 key counts with surviving pool keys
        pos = np.searchsorted(up, u1)
        pos_clipped = np.minimum(pos, max(up.size - 1, 0))
        hit = (up.size > 0) & (pos < up.size) & (up[pos_clipped] == u1)
        cnt_pool = np.where(hit, cp[pos_clipped], 0).astype(np.int64)

        prod = c1.astype(np.int64) * cnt_pool
        valid = hit & (prod <= cfg.max_prod_per_key)
        pruned_prod = int((hit & ~valid).sum())
        valid_keys = u1[valid]
        del up, cp, u1, c1, cnt_pool, prod, valid, pos, pos_clipped, hit
        stats["keys_used"] += int(valid_keys.size)
        stats["keys_pruned_pool"] += pruned_pool
        stats["keys_pruned_prod"] += pruned_prod
        if valid_keys.size == 0:
            log(f"[blocking] {country}: all keys pruned, skipping")
            continue

        # restrict both sides to valid keys
        pos1 = np.searchsorted(valid_keys, k1)
        ok1 = (pos1 < valid_keys.size) & (
            valid_keys[np.minimum(pos1, valid_keys.size - 1)] == k1
        )
        k1f, own1f = k1[ok1], own1[ok1]
        del pos1, ok1
        posp = np.searchsorted(valid_keys, kp)
        okp = (posp < valid_keys.size) & (
            valid_keys[np.minimum(posp, valid_keys.size - 1)] == kp
        )
        kpf, ownpf = kp[okp], ownp[okp]
        n_valid_keys = int(valid_keys.size)
        del posp, okp, valid_keys, k1, own1, kp, ownp

        # sort pool keys once; every S1 key range is then a searchsorted
        order = np.argsort(kpf, kind="stable")
        ks_sorted = kpf[order]
        own_sorted = ownpf[order].astype(np.int64)
        del order, kpf, ownpf

        # _explode_topk sub-chunks internally on owner boundaries, so peak
        # memory stays a few hundred MB no matter the country size
        li, lj = _explode_topk(
            k1f, own1f, ks_sorted, own_sorted, n_pool, cfg.top_k
        )
        del k1f, own1f, ks_sorted, own_sorted
        kept = int(li.size)
        if li.size:
            out_ci.append(s1_local[li].astype(np.int32, copy=False))
            out_cj.append(pool_local[lj].astype(np.int32, copy=False))
        stats["kept_pairs"] += kept
        log(
            f"[blocking] {country}: S1={s1_local.size:,} pool={pool_local.size:,} "
            f"kept={kept:,} keys={n_valid_keys:,} "
            f"(pruned pool={pruned_pool:,} prod={pruned_prod:,}) "
            f"[{time.time() - t0:.1f}s]"
        )

    if out_ci:
        ci = np.concatenate(out_ci)
        cj = np.concatenate(out_cj)
    else:
        ci = np.empty(0, dtype=np.int32)
        cj = np.empty(0, dtype=np.int32)

    if ci.size:
        stats["s1_with_candidates"] = int(np.unique(ci).size)
    log(
        f"[blocking] total: pairs={ci.size:,} "
        f"avg_per_s1={ci.size / max(n_s1, 1):.1f} "
        f"s1_covered={stats['s1_with_candidates']:,}/{n_s1:,} "
        f"[{time.time() - t0:.1f}s]"
    )
    return ci, cj, stats

    if not out_i:
        empty = np.empty(0, dtype=np.int32)
        return empty, empty
    return np.concatenate(out_i), np.concatenate(out_j)

