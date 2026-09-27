"""CLI entry point for the business entity-resolution pipeline.

Run from ``code/business_entity_resolution/`` with the workspace Python::

    python -m src.pipeline predict  --test-dir ../../dataset/test  --output-dir ../../output
    python -m src.pipeline train    --train-dir ../../dataset/train --artifacts ../../artifacts
    python -m src.pipeline evaluate --pred ../../output/matching_results.tsv \\
                                     --gt  ../../dataset/train/train_ground_truth.tsv

``predict`` runs blocking -> features -> scoring and writes
``matching_results.tsv`` + ``candidate_pairs.tsv``.

``train`` is a no-op-safe command: it requires the training source files
(``train_source1/2/3.tsv``) which may be missing from the workspace; it will
tell you so instead of crashing. When present it fits a gradient-boosted model
on blocking candidates, tunes the F_0.5 threshold on a held-out entity split,
and saves ``artifacts/model.pkl`` + ``artifacts/threshold.json`` which
``predict`` picks up automatically.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

from .blocking import build_candidates
from .config import BlockingConfig, ScoringConfig
from .features import FEATURE_NAMES, compute_features
from .io_data import load_ground_truth, load_source, write_pair_file
from .metrics import macro_f05, parse_id_list
from .scorer import Scorer, save_model


def log(msg: str) -> None:
    print(msg, flush=True)


def _pair_columns(df):
    """Tuple of list/array columns consumed by the feature workers.

    String columns become Python lists (faster item access than numpy object
    arrays and they share the same string objects — no duplication).
    """
    return (
        df["name_norm"].tolist(),
        df["name_k"].tolist(),
        df["addr_norm"].tolist(),
        df["postal"].tolist(),
    )


def _stack(ci, cj):
    if len(ci) == 0:
        return np.empty((0, 2), dtype=np.int64)
    return np.stack([ci.astype(np.int64), cj.astype(np.int64)], axis=1)


def _log_score_stats(score_path: str, threshold: float) -> None:
    """Histogram / quantiles of cached scores + reference calibration priors
    from the training ground truth (avg 3.46 matches per S1, 5.6% singletons)."""
    s = np.load(score_path, mmap_mode="r")
    if s.size == 0:
        return
    qs = [10, 25, 50, 75, 90, 99]
    vals = np.percentile(s, qs)
    log("[score] quantiles: " + ", ".join(
        f"p{q}={v:.3f}" for q, v in zip(qs, vals)))
    acc = int((s >= threshold).sum())
    hist, edges = np.histogram(s, bins=np.linspace(0.0, 1.0, 21))
    log("[score] histogram (0.0->1.0, width .05):")
    for c, e0, e1 in zip(hist, edges[:-1], edges[1:]):
        bar = "#" * int(60 * c / max(hist.max(), 1))
        log(f"    {e0:.2f}-{e1:.2f} {c:>12,} {bar}")
    log(f"[score] accepted @ {threshold:.3f}: {acc:,} links "
        f"({acc * 100.0 / s.size:.1f}% of candidates). "
        f"Train prior: ~3.46 true links/S1, ~5.6% singletons.")


# --------------------------------------------------------------------------
# predict
# --------------------------------------------------------------------------
def cmd_predict(args) -> int:
    t0 = time.time()
    cfg = BlockingConfig(
        max_pool_per_key=args.max_pool_per_key,
        max_prod_per_key=args.max_prod_per_key,
        top_k=args.top_k,
    )
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    log(f"[load] S1 test: {args.test_dir}/test_source1.tsv (limit={args.limit_s1})")
    s1 = load_source(
        os.path.join(args.test_dir, "test_source1.tsv"), limit=args.limit_s1
    )
    log(f"[load] S1 rows={len(s1):,}")

    log("[load] pool: test_source2.tsv + test_source3.tsv "
        f"(limit={args.limit_pool})")
    p2 = load_source(
        os.path.join(args.test_dir, "test_source2.tsv"), limit=args.limit_pool
    )
    p3 = load_source(
        os.path.join(args.test_dir, "test_source3.tsv"), limit=args.limit_pool
    )
    import pandas as pd

    pool = pd.concat([p2, p3], ignore_index=True)
    del p2, p3
    log(f"[load] pool rows={len(pool):,} [{time.time() - t0:.1f}s]")

    s1_ids = s1["entity_id"].to_numpy(dtype=object)
    pool_ids = pool["entity_id"].to_numpy(dtype=object)

    ci, cj, bstats = build_candidates(s1, pool, cfg, log=log)

    cand_path = os.path.join(out_dir, "candidate_pairs.tsv")
    rows, non_empty = write_pair_file(
        cand_path, s1_ids, pool_ids, ci, cj, header_name="candidate_entity_ids"
    )
    log(
        f"[write] {cand_path}: {rows:,} rows, {non_empty:,} with candidates "
        f"[{time.time() - t0:.1f}s]"
    )

    scorer = Scorer.load(args.model, threshold=args.threshold, cfg=ScoringConfig())
    log(f"[score] scorer={scorer.kind} threshold={scorer.threshold:.3f}")

    # convert the frames to worker-friendly lists, then free the frames (the
    # normalized strings themselves are shared, not copied)
    s1_cols = _pair_columns(s1)
    pool_cols = _pair_columns(pool)
    del s1, pool
    import gc

    gc.collect()

    n_pairs = int(ci.size)
    # pre-allocate an on-disk score cache so the threshold can be re-tuned
    # later (subcommand `apply`) without recomputing features
    cache_dir = out_dir
    cache_i = np.lib.format.open_memmap(
        os.path.join(cache_dir, "score_pair_i.npy"), mode="w+",
        dtype=np.int32, shape=(n_pairs,),
    )
    cache_j = np.lib.format.open_memmap(
        os.path.join(cache_dir, "score_pair_j.npy"), mode="w+",
        dtype=np.int32, shape=(n_pairs,),
    )
    cache_s = np.lib.format.open_memmap(
        os.path.join(cache_dir, "score_pair_s.npy"), mode="w+",
        dtype=np.float32, shape=(n_pairs,),
    )
    acc_i: list = []
    acc_j: list = []
    offset = 0
    scores_seen = 0
    for X in compute_features(
        ci, cj, s1_cols, pool_cols, workers=args.workers
    ):
        stop = offset + X.shape[0]
        s = scorer.score_block(X)
        cache_i[offset:stop] = ci[offset:stop]
        cache_j[offset:stop] = cj[offset:stop]
        cache_s[offset:stop] = s
        mask = s >= scorer.threshold
        if mask.any():
            acc_i.append(ci[offset:stop][mask])
            acc_j.append(cj[offset:stop][mask])
        scores_seen += X.shape[0]
        offset = stop
        if scores_seen and scores_seen % 5_000_000 < X.shape[0]:
            log(f"[score] {scores_seen:,} pairs scored "
                f"[{time.time() - t0:.1f}s]")
    cache_i.flush()
    cache_j.flush()
    cache_s.flush()
    del cache_i, cache_j, cache_s
    log(f"[score] cache written to {cache_dir}/score_pair_*.npy")
    _log_score_stats(
        os.path.join(cache_dir, "score_pair_s.npy"), scorer.threshold
    )

    if acc_i:
        ai = np.concatenate(acc_i)
        aj = np.concatenate(acc_j)
    else:
        ai = np.empty(0, dtype=np.int32)
        aj = np.empty(0, dtype=np.int32)

    match_path = os.path.join(out_dir, "matching_results.tsv")
    rows, non_empty = write_pair_file(
        match_path, s1_ids, pool_ids, ai, aj, header_name="matched_entity_ids"
    )
    log(
        f"[write] {match_path}: {rows:,} rows, {non_empty:,} with matches, "
        f"{ai.size:,} total match links "
        f"(avg {ai.size / max(len(s1_ids), 1):.2f}/S1) [{time.time() - t0:.1f}s]"
    )
    if rows != len(s1_ids) and args.limit_s1 is None:
        log("[warn] row count != S1 count")
    log(f"[done] total {time.time() - t0:.1f}s")


# --------------------------------------------------------------------------
# apply (re-threshold from the score cache, no recompute)
# --------------------------------------------------------------------------
def cmd_apply(args) -> int:
    """Rewrite submission files from a cached score run at a new threshold."""
    t0 = time.time()
    ci_p = os.path.join(args.cache_dir, "score_pair_i.npy")
    cj_p = os.path.join(args.cache_dir, "score_pair_j.npy")
    cs_p = os.path.join(args.cache_dir, "score_pair_s.npy")
    for p in (ci_p, cj_p, cs_p):
        if not os.path.isfile(p):
            log(f"[apply] missing cache file: {p} — run `predict` first")
            return 2

    import pandas as pd

    s1_ids = pd.read_csv(
        os.path.join(args.test_dir, "test_source1.tsv"), sep="\t",
        usecols=["entity_id"], dtype=str,
    )["entity_id"].to_numpy(dtype=object)
    pool_ids = np.concatenate(
        [
            pd.read_csv(os.path.join(args.test_dir, f"test_source{s}.tsv"),
                        sep="\t", usecols=["entity_id"], dtype=str,
                        )["entity_id"].to_numpy(dtype=object)
            for s in (2, 3)
        ]
    )

    ci = np.load(ci_p, mmap_mode="r")
    cj = np.load(cj_p, mmap_mode="r")
    scores = np.load(cs_p, mmap_mode="r")
    thr = args.threshold
    _log_score_stats(cs_p, thr)

    os.makedirs(args.output_dir, exist_ok=True)
    cand_path = os.path.join(args.output_dir, "candidate_pairs.tsv")
    rows, ne = write_pair_file(
        cand_path, s1_ids, pool_ids,
        np.asarray(ci), np.asarray(cj), header_name="candidate_entity_ids",
    )
    log(f"[write] {cand_path}: {rows:,} rows, {ne:,} non-empty")

    mask = np.asarray(scores) >= thr
    match_path = os.path.join(args.output_dir, "matching_results.tsv")
    rows, ne = write_pair_file(
        match_path, s1_ids, pool_ids,
        np.asarray(ci)[mask], np.asarray(cj)[mask],
        header_name="matched_entity_ids",
    )
    log(f"[write] {match_path}: {rows:,} rows, {ne:,} with matches "
        f"(avg {mask.sum() / max(len(s1_ids), 1):.2f}/S1, "
        f"empty {100.0 * (rows - ne) / max(rows, 1):.1f}%) "
        f"[{time.time() - t0:.1f}s]")
    return 0


# --------------------------------------------------------------------------
# train
# --------------------------------------------------------------------------
def _holdout_mask(s1_ids: np.ndarray, frac: float) -> np.ndarray:
    """Deterministic ~frac holdout of entities, keyed on the id string."""
    keep = np.empty(len(s1_ids), dtype=bool)
    for i, sid in enumerate(s1_ids):
        # stable across runs (hash() is salted — use a simple FNV-1a instead)
        h = 0xCBF29CE484222325
        for b in str(sid).encode("utf-8"):
            h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
        keep[i] = (h % 10_000) < int(frac * 10_000)
    return keep


def _tune_threshold(scores, correct, group_id, n_groups, truth_n, grid):
    """Grid-search the accept threshold maximising macro F_0.5 on a split.

    scores:   (P,) scores of candidate pairs
    correct:  (P,) 1.0 when the pair is a true match else 0.0
    group_id: (P,) int group index of each pair in [0, n_groups)
    truth_n:  (G,) number of true matches per group (0 for singletons)
    """
    order = np.argsort(-scores)
    scores = scores[order]
    correct = correct[order]
    group_id = group_id[order]
    best_t, best_f = float(grid[0]), -1.0
    for t in grid:
        m = scores >= t
        g = group_id[m]
        c = correct[m]
        pred_n = np.bincount(g, minlength=n_groups).astype(np.float64)
        tp = np.bincount(g, weights=c, minlength=n_groups)
        with np.errstate(divide="ignore", invalid="ignore"):
            prec = np.where(pred_n > 0, tp / np.maximum(pred_n, 1), 0.0)
            rec = np.where(truth_n > 0, tp / np.maximum(truth_n, 1), 0.0)
        b2 = 0.25
        f = np.where(
            (pred_n > 0) | (truth_n > 0),
            (1 + b2) * prec * rec / np.maximum(b2 * prec + rec, 1e-12),
            1.0,
        )
        score = float(f.mean())
        if score > best_f:
            best_f, best_t = score, float(t)
    return best_t, best_f



def cmd_train(args) -> int:
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier

    t0 = time.time()
    needed = [
        os.path.join(args.train_dir, f"train_source{i}.tsv") for i in (1, 2, 3)
    ]
    missing = [p for p in needed if not os.path.isfile(p)]
    if missing:
        log("[train] training source files are MISSING — cannot fit a model:")
        for p in missing:
            log(f"        - {p}")
        log("[train] place them there and re-run. Until then `predict` uses "
            "the built-in heuristic scorer.")
        return 2

    cfg = BlockingConfig(
        max_pool_per_key=args.max_pool_per_key,
        max_prod_per_key=args.max_prod_per_key,
        top_k=args.top_k,
    )
    s1 = load_source(needed[0], limit=args.limit_s1)
    pool = pd.concat(
        [load_source(needed[1]), load_source(needed[2])], ignore_index=True
    )
    gt = load_ground_truth(os.path.join(args.train_dir, "train_ground_truth.tsv"))
    log(f"[train] S1={len(s1):,} pool={len(pool):,} gt_entities={len(gt):,} "
        f"[{time.time() - t0:.1f}s]")

    s1_ids = s1["entity_id"].to_numpy(dtype=object)
    ci, cj, _ = build_candidates(s1, pool, cfg, log=log)
    if ci.size == 0:
        log("[train] no candidates produced — check blocking config")
        return 1

    ho = _holdout_mask(s1_ids, args.holdout_frac)
    log(f"[train] holdout entities: {int(ho.sum()):,}/{len(s1_ids):,}")

    # stream features; keep all positives, down-sample negatives
    s1_ids_list = s1_ids.tolist()
    pool_ids_arr = pool_ids
    rng = np.random.default_rng(13)
    X_keep, y_keep = [], []
    hold = []  # (ci, y, X) of holdout entities
    offset = 0
    for X in compute_features(
        ci, cj, _pair_columns(s1), _pair_columns(pool), workers=args.workers
    ):
        stop = offset + X.shape[0]
        blk_i = ci[offset:stop]
        blk_j = cj[offset:stop]
        y = np.empty(X.shape[0], dtype=np.float32)
        for r in range(X.shape[0]):
            truth = gt.get(s1_ids_list[blk_i[r]], ())
            y[r] = 1.0 if pool_ids_arr[blk_j[r]] in truth else 0.0
        is_hold = ho[blk_i]
        tr = ~is_hold
        pos = np.flatnonzero(tr & (y > 0.5))
        neg = np.flatnonzero(tr & (y <= 0.5))
        if neg.size:
            neg = neg[rng.random(neg.size) < args.neg_sample]
        sel = np.concatenate([pos, neg])
        if sel.size:
            X_keep.append(X[sel])
            y_keep.append(y[sel])
        if is_hold.any():
            hold.append(
                (blk_i[is_hold].copy(), y[is_hold].copy(), X[is_hold])
            )
        offset = stop

    Xtr = np.concatenate(X_keep) if X_keep else np.empty((0, len(FEATURE_NAMES)))
    ytr = np.concatenate(y_keep) if y_keep else np.empty(0)
    log(f"[train] fit rows={len(ytr):,} positives={int(ytr.sum()):,} "
        f"[{time.time() - t0:.1f}s]")
    if len(ytr) == 0 or ytr.sum() == 0:
        log("[train] no positive examples — check ground truth alignment")
        return 1

    clf = HistGradientBoostingClassifier(
        max_iter=args.max_iter, learning_rate=0.1, random_state=7
    )
    clf.fit(Xtr, ytr)
    log(f"[train] model fitted [{time.time() - t0:.1f}s]")

    # threshold tuning on holdout entities
    if hold:
        hb = np.concatenate([h[0] for h in hold])
        hy = np.concatenate([h[1] for h in hold])
        hX = np.concatenate([h[2] for h in hold])
    else:
        hb = np.empty((0, 2), np.int64)
        hy = np.empty(0)
        hX = np.empty((0, len(FEATURE_NAMES)))
    if len(hy):
        hs = clf.predict_proba(hX)[:, 1]
        hold_ids = np.unique(hb)
        gid_map = {int(v): k for k, v in enumerate(hold_ids)}
        gids = np.fromiter(
            (gid_map[int(v)] for v in hb), dtype=np.int64, count=hb.size
        )
        truth_n = np.array(
            [len(gt.get(str(v), ())) for v in hold_ids], dtype=np.float64
        )
        grid = np.round(np.arange(0.30, 0.901, 0.01), 4)
        thr, val = _tune_threshold(
            hs, hy.astype(np.float64), gids, len(hold_ids), truth_n, grid
        )
        log(f"[train] threshold={thr:.3f} holdout macro-F0.5={val:.4f}")
    else:
        thr, val = ScoringConfig().heuristic_threshold, float("nan")
        log("[train] empty holdout — defaulting threshold")

    os.makedirs(args.artifacts, exist_ok=True)
    save_model(clf, thr, os.path.join(args.artifacts, "model.pkl"))
    log(f"[train] saved model + threshold to {args.artifacts}/ "
        f"[{time.time() - t0:.1f}s]")
    return 0



# --------------------------------------------------------------------------
# evaluate
# --------------------------------------------------------------------------
def cmd_evaluate(args) -> int:
    """Macro F_0.5 of a predictions file against a ground-truth file."""
    pred: dict = {}
    with open(args.pred, encoding="utf-8") as f:
        next(f, None)
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            pred[parts[0]] = parse_id_list(parts[1] if len(parts) > 1 else "")
    gt = load_ground_truth(args.gt)
    missing = [k for k in gt if k not in pred]
    score = macro_f05(pred, gt)
    log(f"[evaluate] entities={len(gt):,} pred_rows={len(pred):,} "
        f"missing_in_pred={len(missing):,}")
    log(f"[evaluate] macro F_0.5 = {score:.4f}")
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _add_blocking_args(p) -> None:
    p.add_argument("--top-k", type=int, default=BlockingConfig.top_k,
                   help="max candidates per S1 entity")
    p.add_argument("--max-pool-per-key", type=int,
                   default=BlockingConfig.max_pool_per_key)
    p.add_argument("--max-prod-per-key", type=int,
                   default=BlockingConfig.max_prod_per_key)
    p.add_argument("--workers", type=int, default=None,
                   help="feature-computation processes (default: cpus-1)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m src.pipeline",
        description="Business entity-resolution pipeline (ML Challenge 2026).",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("predict", help="run pipeline on the test set")
    p.add_argument("--test-dir", default="../../dataset/test")
    p.add_argument("--output-dir", default="../../output")
    p.add_argument("--model", default="../../artifacts/model.pkl",
                   help="trained model (falls back to heuristic scorer)")
    p.add_argument("--threshold", type=float, default=None,
                   help="accept threshold (default: tuned/heuristic value)")
    p.add_argument("--limit-s1", type=int, default=None,
                   help="smoke test: only first N S1 rows")
    p.add_argument("--limit-pool", type=int, default=None,
                   help="smoke test: only first N rows per pool file")
    _add_blocking_args(p)
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser(
        "apply",
        help="rewrite outputs at a new threshold from the predict score cache",
    )
    p.add_argument("--cache-dir", default="../../output",
                   help="directory holding score_pair_{i,j,s}.npy")
    p.add_argument("--test-dir", default="../../dataset/test")
    p.add_argument("--output-dir", default="../../output")
    p.add_argument("--threshold", type=float, required=True)
    p.set_defaults(func=cmd_apply)

    p = sub.add_parser("train", help="fit model on training data")
    p.add_argument("--train-dir", default="../../dataset/train")
    p.add_argument("--artifacts", default="../../artifacts")
    p.add_argument("--holdout-frac", type=float, default=0.15)
    p.add_argument("--neg-sample", type=float, default=0.25,
                   help="fraction of negative candidate pairs to keep")
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--limit-s1", type=int, default=None)
    _add_blocking_args(p)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="macro F_0.5 of predictions vs GT")
    p.add_argument("--pred", required=True)
    p.add_argument("--gt", required=True)
    p.set_defaults(func=cmd_evaluate)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

