# Business Entity Resolution — ML Challenge 2026

Runnable pipeline: blocking (candidate generation) → pair features → scoring →
submission TSVs.

## Structure

```
code/business_entity_resolution/
├── README.md
├── requirements.txt
└── src/
    ├── pipeline.py    # CLI entry point: predict / train / evaluate
    ├── normalize.py   # name/address normalisation, postal extraction
    ├── blocking.py    # multi-key blocking, popularity pruning, per-S1 cap
    ├── features.py    # rapidfuzz pair features (multiprocess, fork-based)
    ├── scorer.py      # trained model wrapper + heuristic fallback
    ├── metrics.py     # macro F_0.5 (challenge metric)
    ├── io_data.py     # TSV reading / submission writing
    └── config.py      # blocking & scoring knobs
```

## Usage

From `code/business_entity_resolution/` (use the workspace venv):

```bash
PY=../../.venv/bin/python

# 1. Full test-set run -> ../../output/matching_results.tsv + candidate_pairs.tsv
#    (also caches every pair score in ../../output/score_pair_*.npy)
$PY -m src.pipeline predict --test-dir ../../dataset/test --output-dir ../../output

# 2. Re-threshold from the score cache — instant, no feature recompute
$PY -m src.pipeline apply --threshold 0.65 \
    --test-dir ../../dataset/test --output-dir ../../output

# 3. Train a model (needs dataset/train/train_source{1,2,3}.tsv)
$PY -m src.pipeline train --train-dir ../../dataset/train --artifacts ../../artifacts

# 4. Score any prediction file against ground truth
$PY -m src.pipeline evaluate --pred ../../output/matching_results.tsv \
    --gt ../../dataset/train/train_ground_truth.tsv
```

Then validate the outputs from the `student_resource/` directory:

```bash
.venv/bin/python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

Useful flags: `--limit-s1 20000 --limit-pool 200000` for a fast smoke run,
`--threshold`, `--top-k`, `--workers`.

## Methodology

**Blocking.** Six country-scoped keys per record — postal code, first/second
name-token prefix, sorted name tokens (word-order robustness), consonant
skeleton (vowel/ transliteration robustness), house-number+street — merged via
hashed keys. Popularity pruning drops keys whose pool-side document frequency
or `n_S1 × n_pool` product exceeds a cap; each S1 entity keeps at most
`--top-k` (60) candidates ranked by number of distinct shared keys. Blocking
runs country-by-country to bound peak memory (country labels are consistent
across all three sources; unknown countries — France in the test set — are
handled as an open set since nothing is hard-coded).

**Features (7).** rapidfuzz `token_set_ratio` on suffix-stripped core names,
`token_sort_ratio` and `ratio` on normalised names, `token_sort_ratio` and
`ratio` on normalised addresses (0 when either address is missing), an
address-both-present flag, and a postal-code match flag.

**Scoring.** If `artifacts/model.pkl` exists (produced by `train`), a
gradient-boosted trees model (`HistGradientBoostingClassifier`) scores the
pairs with a threshold tuned for macro F_0.5 on a held-out entity split.
Otherwise a documented precision-leaning heuristic score is used with default
threshold 0.58 (tune with `--threshold`).

**Output.** `candidate_pairs.tsv` is exactly the set the scorer runs over;
`matching_results.tsv` keeps pairs scoring ≥ threshold, one row per test S1
entity, empty cell for singletons.

## Notes / limitations

* The workspace currently ships **only** `dataset/train/train_ground_truth.tsv`
  — the `train_source*.tsv` files are missing, so the heuristic scorer is the
  default. Drop the source files in place and run `train` to switch to the
  learned model (threshold is then tuned on real data instead of being a
  hand-set default).
* Empty addresses exist in the pool sources (~130k/136k rows in S2/S3); the
  feature code treats a missing address as "no evidence" (score contribution
  0) rather than a match.
