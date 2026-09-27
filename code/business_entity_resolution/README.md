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

The submission package ships `artifacts/model.pkl` + `artifacts/threshold.json`
one level up (`../../artifacts/`), so `predict` reproduces the submitted TSVs
directly; run `train` only if you want to refit from scratch.

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
skeleton (vowel/transliteration robustness), house-number+street — merged via
hashed keys. Popularity pruning drops keys whose pool-side document frequency
or `n_S1 × n_pool` product exceeds a cap; each S1 entity keeps at most
`--top-k` (120) candidates ranked by number of distinct shared keys. Blocking
runs country-by-country to bound peak memory (country labels are consistent
across all three sources; unknown countries — France in the test set — are
handled as an open set since nothing is hard-coded).

Measured on a 20k-entity train sample against the full 10.3M pool (pair recall /
candidates per S1): 0.705/47 at `--top-k 60`, 0.736/72 at 100, 0.760/104 at 160,
0.810/231 at 500. Recall is governed by `--top-k`, not by the number of key
families: an extended key set (unigrams, token pairs, token windows) is available
behind `--weak-keys` but was measured to add no recall at an equal candidate
budget (0.720 @ 58 vs 0.734 @ 38), because low-information keys let junk pairs
outrank the distinctive key that identifies the true partner. It is therefore
off by default. `candidate_pairs.tsv` is not scored on the leaderboard, so
`--top-k` is bounded only by how much scoring time you have.

**Features (18).** Columns 0-6: rapidfuzz `token_set_ratio` on suffix-stripped
core names, `token_sort_ratio` and `ratio` on normalised names,
`token_sort_ratio` and `ratio` on normalised addresses (0 when either address is
missing), an address-both-present flag, and a postal-code match flag. Columns
7-17 (added after the first leaderboard feedback): `WRatio` on core names and on
addresses, `token_set_ratio` on full names and on addresses, exact-match
indicators for full name / core name / address / consonant skeleton,
postal-region-prefix agreement, and name/address token-count ratios. See
`features.FEATURE_NAMES` for the authoritative order.

**Scoring.** If `artifacts/model.pkl` exists (produced by `train`), a
gradient-boosted trees model (`HistGradientBoostingClassifier`) scores the
pairs with a threshold tuned for macro F_0.5 on a held-out entity split.
Otherwise a documented precision-leaning heuristic score is used with default
threshold 0.58 (tune with `--threshold`).

**Output.** `candidate_pairs.tsv` is exactly the set the scorer runs over;
`matching_results.tsv` keeps pairs scoring ≥ threshold, one row per test S1
entity, empty cell for singletons.

## Training the shipped model (reproduces `artifacts/model.pkl`)

```bash
PY=../../.venv/bin/python
$PY -m src.pipeline train --max-train-s1 250000 --top-k 120 \
    --neg-sample 0.12 --workers 5
```

`train` prints the diagnostics that drove every design decision, so you can
re-derive them:

* `BLOCKING RECALL pair=… entity-full=…` — the recall ceiling of the candidate set
* `blocking CEILING macro-F0.5 (perfect precision) = …`
* `tuned threshold=… holdout macro-F0.5=…` (grid search over 0.30-0.90)
* a full threshold sweep table with micro precision/recall and links per entity

Reference run (250k Source-1 sample, `top_k=120`, 20.6M candidate pairs):

```
[train] holdout entities=37,718 singletons=2,066 true_links=130,668
[train] BLOCKING RECALL pair=0.7439 entity-full=0.4867
[train] blocking CEILING macro-F0.5 (perfect precision) = 0.8787
[train] tuned threshold=0.900 holdout macro-F0.5=0.7900   (micro P 0.920 / R 0.672)
```

The holdout average includes held-out entities that blocking left with **zero**
candidates; they can only score 0 and are part of the real metric, so dropping
them (as a naive implementation does) overstates the score.

## Notes / limitations

* **Memory is the binding constraint on a 16 GB machine.** The 10M-record pool is
  held in RAM as Python strings for both blocking and feature extraction; at ~48M
  candidate pairs we observed swap thrashing (8 GB+ swap, feature workers stalled
  below 20% CPU). The defaults above stay inside ~6 GB RSS. If you raise `--top-k`
  or `--max-train-s1`, watch the swap: candidate count is free on the leaderboard
  but scoring time and peak memory are not.
* Feature computation is CPU- and memory-bandwidth-bound: measured ≈9 µs/pair
  in-process, but end-to-end throughput ranges 25-75k pairs/s on the test set
  depending on how much RAM the rest of the machine is using.
* `--weak-keys` is implemented and off by default; see the measurement above.
* Empty addresses exist in the pool sources; the feature code treats a missing
  address as "no evidence" (`addr_*` = 0, `addr_both_present` = 0) rather than as
  a match, and feature 5 exposes presence to the model instead.
* Cross-script transliteration (e.g. a Telugu-script rendering of a business whose
  other records are Latin script) is not reachable by any string-overlap key and
  is the largest remaining blocking gap; see the delivered documentation, §5.3.

