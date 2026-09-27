#!/bin/bash
# Full supervised submission run: fit the pair-scoring model on the train
# split (blocking candidates labelled by train_ground_truth.tsv), then
# predict on the test set with the learned model + tuned F_0.5 threshold,
# then validate the output files.
#
# Run from anywhere:  bash run_submission.sh
# Progress log:       run_full.log  (written next to this script)
#
# MEMORY BUDGET (important on a 16 GB machine).  The train pool file holds
# 10.3M records and the test pool 9.97M, and both are held in RAM as Python
# string objects while blocking + feature extraction run.  Observed swap
# thrashing (8 GB+ swap, feature workers stalled below 20% CPU) at 48M
# candidate pairs, so the defaults below are sized to stay inside ~6 GB RSS:
#   * train 250k S1 x top_k 120  ~ 22M candidate pairs
#   * predict     1.73M S1 x top_k 120  ~ 132M blocking pairs, which the
#     similarity prefilter (name>=70 OR addr>=80, on by default) trims to
#     ~77M — candidate_pairs.tsv must also fit the 512 MB submission cap.
# Raise --top-k only with the free-RAM headroom to match (candidate count is
# NOT scored on the leaderboard, but scoring time scales with it).
set -x
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT/code/business_entity_resolution"
PY="$ROOT/.venv/bin/python"

# 1) train -> artifacts/model.pkl + artifacts/threshold.json
#    250k entities give ~860k labelled positives for the 18-feature model.
"$PY" -m src.pipeline train --max-train-s1 250000 --top-k 120 \
    --neg-sample 0.12 --workers 5
echo "TRAIN_RC=$?"

# 2) predict -> output/matching_results.tsv + output/candidate_pairs.tsv
#    (picks up artifacts/model.pkl + threshold.json automatically)
"$PY" -m src.pipeline predict --top-k 120
echo "PREDICT_RC=$?"

# 3) validate submission format
cd "$ROOT"
"$PY" utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv
echo "VALIDATE_RC=$?"

echo "RUN_DONE $(date)"
