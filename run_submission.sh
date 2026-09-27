#!/bin/bash
# Full supervised submission run: fit the pair-scoring model on the train
# split (blocking candidates labelled by train_ground_truth.tsv), then
# predict on the test set with the learned model + tuned F_0.5 threshold,
# then validate the output files.
#
# Run from anywhere:  bash run_submission.sh
# Progress log:       run_full.log  (written next to this script)
set -x
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT/code/business_entity_resolution"
PY="$ROOT/.venv/bin/python"

# 1) train -> artifacts/model.pkl + artifacts/threshold.json
#    --max-train-s1 keeps peak RAM safe on a 16 GB machine (800k entities
#    still give ~2.8M labeled positives for the 7-feature model).
"$PY" -m src.pipeline train --max-train-s1 800000
echo "TRAIN_RC=$?"

# 2) predict -> output/matching_results.tsv + output/candidate_pairs.tsv
#    (picks up artifacts/model.pkl + threshold.json automatically)
"$PY" -m src.pipeline predict
echo "PREDICT_RC=$?"

# 3) validate submission format
cd "$ROOT"
"$PY" utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv
echo "VALIDATE_RC=$?"

echo "RUN_DONE $(date)"
