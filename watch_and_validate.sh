#!/bin/bash
# Waits for the predict run (given PID) to exit, then validates the
# submission files and appends the result to watch_validate.log.
PID="$1"
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
cd /Users/namitraj/Downloads/student_resource
.venv/bin/python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv
echo "WATCH_DONE $(date)"
