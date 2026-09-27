#!/bin/bash
# Assemble the final submission zip:
#   <team>_submission.zip
#   ├── output/                         (matching_results.tsv, candidate_pairs.tsv)
#   ├── code/business_entity_resolution/ (src/, README.md, requirements.txt)
#   └── Documentation_template.md
#
# Run from anywhere:  bash make_submission_zip.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
TEAM="EntityResolvers"
STAGE="$ROOT/build_submission/${TEAM}_submission"
ZIP="$ROOT/build_submission/${TEAM}_submission.zip"

for f in output/matching_results.tsv output/candidate_pairs.tsv; do
    if [ ! -f "$ROOT/$f" ]; then
        echo "MISSING $f — run the pipeline first (bash run_submission.sh)"
        exit 1
    fi
done

rm -rf "$STAGE" "$ZIP"
mkdir -p "$STAGE/output" "$STAGE/code"

cp "$ROOT/output/matching_results.tsv" "$ROOT/output/candidate_pairs.tsv" \
   "$STAGE/output/"
cp -R "$ROOT/code/business_entity_resolution" "$STAGE/code/"
cp "$ROOT/Documentation_template.md" "$STAGE/"
rm -rf "$STAGE/code/business_entity_resolution/src/__pycache__"
find "$STAGE" -name '.DS_Store' -delete

( cd "$ROOT/build_submission" && zip -qr "$(basename "$ZIP")" "$(basename "$STAGE")" )
echo "built: $ZIP"
unzip -l "$ZIP" | tail -20
du -sh "$ZIP"
