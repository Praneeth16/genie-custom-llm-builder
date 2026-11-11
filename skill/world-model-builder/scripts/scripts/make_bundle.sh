#!/usr/bin/env bash
# Generate a Databricks Asset Bundle for the batch-scoring job from workloads/batch_score.yaml,
# with this workspace's locations filled in from .env. The bundle holds workspace-specific values,
# so it is written outside the repo. It cannot live in a gitignored folder here: bundle packaging
# skips ignored files and would find no code.
#   ./scripts/make_bundle.sh && cd /tmp/jepa-wm-bundle && databricks bundle deploy -p <profile>
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
out=${BUNDLE_DIR:-/tmp/jepa-wm-bundle}
rm -rf "$out" && mkdir -p "$out/code"
sed -e "s|/Volumes/catalog/schema|/Volumes/$JEPA_CATALOG/$JEPA_SCHEMA|" \
    -e "s|catalog.schema|$JEPA_CATALOG.$JEPA_SCHEMA|" \
    -e "s|0123456789abcdef|$WAREHOUSE_ID|" \
    -e "s|/Workspace/Shared/jepa_wm|$MLFLOW_EXPERIMENT_DIRECTORY|" \
    -e 's|root_path: ".."|root_path: "code"|' \
    workloads/batch_score.yaml > "$out/batch_score.yaml"
cp -R src train "$out/code/"  # convert-to-dabs needs the code in a subfolder of the bundle
find "$out/code" -name __pycache__ -prune -exec rm -rf {} +
databricks air convert-to-dabs "$out/batch_score.yaml" --output-dir "$out" --force
