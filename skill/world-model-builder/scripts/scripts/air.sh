#!/usr/bin/env bash
# Submit an AI Runtime workload with this workspace's locations filled in.
#   ./scripts/air.sh workloads/sensor.yaml --dry-run
#   ./scripts/air.sh workloads/sensor.yaml --idempotency-key <key> [--override ...]
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
file=$1; shift
extra=()
if grep -q UC_SCHEMA "$file"; then
  extra+=(--override "env_variables.UC_SCHEMA=$JEPA_CATALOG.$JEPA_SCHEMA" --override "env_variables.WAREHOUSE_ID=$WAREHOUSE_ID")
fi
databricks air run --file "$file" --profile "$DATABRICKS_PROFILE" \
  --override "mlflow_experiment_directory=$MLFLOW_EXPERIMENT_DIRECTORY" \
  --override "env_variables.OUTPUT=/Volumes/$JEPA_CATALOG/$JEPA_SCHEMA/artifacts/$(grep -o 'artifacts/[a-z_]*' "$file" | head -1 | cut -d/ -f2)" \
  ${extra[@]+"${extra[@]}"} "$@"
