# AI Runtime glue

Leave workload field details and lifecycle commands to `databricks-ai-runtime`. Confirm fields with `databricks air run -h config.<field>` and a dry run before every submission.

## Workload files

- Set `environment.version: databricks_ai_v6`. This image has torch and transformers. The plain value `"6"` selects a Python without torch.
- Set `max_retries: 0` while developing. The default is not zero.
- `code_source.root_path` is relative to the YAML file. Add `include_paths` so that the upload skips virtual environments and run outputs.
- Keep placeholder values in `env_variables` and replace them at submission with `--override env_variables.NAME=value`. `scripts/air.sh` does this from a `.env` file, so no workspace name is committed.
- Add extra packages under `environment.dependencies`, e.g., `av` to decode video.
- Use an idempotency key on every submission and change it for every resubmission.
- Use `timeout 30 databricks air logs <id> --tail 80`. Without the timeout the command streams until the run ends.

## MLflow

AI Runtime sets `MLFLOW_RUN_ID` for the workload. Call plain `mlflow.start_run()` to attach to it, and open a nested child run for each step. `src/jepa_wm/tracking.py` does this and prints the metrics when run outside AI Runtime. Never call `mlflow.set_experiment` inside the run. Use only letters, digits, hyphens, and underscores in run and experiment names.

## Unity Catalog

- A UC Volume does not accept the random writes that `np.save` and parquet writers can make. Write the file to local disk and copy it to the volume.
- AI Runtime has no Spark session. To make a Delta table, write parquet to the volume and run `CREATE OR REPLACE TABLE ... AS SELECT * FROM read_files(...)` through a SQL warehouse. `src/jepa_wm/uc.py` does this.
- Read package versions from the imported modules when you pin requirements for a registered model. The image installs `mlflow` under another distribution name, so `importlib.metadata.version("mlflow")` fails. Remove local build tags such as `+cu130` from the torch version.
- Package encoder weights as an artifact of the registered model in the user's own workspace. The endpoint then does not need to reach the Hugging Face Hub.

## Bundles

`databricks air convert-to-dabs` turns a workload file into a job. The code must be in a subfolder of the bundle. The bundle must not be in a gitignored folder, because packaging skips ignored files and then finds no code. `scripts/make_bundle.sh` writes the bundle outside the repo for that reason.

## Serving

A plain pyfunc model on `GPU_SMALL` (one T4) served the 300 million parameter encoder in the reference build and was ready in 12 minutes. The sensor model runs on CPU. Create endpoints with scale to zero for demos, and expect a slow first request.
