"""Upload the repo to a workspace folder and run every notebook there as a one-time job.

The setup notebook runs on serverless compute. The other three run on serverless GPU (one A10)
with the AI Runtime environment, which is how they are meant to be opened.

    python scripts/run_notebooks.py            # all four
    python scripts/run_notebooks.py 03         # only notebooks whose name starts with 03
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import compute, jobs

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    profile = os.environ["DATABRICKS_PROFILE"]
    client = WorkspaceClient(profile=profile)
    folder = f"/Workspace/Users/{client.current_user.me().user_name}/jepa-world-models-databricks"

    with tempfile.TemporaryDirectory() as staging:
        subprocess.run(
            ["rsync", "-a", "--exclude", "__pycache__", "notebooks", "src", "train", staging], cwd=ROOT, check=True
        )
        subprocess.run(
            ["databricks", "workspace", "import-dir", staging, folder, "--overwrite", "--profile", profile],
            check=True,
            capture_output=True,
        )

    parameters = {"catalog": os.environ["JEPA_CATALOG"], "schema": os.environ["JEPA_SCHEMA"]}
    names = sorted(p.stem for p in (ROOT / "notebooks").glob("*.py") if p.stem.startswith(tuple(sys.argv[1:]) or ""))
    tasks = [
        jobs.SubmitTask(
            task_key=name,
            notebook_task=jobs.NotebookTask(notebook_path=f"{folder}/notebooks/{name}", base_parameters=parameters),
            depends_on=[jobs.TaskDependency(task_key=names[0])]
            if name != names[0] and names[0].startswith("00")
            else None,
            compute=None
            if name.startswith("00")
            else jobs.Compute(hardware_accelerator=compute.HardwareAcceleratorType.GPU_1X_A10),
            environment_key=None if name.startswith("00") else "ai",
        )
        for name in names
    ]
    environments = [
        jobs.JobEnvironment(
            environment_key="ai",
            spec=compute.Environment(base_environment="workspace-base-environments/databricks_ai_v6"),
        )
    ]
    run = client.jobs.submit(run_name="jepa-wm-notebooks", tasks=tasks, environments=environments)
    print(f"run {run.run_id} submitted; waiting")
    finished = run.result()
    for task in finished.tasks:
        print(f"{task.task_key}: {task.state.result_state.value} ({(task.end_time - task.start_time) // 1000} s)")
    if any(task.state.result_state != jobs.RunResultState.SUCCESS for task in finished.tasks):
        sys.exit(1)


if __name__ == "__main__":
    main()
