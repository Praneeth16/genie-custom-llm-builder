# Genie Code CLI issues found during this build

Genie Code CLI 0.3.0 (macOS arm64), model provider Unity Gateway, workspace `fe-vm-lakebase-praneeth`.

## 1. `apply_patch` fails when one patch has several hunks for the same file
- What happens: the agent's `apply_patch` tool call is rejected with `apply_patch verification failed: invalid patch: multiple operations target <file>`. The agent recovers by retrying with separate patches, but each failure costs a turn and tokens.
- Seen in: session A2 (2 times, `scripts/task_adapter.py`), session B (1 time). See `sessions/A2.stderr`, `sessions/B.stderr`.
- Expected: either the tool accepts several hunks for one file, or the tool description tells the model to group hunks under one file operation.

## 2. `--sandbox` and `--approve-for-me` cannot be combined, and the help text does not say so
- What happens: `genie exec -s workspace-write --approve-for-me ...` exits with `error: the argument '--sandbox <SANDBOX_MODE>' cannot be used with '--approve-for-me'`.
- The help text for `--approve-for-me` says it routes approvals "using the workspace-write sandbox", which suggests the two flags fit together.
- Expected: say in the help text that `--approve-for-me` sets the sandbox itself and cannot be combined with `-s`.

## 3. `genie exec resume` does not accept the parent `exec` options after the subcommand
- What happens: `genie exec resume <id> --approve-for-me` and `-C` are not listed for `resume`. They only work when placed before it: `genie exec --approve-for-me -C <dir> --json resume <id> -`.
- Expected: document this order in `genie exec resume --help`, or accept the options in either position.

## 4. Local CLI session IDs are not visible in the workspace
- What happens: `genie exec` sessions (for example `01a1a310-330d-7cf3-9746-33e5f403c213`) are stored only under `~/.genie/sessions`. There is no workspace view to audit them, which makes "submit your session ID" hard to verify for CLI work. Another user asked the same question in #how-i-genie.
- Expected: a way to share or export a CLI session (for example a link or an upload to the workspace).
