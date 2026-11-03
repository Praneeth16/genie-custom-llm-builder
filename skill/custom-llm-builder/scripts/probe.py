from __future__ import annotations

import argparse
import json
import subprocess
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="+")
    args = parser.parse_args()
    if not any(part == "--max-steps" or part.startswith("--max-steps=") for part in args.command):
        args.command.extend(["--max-steps", "30"])
    started = time.perf_counter()
    result = subprocess.run(args.command, check=False)
    elapsed = time.perf_counter() - started
    try:
        import torch
        peak_memory = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    except ImportError:
        peak_memory = 0
    print(json.dumps({"returncode": result.returncode, "steps": 30, "elapsed_seconds": elapsed, "seconds_per_step": elapsed / 30, "peak_memory_bytes": peak_memory, "required_rl_metrics": ["completions/clipped_ratio", "frac_reward_zero_std"]}))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
