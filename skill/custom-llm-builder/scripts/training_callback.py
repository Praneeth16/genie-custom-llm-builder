from __future__ import annotations

import math
import time


def build_probe_callback(max_memory_fraction: float = 0.9):
    import mlflow
    import torch
    from transformers import TrainerCallback

    class ProbeCallback(TrainerCallback):
        step_seconds: list[float] = []
        peak_memory_fraction = 0.0

        def on_step_begin(self, args, state, control, **kwargs):
            self.started = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

        def on_step_end(self, args, state, control, **kwargs):
            elapsed = time.perf_counter() - self.started
            self.step_seconds.append(elapsed)
            mlflow.log_metric("probe_seconds_per_step", elapsed, step=state.global_step)
            if torch.cuda.is_available():
                peak = float(torch.cuda.max_memory_allocated())
                total = float(torch.cuda.get_device_properties(0).total_memory)
                fraction = peak / total
                self.peak_memory_fraction = max(self.peak_memory_fraction, fraction)
                mlflow.log_metric("probe_peak_gpu_memory_bytes", peak, step=state.global_step)
                mlflow.log_metric("probe_peak_gpu_memory_fraction", fraction, step=state.global_step)
                if fraction > max_memory_fraction:
                    raise RuntimeError(f"peak GPU memory {fraction:.1%} exceeds {max_memory_fraction:.0%} gate")

        def on_log(self, args, state, control, logs=None, **kwargs):
            loss = (logs or {}).get("loss")
            if loss is not None and not math.isfinite(float(loss)):
                raise RuntimeError(f"non-finite training loss at step {state.global_step}: {loss}")

    return ProbeCallback()
