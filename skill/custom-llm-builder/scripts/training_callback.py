from __future__ import annotations

import time


def build_probe_callback():
    import mlflow
    import torch
    from transformers import TrainerCallback

    class ProbeCallback(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            self.started = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

        def on_step_end(self, args, state, control, **kwargs):
            mlflow.log_metric("probe_seconds_per_step", time.perf_counter() - self.started, step=state.global_step)
            if torch.cuda.is_available():
                mlflow.log_metric("probe_peak_gpu_memory_bytes", float(torch.cuda.max_memory_allocated()), step=state.global_step)

    return ProbeCallback()
