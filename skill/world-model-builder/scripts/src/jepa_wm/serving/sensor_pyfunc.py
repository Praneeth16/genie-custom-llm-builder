"""MLflow model for the sensor JEPA: an anomaly score and an embedding per window."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch


class SensorJepaModel(mlflow.pyfunc.PythonModel):
    """Input: a `window` column, each value a (window, channels) nested list of readings.

    Output: `anomaly_score` (distance of the window's latents from normal data) and `embedding`.
    """

    def load_context(self, context):
        from jepa_wm.sensor_jepa import SensorMaskedModel

        config = json.loads(Path(context.artifacts["config"]).read_text())
        self.model = SensorMaskedModel(config["channels"], config["window"], target="latent")
        self.model.load_state_dict(torch.load(context.artifacts["weights"], map_location="cpu"))
        self.model.eval()

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        windows = torch.from_numpy(np.stack([np.asarray(w, dtype=np.float32) for w in model_input["window"]]))
        with torch.no_grad():
            return pd.DataFrame(
                {
                    "anomaly_score": self.model.score(windows).numpy(),
                    "embedding": list(self.model.embed(windows).numpy()),
                }
            )
