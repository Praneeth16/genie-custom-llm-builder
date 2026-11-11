"""MLflow model for the frozen V-JEPA 2 frame encoder plus the position probe."""

from __future__ import annotations

import base64
import io

import mlflow
import numpy as np
import pandas as pd
import torch


class FrameEncoderModel(mlflow.pyfunc.PythonModel):
    """Input: an `image` column of base64-encoded JPEG or PNG frames.

    Output: `embedding` (mean of the token latents) and the probe's `agent_x`, `agent_y` in
    [0, 1]. The encoder weights are an artifact of the registered model, so serving does not
    need to reach the Hugging Face Hub. Requests are capped at 16 MB by Model Serving: send
    frames, not whole videos.
    """

    def load_context(self, context):
        from jepa_wm.encoders import FrameEncoder
        from jepa_wm.probe import AttentiveProbe

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = FrameEncoder(context.artifacts["encoder"]).to(self.device)
        self.probe = AttentiveProbe(self.encoder.dim, 2).to(self.device)
        self.probe.load_state_dict(torch.load(context.artifacts["probe"], map_location=self.device))
        self.probe.eval()

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        from PIL import Image

        frames = [np.asarray(Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB")) for b in model_input["image"]]
        with torch.no_grad():
            latents = self.encoder(torch.from_numpy(np.stack(frames)).to(self.device))
            position = self.probe(latents).cpu().numpy()
        return pd.DataFrame(
            {
                "embedding": list(latents.mean(1).cpu().numpy()),
                "agent_x": position[:, 0],
                "agent_y": position[:, 1],
            }
        )
