# Level 3: a JEPA from scratch with SIGReg

Train a small encoder and predictor on windows of multivariate sensor data. No pretrained encoder is needed.

## Method

- Cut each window of 128 time steps into 16 patches of 8 steps. Each patch is one token.
- Hide a block of four tokens. The encoder sees the rest. The predictor returns latents for the hidden tokens.
- The target is the encoder's own latents for the full window. There is no second teacher network and no stopped gradient.
- SIGReg stops the latents from collapsing to a constant. It projects the latents onto random directions and measures how far each projection is from a standard Gaussian with the Epps-Pulley statistic. `src/jepa_wm/sigreg.py` is written from the maths in the LeJEPA paper (arXiv 2511.08544).
- The loss is 0.05 times SIGReg plus 0.95 times the prediction error.

## Facts about SIGReg that are easy to get wrong

- The statistic is scaled by the batch size. For Gaussian latents it is about 1.06, not 0. A collapsed batch gives several hundred.
- The paper's text and its code listing use different weight functions. `sigreg.py` uses exp(-t^2 / 2) and says so.
- Leave out the final LayerNorm in the encoder. It fixes the scale of every token, which works against the Gaussian target.

## Scoring anomalies

A window can be scored in two ways.

- **Density.** Fit a Gaussian to the token latents of normal windows. The score is the largest Mahalanobis distance of any token. This is the better score.
- **Prediction error.** The largest error over the hidden blocks. This is a weak score. A JEPA learns to leave out detail it cannot predict, and a small sensor fault is that kind of detail.

## What the reference build showed

On synthetic telemetry, over three seeds, the test AUROC for local and dynamics anomalies was:

| Method | Local | Dynamics |
|---|---|---|
| Gaussian on raw patches, no network | 0.952 | 0.982 |
| Sensor JEPA, density score | 0.908 | 0.972 |
| Masked reconstruction, density score | 0.869 | 0.961 |
| Sensor JEPA, prediction error | 0.675 | 0.717 |
| Isolation forest | 0.550 | 0.611 |

The Gaussian with no network was best. The JEPA's latents were better than the reconstruction model's latents. This data is close to linear, so a Gaussian fits it well. The order can change on data with several operating modes, which is why you run the comparison on the user's data. If the Gaussian wins there too, ship the Gaussian for detection and use the JEPA only where an embedding is needed.
