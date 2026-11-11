# Evaluation protocol

## Splits

- Split by episode for robot and video data, and by time for sensor data. Frames next to each other are nearly identical, so a split by frame puts the same content in training and evaluation.
- Use three splits. Train on the first. Choose settings and thresholds on dev. Score test once, after the settings are frozen.

## Baselines that every report must include

| Task | Baseline that needs no trained network |
|---|---|
| Anomaly detection on sensor windows | Mahalanobis distance of each raw patch from normal data |
| Reading a value from a frame | Ridge regression on downsampled pixels, and predicting the training mean |
| Forecasting latents | "Nothing changes", which repeats the last latents |
| Planning | Random actions, the mean action, and holding the current position |

Give a learned model's competitor the same options the model has. In the reference build, the reconstruction model was scored with the same density score as the JEPA.

## Checks that caught mistakes in the reference build

- **Score an untrained network.** A density score on the latents of a network with random weights reached 0.92 AUROC. Most of the score came from the distance measure, not from training.
- **Shuffle actions with distant rows.** Swapping actions between neighboring frames changes almost nothing, because neighboring frames have nearly the same action. Swap with rows far away.
- **Decode imagined latents.** Latent error is hard to read. Apply the frozen probe to imagined latents and report the error in physical units.
- **Run more than one seed** before comparing two learned models. Report the mean and the spread.

## What to report

Report every method on the same split with the same metric. Include the methods that lost. State the hardware, the run IDs, the step time, and the peak GPU memory. State which results are dev and which are test.
