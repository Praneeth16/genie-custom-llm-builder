# Level selection

Choose the lowest level that can meet the target. A higher level costs more compute and has more ways to fail.

| Level | Use it when | Trains | Cost in the reference build | Recipe |
|---|---|---|---|---|
| 1. Frozen encoder and probe | The data is images or video, and the task is to read something from a frame, e.g., a position or a class. Some labels exist. | A small head on frozen V-JEPA 2 latents | 4 minutes to encode 25,650 frames and 16 seconds to train the probe, on one A10 | [frozen probe](recipes/frozen-probe.md) |
| 2. Latent predictor | Actions are recorded with the frames, and the task needs a forecast of what an action will do. | A small transformer on cached latents | 15 minutes on one A10 | [latent predictor](recipes/latent-predictor.md) |
| 3. JEPA from scratch | The data is not images, e.g., multivariate sensor windows, so no pretrained encoder fits. | A small encoder and predictor with SIGReg | About 90 seconds per seed on one A10, including the baselines | [sensor JEPA](recipes/sensor-jepa-sigreg.md) |
| 4. Continued pretraining on video | Level 1 fails because the video looks unlike the pretraining data, and there are many hours of unlabeled video. | The V-JEPA 2 encoder itself | Multi-GPU, hours to days. Not run in the reference build. | [continued pretraining](recipes/video-ssl-continued.md) |

## Signals to measure before choosing

- **Labels.** Count the labeled examples. Level 1 needs some. Level 3 can train without labels but still needs labels to evaluate.
- **Actions.** Check that an action is recorded for every frame and that you know which frame each action leads to. Without actions, level 2 is a video forecaster and cannot be used to plan.
- **Baseline gap.** Measure how far the baseline is from the target. If the gap is small, a higher level is unlikely to be worth its cost.
- **Frame noise.** For level 2, measure the error of "nothing changes" one step ahead in normalized latents. In the reference build it was 0.69 of the latent variance, which limits how well short plans can be scored.

No source gives a data-size threshold for moving between levels. Decide from the dev metric.
