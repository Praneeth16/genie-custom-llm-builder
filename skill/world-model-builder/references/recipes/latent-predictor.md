# Level 2: action-conditioned latent predictor

Train a small transformer that takes the latents of the last frames and an action, and returns the latents of the next frame. The encoder stays frozen, so training runs on cached latents and is cheap.

## Steps

1. Run level 1 first. The predictor needs the cached latents, and the probe is used to read imagined latents.
2. Check the alignment of actions and frames. In `train/train_predictor.py` the action recorded at frame t leads to frame t+1. A shift of one step is easy to miss and makes the actions useless.
3. Train with `workloads/world_model.yaml`. The loss rolls the predictor forward three steps, so the model learns to stay stable when it is fed its own output.
4. Evaluate with `--probe` so that imagined latents are decoded to physical units.

## Design choices that the reference build needed

- **Add the action to every token.** The first version gave the action one token among 129. The predictor ignored it and forecast motion from the two context frames alone. Adding the action embedding to every token fixed this.
- **Predict the change.** The output layer starts at zero, so an untrained predictor returns the last latents. Training then learns only what changes.
- **Normalize the latents** with the mean and spread of the training latents. The loss and the reported errors are in these units.

## Checks

- **No change.** The predictor must beat repeating the last latents at every horizon.
- **What the action already states.** If the action names the quantity you decode, e.g., a target position, use it as a baseline. In the reference build the predictor's test error for the agent's position was 13.3 px one step ahead and 22.6 px eight steps ahead. The guess "the agent is where the last action sent it" had 14.8 px and 14.2 px. The predictor won only one step ahead. Test a world model on something the action does not state, e.g., the pose of an object the agent pushes.
- **Distant actions.** Roll out with actions taken from rows far away in the batch. The decoded error should rise sharply. If it does not, the predictor is not using the actions.
- **Planning.** `src/jepa_wm/planning.py` searches for actions with the cross-entropy method. Compare the plan with random actions, the mean action, and holding position. In the reference build the planner lost to holding position, and a larger search was worse. Over four steps the goal frame differed little from the current frame, and the search found action sequences that exploited the predictor's errors. Do not present planning as working until it is tested in a simulator or on the real system.
