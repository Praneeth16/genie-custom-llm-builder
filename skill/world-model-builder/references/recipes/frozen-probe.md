# Level 1: frozen encoder and probe

Use a pretrained V-JEPA 2 encoder without training it. Train only a small head that reads the answer from its latents.

## Steps

1. Encode every frame once and cache the latents in a UC Volume. `train/embed.py` with `workloads/embed.yaml` does this. It repeats each frame to fill one tubelet of two frames, so every frame gets its own latents.
2. Keep the token latents, not only their mean. `embed.py` pools the 16 by 16 grid to 8 by 8 tokens per frame, stored as float16.
3. Write the mean of the tokens to a parquet file for search. `src/jepa_wm/uc.py` turns it into a Delta table, and `scripts/vector_index.py` builds a Delta Sync index with self-managed embeddings.
4. Train the attentive probe with `train/train_probe.py`. One learned query attends over the tokens, and a linear layer gives the answer.
5. Compare with an MLP on the mean latent trained the same way, with ridge regression on the mean latent and on downsampled pixels, and with predicting the training mean.

## What the reference build showed

The task was to read the agent's position in PushT. The attentive probe on token latents had a test error of 10.4 px in a 512 px arena. An MLP on the mean latent with the same training had 20.2 px. Ridge regression on the mean latent had 29.0 px, and ridge regression on pixels had 30.0 px. Keep the tokens when the task depends on where something is.

## Gates

- The 30-step probe run shows peak GPU memory below 90 percent. In the reference build the latents of 25,650 frames used 48 percent of an A10.
- The probe beats every baseline on dev.

## Changing the encoder

`src/jepa_wm/encoders.py` takes any Hugging Face repository that `transformers.AutoModel` loads as a V-JEPA 2 model. A larger encoder needs more memory and a larger GPU endpoint. V-JEPA 2.1 has no official Hugging Face repository. Load it with `torch.hub` from `facebookresearch/vjepa2` and adapt `FrameEncoder`.
