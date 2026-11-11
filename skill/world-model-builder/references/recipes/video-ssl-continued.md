# Level 4: continued pretraining on domain video

This level updates the V-JEPA 2 encoder itself on the user's unlabeled video. The reference build documents it and does not run it.

## When it is justified

- Level 1 was measured and missed the target.
- The video looks unlike natural video, e.g., thermal, microscope, or X-ray footage.
- There are many hours of unlabeled video and a budget for multi-GPU training.

## How to approach it

1. Start from the training code in `facebookresearch/vjepa2`, which is MIT licensed. Keep the three Apache 2.0 files and their notices.
2. Use `GPU_8xH100` on AI Runtime. Multi-GPU training goes through the CLI. Check the current distributed training API in the Databricks docs before you write the launcher, because the package was renamed.
3. Run a short probe first and record the step time. Estimate the full cost from it and get the user's approval before the full run.
4. Save checkpoints to the UC Volume during the run. A CLI job can run for up to 14 days.
5. Evaluate with the level 1 probe on the same splits. Continued pretraining is only worth keeping if the probe improves on dev.

## Risks

- Further training on a small dataset can make the encoder worse on everything else.
- The cost is far higher than levels 1 to 3. State the estimate in the task card.
