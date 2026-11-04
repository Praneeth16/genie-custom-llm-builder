# Post draft (LinkedIn / X). Not posted.

I gave Genie Code one job: act as an AI engineer and build a small model that beats a frontier model on a real task.

The task: pull adverse drug events out of medical case reports (the PHEE dataset). The model: Qwen3-4B. The compute: one H100 on Databricks AI Runtime.

I first wrote a skill for Genie Code that makes it work like an engineer, not a script runner. It measures baselines first. It computes signals that say which training method fits (prompting, fine-tuning, distillation or RL). It runs a short probe before every long run. It compares methods at the same compute. And it only touches the test set once, at the end.

Results on the held-out test set:
- Claude Sonnet 5.5, 5-shot: 0.708 F1
- Qwen3-4B base: 0.304
- Qwen3-4B after LoRA fine-tuning: 0.786, with 99.7% valid JSON, at about 28x the speed

The part I did not expect: RL did not help. With the same 40 extra GPU minutes, GRPO held the score (0.782) while more fine-tuning overfit (0.766). For extraction with good labels, fine-tuning did the work. RL is the safer use of spare compute, but it was not needed.

The most useful finding was a bug. The Unsloth copy of the Qwen model uses a chat template that adds an empty "think" block to every answer. The fine-tuned model learned it. With greedy decoding the outputs looked perfect. With sampling, which RL depends on, 99% of outputs broke. If I had trusted greedy evals, I would have concluded that RL is useless for this task. The skill now checks for it.

The model is in Unity Catalog with a weekly batch job. The skill works for any extraction or classification task you can score.

#Databricks #GenieCode #LLM #FineTuning #ReinforcementLearning
