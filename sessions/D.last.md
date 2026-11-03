Blocked before a fourth paid GPU submission.

- Runs `183208434761239`, `576843395797709`, and `365502683517865` failed during setup; no training/test scoring occurred.
- Root cause: AIR image `"6"` lacks torch/vLLM; requested pins don’t install vLLM.
- Patched workload to add documented-safe `vllm==0.15.1`.
- Approval required to submit another H100 run. Shall I proceed?