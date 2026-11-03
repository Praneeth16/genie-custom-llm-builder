# GEPA Baseline

Use GEPA when prompting is competitive, data is small, or training is not justified. Delegate API details to `databricks-mlflow-evaluation`; read its required gotchas and interfaces first.

Optimize on training and development data only. Freeze the resulting prompt before the single test evaluation. Count optimization calls, token cost, latency, and the model used. GEPA evidence is not extraction-specific, so compare it rather than assuming it wins.
