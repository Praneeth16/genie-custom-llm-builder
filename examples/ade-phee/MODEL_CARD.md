# ADE-PHEE Extractor

## Model

- **Unity Catalog:** `serverless_lakebase_praneeth_catalog.custom_llm.ade_extractor` version `1`, alias `champion`.
- **Architecture:** `Qwen/Qwen3-4B-Instruct-2507` with the SFT-full v2 LoRA adapter, loaded through vLLM by an MLflow pyfunc.
- **License:** Qwen is Apache-2.0. PHEE V2 is MIT.

## Intended Use

Research and demonstration pharmacovigilance triage over individual case-report sentences. The model extracts candidate adverse-event and potential-therapeutic-event structures for downstream review. It is not a clinical decision system, diagnostic device, causality assessor, safety-reporting substitute, or autonomous pharmacovigilance workflow.

## Data

PHEE V2 comes from `https://github.com/ZhaoyueSun/phee-with-chatgpt` at source commit `a90c680b08464e731635eb2d2f2c49be6ad5e3e5`. Upstream train/dev/test splits were preserved: 2,898/961/968 converted rows. Conversion preserves verbatim spans, flattens discontinuous role pieces in source order, removes duplicate source spans after their first occurrence, and drops only incompatible events while retaining their sentences. See `data/data_card.md`.

## Schema

The JSON object contains one `events` array. Each event has `event_type` (`Adverse_event` or `Potential_therapeutic_event`), a verbatim string `trigger`, and unique verbatim string arrays `subject`, `treatment`, and `effect`. No additional keys are allowed.

## Metric

The frozen primary metric is gated strict event F1. Invalid JSON, schema violations, duplicate list items, invalid event types, and non-verbatim spans score zero. Valid events are matched one-to-one; trigger, treatment, and effect each combine 0.9 token F1 with 0.1 exact span-set agreement. The dev target was 0.7655.

## Results

| Model | Dev strict F1 | Dev valid rate | Test strict F1 | Test valid rate | Test PHEE EM | Test token F1 | Test rows/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Claude Sonnet 5.5 zero-shot | 0.6209 | 0.9053 | 0.6351 | 0.9473 | 0.4796 | 0.6394 | 2.71 |
| Claude Sonnet 5.5 five-shot | 0.7155 | — | 0.7083 | 0.9855 | 0.5636 | 0.7173 | 2.82 |
| Qwen3-4B five-shot | 0.3007 | — | 0.3037 | 0.5826 | 0.2273 | 0.3102 | 185.78 |
| SFT-full v2 (final) | 0.7817 | 0.9896 | **0.7856** | **0.9969** | **0.6695** | **0.7927** | 79.16 |
| GRPO Arm B (secondary) | 0.7818 | 0.9886 | 0.7838 | 0.9969 | 0.6676 | 0.7909 | 81.98 |

The dev gap between GRPO and SFT-full v2 was 0.0001, below the 0.008 noise reference. The pre-test freeze therefore selected the cheaper SFT model. GRPO remained a secondary test result only.

## Known Errors

Five real SFT-full v2 test errors illustrate recurring boundary and event-separation failures:

1. Test row 0 selected trigger `developed` instead of gold `After`, added `the patient`, and shortened the treatment/effect boundaries (strict F1 0.5348).
2. Test row 3 merged two drug/patient events into one event containing both subjects and treatments (strict F1 0.5778).
3. Test row 5 selected causal-context trigger `taking` instead of the annotated event trigger `developing` (strict F1 0.6667).
4. Test row 21 emitted only one of three annotated events and missed the warfarin and normal-volunteer relations (strict F1 0.1714).
5. Test row 23 narrowed combined pegvisomant/octreotide treatment to pegvisomant and expanded the effect to `Drug-induced hepatitis` (strict F1 0.1745).

## Training

LoRA SFT used 2,898 gold rows, seed 42, rank 32, alpha 32, learning rate `2e-4`, completion-only loss, three epochs, and 1,089 optimizer steps. It ran for 19.26 accelerator minutes on one H100 and peaked at 11.64% H100 memory. The reference Qwen chat template fixed an earlier corrupted-special-token preamble learned from the Unsloth mirror.

The strategy path was: measure frontier/base signals; select SFT because gold labels and a closed schema were available; diagnose and correct the reasoning/chat-template parser bug; pass the corrected sampling gate; compare 40-minute continued-SFT and GRPO arms; observe that GRPO only protected against an over-trained control and improved over SFT-full v2 by 0.0001, below noise; freeze and ship SFT-full v2.

## Runtime

Evaluation and packaging used one H100 with `mlflow==3.13.0`, `torch==2.11.0+cu130`, `transformers==5.8.1`, `vllm==0.24.0`, `peft==0.19.1`, `pandas==2.3.3`, and `databricks-sdk==0.122.0`. The registered pyfunc includes the base model ID, LoRA adapter files, prompt builder, model signature, input example, and pinned requirements.

Batch scoring loads `models:/serverless_lakebase_praneeth_catalog.custom_llm.ade_extractor@champion`. The shipping check wrote 50 rows to `serverless_lakebase_praneeth_catalog.custom_llm.ade_events`; all 50 outputs parsed.
