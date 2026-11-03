# PHEE V2 data card

## Source

- Dataset: PHEE V2.
- Repository: https://github.com/ZhaoyueSun/phee-with-chatgpt
- License: MIT.
- Source commit: `a90c680b08464e731635eb2d2f2c49be6ad5e3e5`.
- Real source layout: `data/datasets/phee2/source/{train,dev,test}.json`. The files use JSON Lines despite the `.json` suffix.

## Counts

| Split | Source sentences | Converted sentences | Source events | Retained events | Dropped events | Sentences with no retained event |
|---|---:|---:|---:|---:|---:|---:|
| Train | 2,898 | 2,898 | 3,004 | 2,473 | 531 | 472 |
| Dev | 961 | 961 | 1,003 | 808 | 195 | 173 |
| Test | 968 | 968 | 1,010 | 815 | 195 | 165 |
| Total | 4,827 | 4,827 | 5,017 | 4,096 | 921 | 810 |

The sentence counts exactly match the published split sizes. Sentences are never dropped. A sentence may have an empty `events` list after incompatible source events are removed.

## Mapping

- `event_type`: preserve `Adverse_event` or `Potential_therapeutic_event`.
- `trigger`: one contiguous verbatim value from `Trigger.text`.
- `subject`: one contiguous verbatim value from `Subject.text`; `null` when absent. An incompatible subject is mapped to `null` and counted.
- `treatment`: one contiguous verbatim value from `Treatment.text`.
- `effect`: one contiguous verbatim value from `Effect.text`.
- Events with a missing, multiple, discontinuous, or non-verbatim required field are removed. Values are not inferred, merged, or silently shortened.

## Incompatible gold

Reason counts can overlap because one dropped event can have more than one problem.

| Reason | Train | Dev | Test | Total |
|---|---:|---:|---:|---:|
| Missing effect | 223 | 94 | 107 | 424 |
| Missing treatment | 1 | 0 | 0 | 1 |
| Multiple treatment entities | 10 | 2 | 2 | 14 |
| Multiple effect entities | 25 | 11 | 8 | 44 |
| Discontinuous treatment | 170 | 61 | 54 | 285 |
| Discontinuous effect | 140 | 36 | 38 | 214 |
| Non-verbatim effect | 0 | 0 | 1 | 1 |
| Dropped events | 531 | 195 | 195 | 921 |

Subject-only incompatibilities do not require dropping an event because the output schema permits `null`.

| Subject handling | Train | Dev | Test | Total |
|---|---:|---:|---:|---:|
| Discontinuous subject | 45 | 9 | 16 | 70 |
| Multiple subject entities | 3 | 1 | 2 | 6 |
| Subjects mapped to `null` | 39 | 8 | 14 | 61 |

## Split overlap

- Normalized exact sentence overlap: train/dev 0, train/test 0, dev/test 0.
- Near-duplicate definition: cross-split token-set Jaccard similarity at least 0.90, excluding normalized exact matches.
- Near-duplicate pairs: train/dev 0, train/test 0, dev/test 0.

## Hand check

Five converted dev rows were compared with their source records on 2026-11-04.

| Source ID | Result |
|---|---|
| `10907391_3` | Pass. Trigger, treatment, and effect match the source; absent subject remains `null`. |
| `18957000_3` | Pass. Hyphenated source sentence is preserved and all mapped spans are verbatim. |
| `6405633_1` | Pass. The source event has no `Effect`, so it is reported as incompatible and the sentence is retained with an empty event list. |
| `4044222_1` | Pass. Subject, trigger, treatment, and effect match the source exactly. |
| `12962465_2` | Pass. Multiword trigger and treatment match the source exactly. |

Result: 5 of 5 conversions agree with the stated mapping.

## Files

- `train.jsonl`, `dev.jsonl`, and `test.jsonl`: full converted splits.
- `sample-20.jsonl`: first 20 converted dev rows for review.
- `data_card.json`: machine-readable counts and overlap report.
