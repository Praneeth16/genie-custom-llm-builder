# PHEE V2 data card

## Source and counts

- Repository: https://github.com/ZhaoyueSun/phee-with-chatgpt
- License: MIT.
- Commit: `a90c680b08464e731635eb2d2f2c49be6ad5e3e5`.
- Source layout: `data/datasets/phee2/source/{train,dev,test}.json`, using JSON Lines despite the suffix.

| Split | Sentences | Source events | Retained events | Dropped events | Empty gold sentences |
|---|---:|---:|---:|---:|---:|
| Train | 2,898 | 3,004 | 3,004 | 0 | 0 |
| Dev | 961 | 1,003 | 1,003 | 0 | 0 |
| Test | 968 | 1,010 | 1,009 | 1 | 1 |
| Total | 4,827 | 5,017 | 5,016 | 1 | 1 |

The sentence counts match the published splits. The only empty gold sentence contains the one source event whose effect annotation is not verbatim in the sentence.

## Mapping

- `event_type` preserves `Adverse_event` or `Potential_therapeutic_event`.
- `trigger` is one verbatim string.
- `subject`, `treatment`, and `effect` are lists of verbatim strings.
- Multiple entities and discontinuous pieces become separate list items in source order.
- An absent role becomes an empty list.
- Duplicate source spans are represented once, at their first source position.
- An event is dropped only for a truly non-verbatim span, invalid source shape, unrepresentable trigger, or invalid event type.

## Representation audit

These cases were previously incompatible with the scalar schema. They are now represented as lists and do not cause event removal.

| Represented case | Train | Dev | Test | Total |
|---|---:|---:|---:|---:|
| Missing subject → empty list | 1,560 | 487 | 546 | 2,593 |
| Missing treatment → empty list | 1 | 0 | 0 | 1 |
| Missing effect → empty list | 223 | 94 | 107 | 424 |
| Multiple subject entities → list | 3 | 1 | 2 | 6 |
| Multiple treatment entities → list | 10 | 2 | 2 | 14 |
| Multiple effect entities → list | 25 | 11 | 8 | 44 |
| Discontinuous subject pieces → list | 45 | 9 | 16 | 70 |
| Discontinuous treatment pieces → list | 170 | 61 | 54 | 285 |
| Discontinuous effect pieces → list | 141 | 36 | 38 | 215 |
| Duplicate treatment span → represented once | 4 | 0 | 0 | 4 |

## Dropped gold

Reason counts can overlap if a future event has more than one problem.

| Dropped reason | Train | Dev | Test | Total |
|---|---:|---:|---:|---:|
| Non-verbatim effect | 0 | 0 | 1 | 1 |
| Dropped events | 0 | 0 | 1 | 1 |

## Split overlap

- Normalized exact overlap: train/dev 0, train/test 0, dev/test 0.
- Near duplicates use cross-split token-set Jaccard similarity at least 0.90, excluding exact matches.
- Near-duplicate pairs: train/dev 0, train/test 0, dev/test 0.

## Hand check

Five converted dev rows were compared with their source records on 2026-11-04.

| Source ID | Result |
|---|---|
| `6405633_1` | Pass. The missing source effect is represented as an empty list; the event is retained. |
| `18159131_4` | Pass. The discontinuous subject is represented as two verbatim list items in source order. |
| `10907391_3` | Pass. Trigger, treatment, and effect match; absent subject is an empty list. |
| `4044222_1` | Pass. Subject, trigger, treatment, and effect match exactly. |
| `12962465_2` | Pass. Multiword trigger, treatment, and effect match exactly. |

Result: 5 of 5 conversions agree with the revised mapping.

## Files

- `train.jsonl`, `dev.jsonl`, and `test.jsonl`: full converted splits.
- `sample-20.jsonl`: first 20 regenerated dev rows.
- `data_card.json`: machine-readable counts and audit details.
