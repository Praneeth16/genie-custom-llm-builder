Review of the data card found a label-noise problem that we must fix before any training.

Dropping 921 incompatible events while keeping their sentences leaves 810 sentences whose gold `events` list is empty even though the source has real events. SFT would teach the model to output nothing for real adverse events, and the reward would pay it for doing so. The cause is the schema (single required string for treatment and effect), not your mapping.

Change the schema so PHEE events are representable without dropping:
- `trigger`: string, verbatim (unchanged).
- `subject`, `treatment`, `effect`: each a list of verbatim strings. Multiple entities and discontinuous pieces become separate list items in source order. An empty list means the role is absent. Remove the `null` subject convention.
- `event_type`: unchanged.

Update everything that depends on the schema, in one consistent pass:
1. `task_adapter.py` schema and prompt (show one short example of the list form in the instructions).
2. `prepare_data.py` mapping: no event is dropped for missing, multiple or discontinuous roles. Only drop an event if a span is truly not verbatim in the sentence, and count those. Keep the reason table, now as "represented as list" vs "dropped".
3. `rewards.py` and `metrics.py`: verbatim check applies to every list item. Per-event score = mean over trigger, treatment and effect of token F1 between the predicted and gold spans (bag of tokens over the joined items; both empty = 1.0, one empty = 0.0), with the small exact-match bonus when the span sets match exactly. Subject stays out of the primary score but is reported as a secondary metric. Keep the empty-vs-empty = 1.0 and empty-vs-gold = 0.0 sentence rules, and the one-to-one matching with precision and recall.
4. PHEE-style secondary metrics: main-argument exact match = exact set match of spans per role; token F1 as above.
5. Tests for the new cases: multiple treatments, discontinuous effect, missing effect (empty list), a list item that is not verbatim, duplicate list items.
6. Re-run data prep, re-upload the JSONL files and the data card to the same volume path (overwrite), refresh `sample-20.jsonl`, update the task card mapping and metric definitions, and redo the 5-row hand check including at least one row with a multi-span role and one with a missing effect.

Run the full test suite and make it pass. No AI Runtime runs. Reply with the new reason table, the new counts of sentences with empty gold per split, the test result line, and anything you could not fix.
