# Euphemism Axis Corpus

Mirror of `data/purple/` for the **euphemism** classifier axis (`DEPURPLE_AXIS=euphemism`).
The ablation path is implemented; the remaining gap is first-class synthetic pair generation.

```
data/euphemism/
  raw/good/*.txt            real direct prose seeds (the anchor/target "good" side)
  interim/euphemism_pairs.jsonl   direct seed + gemma direct paraphrase + gemma euphemized rewrite
  interim/*.jsonl           other generator outputs (mined, hard negatives, ...)
  curated.jsonl             committed hand-authored labeled rows
  human_labeled.jsonl       human-labeled calibration/test rows
  labeled.jsonl             build_dataset output (gitignored)
  splits/*.jsonl            train/val/calibration/test (gitignored)
```

`depurple/direction.py` expects each `euphemism_pairs.jsonl` row to include `direct`,
`direct_paraphrase`, `euphemistic`, and `group_id`. The direction uses
`direct_paraphrase -> euphemistic` so both sides are same-source; `direct` is retained for
classifier rows and provenance diagnostics.

Paths resolve here automatically via `core/paths.py` and `depurple/_model.py` when
`DEPURPLE_AXIS=euphemism` is set. See [../../deeuphemism-plan.md](../../deeuphemism-plan.md)
for the current runbook and remaining generator/build-script work.
