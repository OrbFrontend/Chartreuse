# purple_para — paragraph-level purple

Purple at the paragraph level is a matter of density, not of any one sentence. Modern models
(Haiku 5.5 here) write paragraphs where each sentence passes a sentence classifier and the excess
only shows in aggregate: micro-beats (`A pause.`, `You breathe.`), tags (`Just that. Nothing
more.`), atmosphere and significance-signalling in place of events, fragment rhythms. The sentence
head cannot see this. On these paragraphs it scores AUC 0.60 on natural replies, the same as word
count, and 0.40 on camouflaged-vs-terse: it rates clipped hardboiled prose as more purple than
camouflaged filler. So this axis has its own unit. One input is one paragraph (no blank line
inside; a single newline is fine). It is a classifier axis only. It has no matched pairs, so it
is not a depurple axis.

## Sources

| source | rows | unit | where it goes |
|---|---|---|---|
| `interim/haiku_paragraphs.jsonl` | 12,345 | paragraph | train / val / calibration / test, following the source split |
| `interim/haiku_grades.jsonl` | votes | — | rubric-v2 regrade of the rows above, joined back in by text |
| `interim/human_graded.jsonl` | graded-clean of 2,400 | paragraph | 70% of doc groups → train/val; 30% → calibration/test |
| `interim/aug_graded[_r2].jsonl` | 1,970 + 1,972 | paragraph | train / val, following the base paragraph's group |
| `../purple/` (curated, pairs, triaged, common negatives) | ~4,900 | sentence | train / val |
| `../purple/human_labeled.jsonl` | 193 | sentence | calibration / test |
| `curated.jsonl`, `human_labeled.jsonl` here | — | paragraph | same rules as the sentence axes, if you add them |

- **`haiku_paragraphs`** comes from `datagen.import_paragraphs`. It reads the read-only copy of Orb's
  `purple_paragraphs` (`PURPLE_PARAGRAPHS_SRC`, default `../OrbRewriter/data/external/purple_paragraphs`),
  after checking its SHA256SUMS. Each paragraph is graded 0–3 and the label is score ≥ 2, the majority
  of blind Haiku audits.
  - **Dropped:** about 690 non-English rows (nearly all label 0: a free shortcut) and 32 refusals
    labelled 0.
  - **Collapsed:** blank lines inside a row become one newline (87 synthetic rows set dialogue off
    with `\n\n`).
- **Splits.** The source's group-based split is kept.
  - **test** is the source test split, so it stays OrbRewriter's held-out `modern_slop` eval.
  - **calibration** is half of the source-val groups, unanimous-audit rows only, so the 0.90
    precision gate is set on clean labels.
  - Split-audit rows in those groups are dropped, so no group straddles val and calibration.
- **Sentence rows** are the purple axis's rows, used as one-sentence paragraphs. RP is full of
  one-line paragraphs, and the Haiku set has none under 20 words. Their labels carry over unchanged:
  a lone ornate sentence is a saturated paragraph. Note the asymmetry the model must learn: one
  ornate sentence *inside* an otherwise lean paragraph is label 0 (the `isolated` variant).
- **`human_graded`** comes from `datagen.grade_paragraphs`: human paragraphs (`human_writing.txt`,
  OrbRewriter's books / fanfic / bluemoon) graded with the same rubric, the same blind batch prompt
  and the same grader, over 3 passes, majority side.
  - **Negatives only.** Graded-clean rows enter as label 0. Graded-purple rows are left out of
    every split.
  - **Why it's built that way:** under v1, Haiku graded 17–28% of human paragraphs purple (books
    28%), including McCarthy and Roy at [2,2,2]. The axis targets LLM slop, so it must not learn
    that dense literary prose is a positive. Forcing every human row to 0 would mislabel genuinely
    purple fanfic.
  - **What test measures:** the `human:*` slices measure the false-positive rate on graded-clean
    human prose.

- **`aug_graded[_r2]`** comes from `datagen.augment_filler`, graded by `grade_paragraphs --set aug
  --round N`. Clean narrative train paragraphs (Haiku replies, native/real rows and graded-clean human
  paragraphs from non-held-out groups) get matched copies in their own group:
  - **filler copy:** k = 1-3 real Haiku filler or negation sentences inserted at sentence breaks;
  - **concrete control:** real Haiku concrete action sentences in the same k slots.

  Every inserted sentence is mined from the train split only and fits the paragraph's tense and
  pronouns. Round 1 drew beats uniformly. Round 2 draws them by train-split frequency, so the
  formulaic ones (`A pause.`, `Just that.`) are in. Each round has 986 bases and about 1,970 items.
  Neither copy is labelled by construction: both are graded blind and take the majority side.

  | arm (round 1) | k=1 purple | k=2 | k=3 |
  |---|---|---|---|
  | filler / negation | 27% | 83% | 95% |
  | concrete control | 9% | 15% | 26% |

  The control is what stops the head learning "an out-of-place sentence = purple".

## Rubric

`rubric.txt` is **v2**. v1 (`rubric_v1.txt`) is the source dataset's RUBRIC, verbatim, and it is
about ornament: filler-bearing lean paragraphs got a 1, which is label 0. v2 splits the signals
into ORNAMENT and FILLER. It names:
- **filler beats:** breath, pause, shift, tag, comment;
- **negation spam:** what characters don't do, what doesn't happen, "It isn't X. It's Y.",
  "No door. No window.";
- **atmospheric staccato:** contrasted explicitly with terse prose, where each fragment carries a
  fact.

v2 also exempts a negation that *is* the event, and moves the line: two or more filler/negation
beats, or a quarter of the sentences carrying no event, is a 2.

Grades are keyed by the rubric's hash (`r<sha8>`), so editing `rubric.txt` starts a fresh vote and
`build_dataset` refuses grades made under another rubric. A regrade relabels the natural rows
(reply / native / real) by majority, ties dropped. Synthetic variants keep their by-construction
label; the votes only re-score `audit_agree`, which gates calibration.

`interim/` is gitignored: it holds app.db-derived RP text and copyrighted excerpts. Rebuild it
with the commands below.

## Build

```bash
export DEPURPLE_AXIS=purple_para ETTIN_MODEL=jhu-clsp/ettin-encoder-400m
python -m datagen.import_paragraphs                     # -> interim/haiku_paragraphs.jsonl
python -m datagen.grade_paragraphs --per-source 600 \
    --src human_writing.txt --src ../OrbRewriter/data/corpus/paragraphs_books.jsonl \
    --src ../OrbRewriter/data/corpus/paragraphs_fanfic.jsonl \
    --src ../OrbRewriter/data/corpus/paragraphs_bluemoon.jsonl   # claude -p Haiku, ~$0.60, resumable
python -m datagen.grade_paragraphs --set haiku          # regrade under rubric.txt, ~3,090 batches, ~$3
python -m datagen.augment_filler --round 1 && python -m datagen.augment_filler --round 2
python -m datagen.grade_paragraphs --set aug --round 1  # ~490 batches, ~$0.60 per round
python -m datagen.grade_paragraphs --set aug --round 2
./scripts/train_classifier.sh                           # build_dataset -> train (512 tok, batch 16, seed 42) -> calibrate -> evaluate
```

One seed only (42, `train.py`'s default). The multi-seed ensemble was retired 2026-11-07.

`evaluate` prints a per-variant breakdown (reply / camouflaged / terse / human:books / sentence
sources ...). Read the `human:*` rows: they are the false-positive rate on real prose.

## Known gaps

- **Filler recall is below the grader's:** the ensemble flags 70-76% of paragraphs with two
  injected beats, where the grader says 89-99%.
- **Seed variance is large.** One recipe at three seeds flags 46%, 72% and 55% of the filler probe,
  and test AUC runs 0.934-0.957. Judge a change on more than one seed.
- **Multi-paragraph text is out of distribution.** The head trained on single paragraphs. Three
  unrelated clean paragraphs joined score 0.33, against 0.21 alone (`classifier/reward.py`'s chunking
  brings that to 0.28). One slop paragraph in a long document is diluted by the word-weighted mean.
- **v2 relabels only the natural rows.** The synthetic `plain`/`vivid` variants were written
  before v2 and may hold a negation beat or two while staying label 0. Their v2 `audit_agree` shows
  how many graders now disagree; they still train, but only agreeing rows calibrate.
- **The labels are LLM grades.** Calibration and test are Haiku majority votes, not human labels.
  Blind passes agree on the side 86% of the time, which is the ceiling on measured precision.

## Production model, 2026-11-07 (the GRPO reward)

**Superseded, see "Promotion, 2026-11-07 (late)" below.** Production is now one model,
`models/ettin400m-purple_para`, trained with no synthetic negatives. The table here is the retired v2 3-seed
ensemble (`models/retired/ettin400m-purple_para-v2-*`), kept for the history.

```python
from classifier.reward import ParagraphPurple
pp = ParagraphPurple("models/ettin400m-purple_para")
pp.score(texts)   # P(purple) per text
```

Never score it with a sentence-level scorer: that feeds it single sentences at 128 tokens, where
filler is invisible. It trains on 18,837 rows (both augmentation rounds, every graded row).

Test split (1,442 rows, 303 purple), at the threshold calibrated to P ≥ 0.90:

| | threshold | test AUC / P / R | reply P / R | human flagged | +2 filler | +2 negation | +2 concrete |
|---|---|---|---|---|---|---|---|
| **3-seed mean** | 0.507 | **0.960 / 0.824 / 0.832** | 0.745 / 0.727 | 2.0% | **70%** | **76%** | 3.3% |
| seed 42 alone | 0.571 | 0.957 / 0.844 / 0.785 | 0.787 / 0.673 | 2.0% | 46% | 66% | 2.0% |

- The ensemble's 0.507 was calibrated on the calibration split (P 0.902, R 0.898). The per-seed
  `threshold.json` files hold each seed's own threshold. A reward reads the probability, not the
  threshold.
- **Human flagged** is held-out graded-clean human paragraphs (n = 296).
- The **+2** columns are 150 clean test replies with two beats inserted. The grader calls 89%, 99% and
  63% of them purple. Clean replies alone are flagged 0.7%.
- Test precision sits under 0.90 because calibration holds only rows all three graders agreed on,
  while test also has split-vote rows.

Edits a policy could use to lower the reward:

| edit | P(purple), 3-seed mean, test paragraphs |
|---|---|
| strip narration commas | purple 0.797 → 0.777 (the sentence head paid for this) |
| strip -ly adverbs | 0.797 → 0.792 |
| one sentence per paragraph | 0.74 → 0.35 if scored per paragraph; **`reward.py` chunking: 0.74 → 0.73** |
| slop past 512 tokens | unseen when truncated; `reward.py` windows long chunks |
| length | clean r = −0.05; purple r = +0.34 (cutting a purple paragraph lowers P) |

## History, 2026-11-07

Each step was measured against the step before it on the same test split and probe:

| step | test AUC / P / R | human flagged | +2 filler | +2 negation |
|---|---|---|---|---|
| sentence head on paragraphs | 0.66 (reply 0.60, word count's level) | | | |
| rubric v1 labels | 0.933 / 0.749 / 0.789 | 4.1% | 13% | |
| rubric v2 (filler, negation spam) | 0.920 / 0.809 / 0.782 | 1.7% | 5% | 5% |
| + augmentation round 1 (probe strings held out) | 0.940 / 0.836 / 0.809 | 2.7% | 32% | 55% |
| + round 2 (formulaic beats in) | 0.943 / 0.812 / 0.812 | 2.7% | 68% | 75% |
| production, 3-seed mean | 0.960 / 0.824 / 0.832 | 2.0% | 70% | 76% |

- **Rubric v2 alone did not teach filler.** Its filler positives were almost all the one-template
  `camouflaged` paragraphs. Augmentation is what moved the probe.
- **The transfer check:** round 2 was trained without the shift and negative-definition families
  inserted. It still flagged 75% and 45% of paragraphs carrying them, against v2's 35% and 9%. So the
  head learned the shape, not the strings.

## Haiku house tics, 2026-11-07 (candidate, not yet production)

**The gap.** The grader is Haiku grading Haiku. Under rubric v2, one flourish scores 1 = label 0, so 76% of
natural Haiku replies were clean, stock moves included. `He taps the dead dashboard with two knuckles, gently,
as though it might still be persuaded.` was a train row, votes [1,1,0]; the production ensemble scores it 0.155.

**The rule.** `datagen/tics.py` holds 15 tics (look chains, `for half a heartbeat`, `tilts her head`, `her
voice drops`, `something in her X softens`, `lets the question hang`, `a little too wide`, `Somewhere
behind...`, `as though it might...`, ...). Each was kept only when far commoner in Haiku's natural rows than in
graded-clean human prose, and its matches read as slop. Together: 16.4% of Haiku paragraphs, 0.43% of human
ones. `build_dataset` turns any Haiku-generated label-0 row carrying one into label 1 (`tics` field,
`audit_agree` set), aug copies included; human rows are never touched. 1,483 rows flip (99 in test: test
purple 303 -> 402). Pre-relabel splits: `backup_pre_tics_2026-11-07/`.

**Measured** (3-seed means, each at its own calibration threshold: old 0.510, new 0.447). The probe is 94
clean tic-free test replies with hand-written tic / concrete sentences that occur nowhere in the data:

| | production | `-tics` |
|---|---|---|
| test AUC, old / new labels | 0.961 / 0.892 | 0.929 / 0.976 |
| test tic-flipped rows flagged (n = 99) | 14% | 94% |
| human test flagged (n = 296) | 2.0% | 2.4% |
| terse / plain clean flagged (n = 70) | 0% | 0% |
| probe: base / +1 concrete / +2 concrete | 3.2 / 4.3 / 11.7% | 9.6 / 8.5 / 16.0% |
| probe: +1 tic / +2 tic | 23 / 70% | **90 / 100%** |
| probe: +2 filler / +2 negation | 99 / 94% | 100 / 99% |

- **Lexical tics are learned completely (82-100% per tic); the `as though` forms are not (32-52%).**
  `adverb_as_though` and `as_though_mind` match few rows (0.2% / 0.6% of Haiku), and ~300 other Haiku
  `as though/as if` rows stay label 0. Humans use the construction too (2.5% of rows vs Haiku's 8.1%), so
  widening it would teach "as though = purple". The dashboard sentence: 0.238 alone, 0.170 in a plain
  paragraph.
- 100% on a phrase means the head keys on the phrase: a policy can paraphrase around it (`cocks her head`).

## Promotion, 2026-11-07 (late)

**Production: `models/ettin400m-purple_para`**, seed 42 trained on the current splits with no synthetic
negatives (threshold 0.060, calibration P 0.902 / R 0.991). **Telling LLM text from human text is an
intended target, not a side effect:** the head scores provenance and slop together. The tic-relabel seed 42
it replaced (threshold 0.384, trained on `backup_tics_2026-11-07/`) and every older model are in
`models/retired/`.

**Rule now in `build_dataset`: synthetic text is never a negative.** A Haiku, augmented, appdb (LLM draft)
or generated row is label 1 (graded, constructed or tic) or it is dropped. That drops 10,241 negatives;
every remaining negative is human-written. The splits on disk are built this way. A single seed trained
on them is production. Against the tic model it replaced, it trades within-LLM discrimination for provenance:

| | tic model (retired) | production (no synthetic negatives) |
|---|---|---|
| threshold | 0.384 | 0.060 |
| test AUC, current splits (human negatives only) | 0.983 | 0.998 |
| human test flagged (n = 296) | 1.7% | 6.4% |
| clean-graded Haiku replies flagged (n = 527) | 7.4% | 96.8% |
| terse / plain synthetic flagged | 1.4% | 98.6% |
| probe: clean base / +1 concrete / +1 tic, mean P | 0.159 / 0.165 / 0.833 | 0.849 / 0.885 / 0.912 |
| probe +1 as-though flagged | 31% | 100% |
| plain paragraph with / without the dashboard sentence | 0.269 / 0.122 | 0.912 / 0.911 |

As a reward, it gives any LLM paragraph ~0.85-0.91, slop or not: the gradient points toward human-like
prose. A tic still adds a little (+1 tic 0.912 vs base 0.849), but the spread among LLM paragraphs is narrow.
The threshold (0.060) sits low because calibration negatives are all human; read the probability.
