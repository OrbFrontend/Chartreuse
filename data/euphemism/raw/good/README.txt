Plain label-0 SEED sentences — ALREADY sentence-level separated, one sentence per line.

These are neutral negatives that generate_synthetic.py rewrites into euphemistic pairs,
so they must be plain (not literary) AND pre-segmented. This folder does NOT split
text for you.

Got whole stories / blog posts / essays instead? Don't put them here. Drop them in
data/raw/docs/ and run: python -m datagen.import_text  (it segments + labels 0).
