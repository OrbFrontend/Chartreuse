"""Print base-vs-edited lm-eval deltas. Reads the latest results_*.json under each
output dir and shows every primary metric per task (mmlu's 57 leaves collapse to the
group row via group_subtasks). A negative delta past the threshold is flagged — that's
the lobotomy signal; small +/- is noise.

    python depurple/bench_compare.py depurple/bench-<slug>/base depurple/bench-<slug>/edited

This prints a compact table only. The delta is the useful signal; absolute scores
depend on task configuration.
"""
import sys, json, glob, os

REGRESS = -0.02   # flag deltas worse than this (≈ beyond quick-run noise)


def load(d):
    files = glob.glob(os.path.join(d, "**", "results_*.json"), recursive=True)
    if not files:
        raise SystemExit(f"no results_*.json under {d} — did the run finish?")
    data = json.load(open(max(files, key=os.path.getmtime)))
    leaves = {s for subs in data.get("group_subtasks", {}).values() for s in subs}
    return data["results"], leaves


SKIP = {"alias", "sample_len"}


def metrics(entry):
    # keep numeric "<name>,<filter>" keys; the metric name is before the comma,
    # so drop stderr/alias/sample_len by that name, not by the full key.
    out = {}
    for k, v in entry.items():
        name = k.split(",")[0]
        if isinstance(v, (int, float)) and not name.endswith("_stderr") and name not in SKIP:
            out[k] = v
    return out


def main(base_dir, edited_dir):
    base, base_leaves = load(base_dir)
    edited, _ = load(edited_dir)
    shown = sorted(set(base) - base_leaves)        # collapse mmlu subjects into the group row
    print(f"{'task / metric':<42}{'base':>9}{'edited':>9}{'delta':>9}")
    print("-" * 69)
    flagged = False
    for task in shown:
        for k, b in metrics(base[task]).items():
            e = edited.get(task, {}).get(k)
            if not isinstance(e, (int, float)):
                continue
            d = e - b
            # flag only a drop that clears BOTH a floor and the combined error bars
            # (stderr key is "<name>_stderr,<filter>"); small n => wide bars => no false alarm.
            name, _, filt = k.partition(",")
            se = base[task].get(f"{name}_stderr,{filt}", 0.0) + \
                edited.get(task, {}).get(f"{name}_stderr,{filt}", 0.0)
            sig = d < REGRESS and -d > se
            flagged |= sig
            flag = "  <-- regression" if sig else ""
            print(f"{task+' '+name:<42}{b:>9.4f}{e:>9.4f}{d:>+9.4f}{flag}")
    print("-" * 69)
    print("REVIEW: a drop exceeds its error bars — edit may have hurt capability"
          if flagged else "OK: no drop beyond noise — no lobotomy")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: bench_compare.py <base_dir> <edited_dir>")
    main(sys.argv[1], sys.argv[2])
