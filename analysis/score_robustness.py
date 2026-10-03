"""Robustness of the induced-valuation and reasoning results (Oct 2026).

Scores any set of elicitation CSVs on the metrics the Phase 3 writeup reports, with 95%
cluster-bootstrap intervals (resampling gamble clusters, instrument x gamble_id):
  dom   hard accuracy on the dominant controls
  ev    hard agreement with the rule's optimum on clear core cells (|ln EU ratio| >= margin)
  mass  probability mass on the optimum, same cells
  gap   frame gap, mean P(gamble) in the mixed (loss-worded) frame minus the gain frame

The rule is read from each file's `induce` label (sqrt* -> sqrt, anything else, including
the paraphrases riskneutral_* and uninduced runs, -> risk-neutral EV), or set with --rule.

  --vs REF      contrast each file with REF on the cells both cover (e.g. a paraphrase vs the
                original wording, a full-set run vs its 20-gamble half, or a paid model vs the
                unpaid one), with a paired cluster-bootstrap interval for each difference.
  --drop-cells-of OLD  score only the cells OLD does not cover, e.g. the 40 new gambles of a
                full-set reasoning run, to check them against the first 40.
  --greedy REF  for files with several sampled traces per cell (phase3_elicit.py --samples):
                how often the greedy trace's answer matches the majority of sampled ones.

For a sampled file, ev/mass/gap average over traces (the accuracy of a randomly drawn
trace); the sampling block adds majority-vote accuracy and the share of unanimous cells.

Usage:
    python analysis/score_robustness.py data/phase3_qwen-inst_chat_reason_riskneutral_*.csv \\
        --vs data/phase3_qwen-inst_chat_reason_riskneutral.csv
    python analysis/score_robustness.py data/phase3_qwen-inst_chat_reason_riskneutral_k5.csv \\
        --greedy data/phase3_qwen-inst_chat_reason_riskneutral.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from score_induced import load, score  # noqa: E402

KEY = ["instrument", "trial_id"]
METRICS = ("dom", "ev", "mass", "gap")


def rule_of(df, override):
    if override:
        return override
    label = str(df["induce"].iloc[0]) if "induce" in df and len(df) else "none"
    return "sqrt" if label.startswith("sqrt") else "riskneutral"


def prepare(path, rule_override, margin):
    d = load(path).reset_index(drop=True)
    if d.empty:
        return d, rule_override
    if "sample" not in d:
        d["sample"] = 0
    d["sample"] = d["sample"].fillna(0).astype(int)
    rule = rule_of(d, rule_override)
    sc = score(d, rule, margin)
    d = d.assign(correct=sc["hard_correct"].values, clear=sc["clear"].values,
                 mass=sc["mass_on_optimal"].values,
                 cluster=d["instrument"] + ":" + d["gamble_id"].astype(str))
    return d, rule


def stats(d):
    core = d[(d["frame"] != "dominant") & d["clear"]]
    dom = d[d["frame"] == "dominant"]
    gap = d[d["frame"] == "mixed"]["p_gamble"].mean() - d[d["frame"] == "gain"]["p_gamble"].mean()
    return {"dom": dom["correct"].mean() if len(dom) else np.nan,
            "ev": core["correct"].mean(), "mass": core["mass"].mean(), "gap": gap}


def boot(frames, fn, B, rng):
    """Percentile interval of fn(*resampled frames), resampling their shared clusters."""
    clusters = np.array(sorted(set.intersection(*(set(f["cluster"]) for f in frames))))
    groups = [dict(tuple(f.groupby("cluster"))) for f in frames]
    draws = []
    for _ in range(B):
        pick = rng.choice(clusters, len(clusters), replace=True)
        draws.append(fn(*[pd.concat([g[c] for c in pick]) for g in groups]))
    return np.nanpercentile(np.array(draws, dtype=float), [2.5, 97.5], axis=0)


def sampling_block(d, greedy):
    """Majority vote and unanimity across sampled traces; greedy-vs-majority agreement."""
    per = d.groupby(KEY + ["frame", "clear"]).agg(
        k=("sample", "nunique"), share_gamble=("p_gamble", lambda x: (x > 0.5).mean()),
        correct_rate=("correct", "mean")).reset_index()
    core = per[(per["frame"] != "dominant") & per["clear"]]
    maj_correct = (core["correct_rate"] > 0.5).mean()
    unanimous = ((per["share_gamble"] == 0) | (per["share_gamble"] == 1)).mean()
    print(f"    sampling: {per['k'].median():.0f} traces/cell | majority-vote ev {maj_correct:.3f}"
          f" | unanimous cells {unanimous:.3f}")
    if greedy is not None:
        g = greedy[KEY + ["p_gamble", "correct"]].rename(
            columns={"p_gamble": "g_pg", "correct": "g_correct"})
        m = per.merge(g, on=KEY)
        tie = m["share_gamble"] == 0.5
        agree = ((m["g_pg"] > 0.5) == (m["share_gamble"] > 0.5))[~tie].mean()
        gc = m[(m["frame"] != "dominant") & m["clear"]]
        print(f"    greedy answer matches the sampled majority in {agree:.3f} of {(~tie).sum()} "
              f"cells | greedy ev on these cells {gc['g_correct'].mean():.3f}")


def fmt_ci(est, ci):
    return f"{est:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]" if est == est else "   n/a"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--rule", choices=["riskneutral", "sqrt"], default=None)
    ap.add_argument("--vs", default=None, help="reference CSV for paired contrasts")
    ap.add_argument("--greedy", default=None, help="greedy run to compare sampled traces with")
    ap.add_argument("--drop-cells-of", default=None,
                    help="score only cells NOT in this CSV (e.g. the new half of a full-set run)")
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--B", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(1)

    ref = prepare(args.vs, args.rule, args.margin)[0] if args.vs else None
    greedy = prepare(args.greedy, args.rule, args.margin)[0] if args.greedy else None

    for path in args.csv:
        d, rule = prepare(path, args.rule, args.margin)
        if args.drop_cells_of and not d.empty:
            old = pd.read_csv(args.drop_cells_of, usecols=KEY).drop_duplicates()
            d = d.merge(old, on=KEY, how="left", indicator=True)
            d = d[d["_merge"] == "left_only"].drop(columns="_merge")
        if d.empty:
            print(f"\n== {Path(path).name}: no scored cells (all excluded for low A/B mass)")
            continue
        n_cells = len(d.drop_duplicates(KEY))
        label = d["induce"].iloc[0] if "induce" in d else "?"
        print(f"\n== {Path(path).name}  (induce={label}, scored as {rule}; {n_cells} cells, "
              f"{d['cluster'].nunique()} gambles, {len(d)} rows)")
        s = stats(d)
        ci = boot([d], lambda x: [stats(x)[m] for m in METRICS], args.B, rng)
        print("    " + "   ".join(f"{m} {s[m]:.3f} [{lo:.3f}, {hi:.3f}]"
                                  for m, lo, hi in zip(METRICS, ci[0], ci[1])))
        if d["sample"].nunique() > 1:
            sampling_block(d, greedy)
        if ref is not None and Path(path).resolve() != Path(args.vs).resolve():
            shared = d[KEY].drop_duplicates().merge(ref[KEY].drop_duplicates(), on=KEY)
            a, b = ref.merge(shared, on=KEY), d.merge(shared, on=KEY)
            diff = lambda x, y: [stats(y)[m] - stats(x)[m] for m in METRICS]  # noqa: E731
            est = diff(a, b)
            dci = boot([a, b], diff, args.B, rng)
            print(f"    minus {Path(args.vs).name} on {len(shared)} shared cells: "
                  + "  ".join(f"{m} {fmt_ci(e, (lo, hi))}"
                              for m, e, lo, hi in zip(METRICS, est, dci[0], dci[1])))


if __name__ == "__main__":
    main()
