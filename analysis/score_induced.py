"""Score compliance with an induced utility function (Armour arm, Aug 2026).

Induced-value logic: the elicitation run was given an explicit decision rule
(risk-neutral EV maximization, or expected utility with u(x) = sqrt(x)) via
`phase3_elicit.py --induce ...`. This script computes the induced-optimal action
for every cell and reports how much probability mass the model put on it.

If the model cannot follow an induced utility function, its uninduced
"preferences" are on shakier ground; if it can, compliance calibrates the
elicitation. Run on the OLMo staircase to locate where in post-training a model
becomes inducible at all.

Metrics:
  - compliance mass: mean probability assigned to the induced-optimal option
  - hard accuracy:   share of cells where p(optimal option) > 0.5
Both reported overall, excluding near-indifferent cells (|ln EU ratio| below
--margin, where the induced rule barely distinguishes the options), and on the
dominant controls (where both rules agree and the answer is unambiguous).

Usage:
    python analysis/score_induced.py data/phase3_qwen-inst_chat_riskneutral.csv --induce riskneutral
    python analysis/score_induced.py data/phase3_olmo-sft_fewshot_sqrt.csv --induce sqrt \\
        --baseline data/phase3_olmo-sft_fewshot.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def induced_eu(df, rule):
    s = df["sure"].values.astype(float)
    h = df["hi"].values.astype(float)
    p = df["p"].values.astype(float)
    if rule == "riskneutral":
        return p * h, s
    if rule == "sqrt":
        return p * np.sqrt(h), np.sqrt(s)
    raise ValueError(rule)


def score(df, rule, margin):
    eu_g, eu_s = induced_eu(df, rule)
    optimal_gamble = eu_g > eu_s
    pg = df["p_gamble"].values.astype(float)
    mass_on_optimal = np.where(optimal_gamble, pg, 1 - pg)
    hard = (pg > 0.5) == optimal_gamble
    dist = np.abs(np.log(np.clip(eu_g, 1e-9, None) / np.clip(eu_s, 1e-9, None)))
    clear = dist >= margin
    return pd.DataFrame({
        "frame": df["frame"].values,
        "mass_on_optimal": mass_on_optimal,
        "hard_correct": hard,
        "clear": clear,
    })


def report(tag, sc):
    dom = sc[sc["frame"] == "dominant"]
    core = sc[sc["frame"] != "dominant"]
    core_clear = core[core["clear"]]
    print(f"\n===== {tag} =====")
    print(f"  core cells (all {len(core)}):        compliance mass {core['mass_on_optimal'].mean():.3f}"
          f"   hard accuracy {core['hard_correct'].mean():.3f}")
    print(f"  core, clear cells only ({len(core_clear)}): compliance mass "
          f"{core_clear['mass_on_optimal'].mean():.3f}   hard accuracy {core_clear['hard_correct'].mean():.3f}")
    if len(dom):
        print(f"  dominant controls ({len(dom)}):       compliance mass {dom['mass_on_optimal'].mean():.3f}"
              f"   hard accuracy {dom['hard_correct'].mean():.3f}")
    by = core_clear.groupby("frame")["hard_correct"].mean()
    print("  hard accuracy by frame (clear cells): "
          + "  ".join(f"{k}={v:.3f}" for k, v in by.items()))


def load(path):
    df = pd.read_csv(path)
    df = df[df["excluded"] == 0].copy()
    df = df[df["p_gamble"].notna()]
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", help="elicitation CSV from an --induce run")
    ap.add_argument("--induce", required=True, choices=["riskneutral", "sqrt"])
    ap.add_argument("--margin", type=float, default=0.05,
                    help="|ln EU ratio| below this = near-indifferent, excluded from 'clear'")
    ap.add_argument("--baseline", default=None,
                    help="optional comparison CSV from the same model+format (the uninduced "
                         "run, or the immediate-answer induced run when csv is a --reason "
                         "run), restricted to csv's trials and scored against the same rule")
    args = ap.parse_args()

    main_df = load(args.csv)
    sc = score(main_df, args.induce, args.margin)
    report(f"INDUCED ({args.induce})", sc)

    if args.baseline:
        base = load(args.baseline)
        # reason-mode runs may cover a gamble subset (--gambles): compare on the same cells.
        # trial_id restarts per instrument, so match on the pair.
        key = ["instrument", "trial_id"]
        base = base.merge(main_df[key].drop_duplicates(), on=key)
        sb = score(base, args.induce, args.margin)
        report(f"BASELINE ({args.baseline.split('/')[-1]}, scored against {args.induce})", sb)
        d = (sc[sc['frame'] != 'dominant']['mass_on_optimal'].mean()
             - sb[sb['frame'] != 'dominant']['mass_on_optimal'].mean())
        print(f"\n  effect (core compliance mass, this run - baseline): {d:+.3f}")
        print("  (near zero = the instruction did nothing; large positive = the model is"
              " inducible; compliance <<1 even when induced = elicitation ceiling)")


if __name__ == "__main__":
    main()
