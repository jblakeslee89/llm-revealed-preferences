"""Analysis of colab/prefill_comparison.py output (Qwen Instruct, chat, uninduced reasoning).

Non-mixed clear cells (|ln EV ratio| >= 0.05) whose trace contains a comparison against the
sure amount. Reports, by which option the truth favors:
  - mean P(less) at the comparison word under the model's own trace (own), the same trace
    with the EV rule prepended to the prompt (rule), and a one-sentence prefix stating the
    correct expected value ("The expected value of Option X is $E, which is") (minimal);
  - the generated comparison word by construction: whether the trace reaches the word through
    "..., which is ___ than" or any other phrasing;
  - the same with "Since the expected value of Option X, $E, is" (since; separate file), which
    states the same number without the "which is" construction;
  - a clustered logit for using the "which is" construction (cells where the gamble is better).

Usage:
    python analysis/prefill_analysis.py            # Qwen Instruct (chat)
    python analysis/prefill_analysis.py llama-inst_chat
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
WHICH = r"which\s+(?:is|are)\s*(?:slightly|significantly|much|far|still)?\s*$"


def main(stem="qwen-inst_chat"):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    d = pd.read_csv(DATA / f"prefill_{stem}.csv")
    since = DATA / f"prefill_{stem}_since.csv"  # follow-up run, "since" condition only
    if since.exists():
        d = pd.concat([d, pd.read_csv(since)], ignore_index=True)
    d["lr"] = np.log(d.p * d.hi / d.sure)
    d = d[(d.frame != "mixed") & (d.lr.abs() >= 0.05)].copy()
    d["truth"] = np.where(d.lr > 0, "gamble", "safe")

    w = d.pivot_table(index=["instrument", "trial_id"], columns="condition", values="p_less")
    own = d[d.condition == "own"].set_index(["instrument", "trial_id"])
    w = w.join(own[["frame", "truth", "generated_word", "context", "gamble_id"]])
    conds = [c for c in ["own", "rule", "minimal", "since"] if c in w.columns]
    w["gen_less"] = w.generated_word.isin(["less", "lower", "smaller"])
    w["which"] = w.context.str.contains(WHICH, regex=True)
    print(f"{len(w)} cells; own-trace argmax matches the generated word in "
          f"{((w.own > 0.5) == w.gen_less).mean():.3f}")

    print("\nMean P(less) at the comparison word")
    print(w.groupby("truth")[conds].mean().round(3).to_string())
    print("share with P(less) > 0.5:")
    print((w.groupby("truth")[conds].apply(lambda x: (x > 0.5).mean())).round(3).to_string())
    print("\nby frame, truth favors gamble")
    print(w[w.truth == "gamble"].groupby("frame")[conds].mean().round(3).to_string())
    x = w[(w.truth == "gamble") & w.gen_less]
    print(f"\nTraces that wrote a false 'less' (n = {len(x)}): mean P(less) own {x.own.mean():.3f}, "
          f"rule {x.rule.mean():.3f}")

    print("\nGenerated word by construction (rows: truth, 'which is' construction)")
    print(pd.crosstab([w.truth, w.which], w.gen_less.map({True: "less", False: "greater"})).to_string())
    print("\n'which is' construction rate by frame:", w.groupby("frame").which.mean().round(3).to_dict())

    imm = pd.read_csv(DATA / f"phase3_{stem}.csv")[["instrument", "trial_id", "p_gamble"]]
    g = w[w.truth == "gamble"].reset_index().merge(imm, on=["instrument", "trial_id"])
    g["which_i"] = g.which.astype(int)
    if g["which_i"].nunique() < 2:
        print("\nNo variation in the 'which is' construction; logit not estimated.")
        return
    g = g.merge(d[d.condition == "own"][["instrument", "trial_id", "lr"]], on=["instrument", "trial_id"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = smf.glm("which_i ~ p_gamble + lr + C(frame)", g, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": g.gamble_id})
    print("\nClustered logit, 'which is' construction (cells where the gamble is better):")
    for k in m.params.index:
        print(f"  {k:22s} {m.params[k]:+.3f}  se {m.bse[k]:.3f}  p {m.pvalues[k]:.3f}")


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:2])
