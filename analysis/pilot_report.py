"""Summarize API pilot files: cost, tokens per call, parse and refusal rates, P(risky) by domain.

Usage: python analysis/pilot_report.py data/pilot
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from api_elicit import PRICES  # noqa: E402


def main():
    rows = []
    for f in sorted(Path(sys.argv[1]).glob("*.csv")):
        d = pd.read_csv(f)
        notes = d.note.map(json.loads)
        calls = notes.map(lambda n: len(n["letters"]) + n["errors"]).sum()
        tin, tout = notes.map(lambda n: n["tok_in"]).sum(), notes.map(lambda n: n["tok_out"]).sum()
        pin, pout = PRICES.get(d.model.iloc[0], (3.0, 15.0))
        d["domain"] = d.instrument.str.split("_").str[1]
        core = d[(d.excluded == 0) & d.frame.isin(["gain", "mixed"])]
        better = core[core.p * core.hi > core.sure]
        dom = d[(d.frame == "dominant") & (d.excluded == 0)]
        rows.append(dict(
            file=f.stem, cells=len(d), calls=calls,
            cost=tin / 1e6 * pin + tout / 1e6 * pout,
            out_per_call=tout / max(calls, 1),
            parsed=d.ab_mass.mean(), excluded=d.excluded.mean(),
            errors=notes.map(lambda n: n["errors"]).sum(),
            controls_ok=(dom.p_gamble > 0.5).mean(),
            risky_better_money=better[better.domain == "money"].p_gamble.mean(),
            risky_better_civ=better[better.domain == "civilians"].p_gamble.mean(),
        ))
    t = pd.DataFrame(rows)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\ntotal cost ~${t.cost.sum():.2f}")


if __name__ == "__main__":
    main()
