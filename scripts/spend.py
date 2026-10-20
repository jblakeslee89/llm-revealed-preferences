"""Dollars spent so far on the API runs, from the token counts each CSV row records.
    python scripts/spend.py                 # table per file and the total
    python scripts/spend.py --left 50       # just the dollars left under a $50 budget"""
import argparse, glob, json, sys
import pandas as pd
sys.path.insert(0, "src")
from api_elicit import PRICES

FILES = ["data/pilot/stageA_*55_*.csv", "data/stageA_*55_*.csv", "data/stageA_opus55*.csv", "data/pilot/stageA_fable51*.csv", "data/stageA_fable51*.csv"]  # *_high files match these patterns too

# rows dropped from the thinking test slice before the parser fix (Oct 20) were billed too
DROPPED = 0.28

ap = argparse.ArgumentParser()
ap.add_argument("--left", type=float)
args = ap.parse_args()
rows, seen = [], set()
for pat in FILES:
    for f in sorted(glob.glob(pat)):
        if f in seen:
            continue
        seen.add(f)
        d = pd.read_csv(f)
        notes = [json.loads(x) for x in d.note]
        pin, pout = PRICES[d.model.iloc[0]]
        tin, tout = sum(n["tok_in"] for n in notes), sum(n["tok_out"] for n in notes)
        rows.append(dict(file=f, cells=len(d), excluded=round(d.excluded.mean(), 3),
                         out_per_cell=round(tout / len(d)), cost=round(tin / 1e6 * pin + tout / 1e6 * pout, 2)))
t = pd.DataFrame(rows)
if args.left is not None:
    print(f"{max(args.left - t.cost.sum() - DROPPED, 0):.2f}")
else:
    print(t.to_string(index=False)); print(f"dropped rows ${DROPPED:.2f}\ntotal ${t.cost.sum() + DROPPED:.2f}")
