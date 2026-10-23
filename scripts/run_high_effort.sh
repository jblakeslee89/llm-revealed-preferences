#!/usr/bin/env bash
# Does more thinking move Fable and Sonnet toward Opus? (John approved Oct 20, $25 cap in total.)
# Fable at effort high on the test slice, then Sonnet at effort high on the full main grid with
# whatever is left of the $25. Thinking on, one-letter answer, as in the low-effort runs.
#   bash scripts/run_high_effort.sh
set -euo pipefail
cd "$(dirname "$0")/.."
: "${ANTHROPIC_API_KEY:?set ANTHROPIC_API_KEY first}"
api() { python src/api_elicit.py --provider anthropic --go --resume --thinking on --effort high --max-tokens 16000 "$@"; }
gate() { python -c "import sys,pandas as pd; d=pd.read_csv(sys.argv[1]); x=d.excluded.mean(); print(f'gate {sys.argv[1]}: excluded {x:.1%}'); sys.exit(int(x > 0.15))" "$1"; }
cost() { python - "$1" <<'PY'
import sys, json, pandas as pd
sys.path.insert(0, "src"); from api_elicit import PRICES
d = pd.read_csv(sys.argv[1]); pi, po = PRICES[d.model.iloc[0]]
print(f"{sum(json.loads(n)['tok_in'] * pi + json.loads(n)['tok_out'] * po for n in d.note) / 1e6:.2f}")
PY
}
F=data/pilot/stageA_fable51_think_high.csv
api --model claude-fable-5-1 --domains money,civilians --gambles 10 --n 2 --grid data/stageA_grid_reason.csv --out $F --max-spend 12
gate $F
LEFT=$(python -c "print(f'{25 - 0.5 - float(\"$(cost $F)\"):.2f}')")
echo "Sonnet cap \$$LEFT"
api --model claude-sonnet-5-5 --n 3 --grid data/stageA_grid_reason.csv --out data/stageA_sonnet55_api_think_high.csv --max-spend $LEFT
gate data/stageA_sonnet55_api_think_high.csv
