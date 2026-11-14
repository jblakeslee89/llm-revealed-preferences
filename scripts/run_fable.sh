#!/usr/bin/env bash
# Fable 5.1 (John approved Oct 20, ~$10-12): thinking at low effort with a one-letter answer on
# the main grid, the arm where Opus treated lives like money. Fable cannot turn thinking off.
#   bash scripts/run_fable.sh
set -euo pipefail
cd "$(dirname "$0")/.."
: "${ANTHROPIC_API_KEY:?set ANTHROPIC_API_KEY first}"
api() { python src/api_elicit.py --provider anthropic --model claude-fable-5-1 --go --resume --thinking on --effort low "$@"; }
api --domains money,civilians --gambles 10 --n 2 --grid data/stageA_grid_reason.csv --out data/pilot/stageA_fable51_think.csv --max-spend 1.50
python -c "import sys,pandas as pd; d=pd.read_csv('data/pilot/stageA_fable51_think.csv'); x=d.excluded.mean(); print(f'test slice excluded {x:.1%}'); sys.exit(int(x > 0.15))"
api --n 3 --grid data/stageA_grid_reason.csv --out data/stageA_fable51_api_think.csv --max-spend 11
python scripts/spend.py
