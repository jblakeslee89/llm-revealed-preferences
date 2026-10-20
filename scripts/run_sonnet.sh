#!/usr/bin/env bash
# Sonnet 5.5 first (docs/paper-risk-advice-plan.md). Stage 1 = small test slice in all three
# arms, then full snap and written-reasoning runs: caps add to $16. Stage 2 = extended thinking,
# run separately after the stage-1 report: cap $25.
#   bash scripts/run_sonnet.sh 1     # tonight
#   bash scripts/run_sonnet.sh 2     # after checking the stage-1 cost and thinking tokens
# Needs ANTHROPIC_API_KEY in the environment. Never put the key in this file.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${ANTHROPIC_API_KEY:?set ANTHROPIC_API_KEY first}"
M=claude-sonnet-5-5
api() { python src/api_elicit.py --provider anthropic --model $M --go --resume "$@"; }
mkdir -p data/pilot

if [ "${1:-1}" = 1 ]; then
  P="--domains money,civilians --gambles 10 --n 2"
  api $P --grid data/stageA_grid.csv        --out data/pilot/stageA_sonnet55_snap.csv   --thinking off --max-spend 0.50
  api $P --grid data/stageA_grid_reason.csv --out data/pilot/stageA_sonnet55_reason.csv --thinking off --reason --max-spend 1.00
  api $P --grid data/stageA_grid_reason.csv --out data/pilot/stageA_sonnet55_think.csv  --thinking on --effort low --max-spend 2.50
  python analysis/pilot_report.py data/pilot
  api --n 5 --grid data/stageA_grid.csv        --out data/stageA_sonnet55_api.csv        --thinking off --max-spend 2.50
  api --n 3 --grid data/stageA_grid_reason.csv --out data/stageA_sonnet55_api_reason.csv --thinking off --reason --max-spend 10.00
else
  api --n 3 --grid data/stageA_grid_reason.csv --out data/stageA_sonnet55_api_think.csv  --thinking on --effort low --max-spend 25.00
fi
python analysis/stage_a.py --stems sonnet55_api --B 300
