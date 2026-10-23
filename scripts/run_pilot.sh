#!/usr/bin/env bash
# Pilot for the Claude frontier runs (docs/paper-risk-advice-plan.md): money + civilians,
# 10 gambles, n = 2, both models, all three arms. Hard caps add up to $5; expected ~$5-6 of
# work, so the Sonnet thinking arm may stop early at its cap, which is fine for a pilot.
# Needs ANTHROPIC_API_KEY in the environment. Never put the key in this file.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${ANTHROPIC_API_KEY:?set ANTHROPIC_API_KEY first}"
mkdir -p data/pilot
COMMON="--provider anthropic --domains money,civilians --gambles 10 --n 2 --go --resume"
run() { python src/api_elicit.py $COMMON "$@"; }

run --model claude-haiku-5-5  --grid data/stageA_grid.csv        --out data/pilot/stageA_haiku55_snap.csv   --thinking off --max-spend 0.25
run --model claude-haiku-5-5  --grid data/stageA_grid_reason.csv --out data/pilot/stageA_haiku55_reason.csv --thinking off --reason --max-spend 0.25
run --model claude-haiku-5-5  --grid data/stageA_grid_reason.csv --out data/pilot/stageA_haiku55_think.csv  --thinking on --effort low --max-spend 0.50
run --model claude-sonnet-5-5 --grid data/stageA_grid.csv        --out data/pilot/stageA_sonnet55_snap.csv  --thinking off --max-spend 0.50
run --model claude-sonnet-5-5 --grid data/stageA_grid_reason.csv --out data/pilot/stageA_sonnet55_reason.csv --thinking off --reason --max-spend 1.00
run --model claude-sonnet-5-5 --grid data/stageA_grid_reason.csv --out data/pilot/stageA_sonnet55_think.csv --thinking on --effort low --max-spend 2.50

python analysis/pilot_report.py data/pilot
