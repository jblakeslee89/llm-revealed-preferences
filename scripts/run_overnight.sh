#!/usr/bin/env bash
# Overnight batch after Sonnet step 1. John approved Oct 20: up to $50 in total (step 1
# included), and "add opus". Each run's cap is its planned amount or what is left under the
# budget (scripts/spend.py), whichever is smaller; runs that don't fit are skipped for November.
# Stops if a run's exclusion rate passes 15%, which would mean a parse problem.
#   bash scripts/run_overnight.sh
set -euo pipefail
cd "$(dirname "$0")/.."
: "${ANTHROPIC_API_KEY:?set ANTHROPIC_API_KEY first}"
BUDGET=${BUDGET:-50}
MARGIN=0.75  # caps are checked before each cell, and in-flight calls can overshoot a little
cap() { python -c "import sys; l=float(sys.argv[2])-$MARGIN; print(f'{min(float(sys.argv[1]), l):.2f}')" "$1" "$(python scripts/spend.py --left $BUDGET)"; }
run() {  # run PLANNED_CAP OUT ARGS...: one api_elicit run under the remaining budget, then the gate
  local planned=$1 out=$2; shift 2
  local c; c=$(cap "$planned")
  if python -c "import sys; sys.exit(0 if float(sys.argv[1]) >= 0.25 else 1)" "$c"; then
    echo "=== $out (cap \$$c)"
    python src/api_elicit.py --provider anthropic --go --resume --out "$out" --max-spend "$c" "$@"
    gate "$out"
  else
    echo "=== SKIPPED $out: \$$c left under the \$$BUDGET budget"
  fi
}
gate() {
  python - "$1" <<'PY'
import sys, pandas as pd
d = pd.read_csv(sys.argv[1]); x = d.excluded.mean()
print(f"gate {sys.argv[1]}: {len(d)} cells, excluded {x:.1%}")
sys.exit(1 if x > 0.15 else 0)
PY
}
S=claude-sonnet-5-5; H=claude-haiku-5-5; O=claude-opus-5-5
P="--domains money,civilians --gambles 10 --n 2"
TH="--thinking on --effort low"

# 0. cleanups: rerun the thinking test cells the old parser missed; finish the snap run
run 0.40 data/pilot/stageA_sonnet55_think.csv --model $S $P --grid data/stageA_grid_reason.csv $TH
run 0.40 data/stageA_sonnet55_api.csv --model $S --n 5 --grid data/stageA_grid.csv --thinking off

# 1. step 2: Sonnet extended thinking
run 8 data/stageA_sonnet55_api_think.csv --model $S --n 3 --grid data/stageA_grid_reason.csv $TH

# 2. Haiku, all three arms
run 1 data/stageA_haiku55_api.csv        --model $H --n 5 --grid data/stageA_grid.csv --thinking off
run 2 data/stageA_haiku55_api_reason.csv --model $H --n 3 --grid data/stageA_grid_reason.csv --thinking off --reason
run 2 data/stageA_haiku55_api_think.csv  --model $H --n 3 --grid data/stageA_grid_reason.csv $TH

# 3. Opus (thinking cannot be turned off): test slice, then the main grid in both prompt styles,
#    one-letter answer and written reasoning, each with thinking at low effort
run 1  data/pilot/stageA_opus55_think.csv --model $O $P --grid data/stageA_grid_reason.csv $TH
run 8  data/stageA_opus55_api_think.csv   --model $O --n 3 --grid data/stageA_grid_reason.csv $TH
run 12 data/stageA_opus55_api_reason.csv  --model $O --n 3 --grid data/stageA_grid_reason.csv $TH --reason

# 4. Stage A2 (second wording + release domain), cheapest first; what doesn't fit waits for November
run 0.60 data/stageA_haiku55_a2.csv         --model $H --n 5 --grid data/stageA2_grid.csv --thinking off
run 1.50 data/stageA_haiku55_a2_reason.csv  --model $H --n 3 --grid data/stageA2_grid_reason.csv --thinking off --reason
run 1    data/stageA_haiku55_a2_think.csv   --model $H --n 3 --grid data/stageA2_grid_reason.csv $TH
run 5    data/stageA_sonnet55_a2.csv        --model $S --n 5 --grid data/stageA2_grid.csv --thinking off
run 5    data/stageA_sonnet55_a2_think.csv  --model $S --n 3 --grid data/stageA2_grid_reason.csv $TH
run 11.5 data/stageA_sonnet55_a2_reason.csv --model $S --n 3 --grid data/stageA2_grid_reason.csv --thinking off --reason
python scripts/spend.py
