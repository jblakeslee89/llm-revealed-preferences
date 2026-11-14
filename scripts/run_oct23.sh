#!/usr/bin/env bash
# Oct 23 batch (John: "run all of it", ~$32). Every run's cap is its planned amount or what is
# left under the month's $100 credits (BUDGET, from scripts/spend.py), whichever is smaller.
#   bash scripts/run_oct23.sh
set -euo pipefail
cd "$(dirname "$0")/.."
: "${ANTHROPIC_API_KEY:?set ANTHROPIC_API_KEY first}"
BUDGET=${BUDGET:-99.40}  # $100 less a $0.05 probe not in any CSV, less a margin
MARGIN=0.75
cap() { python -c "import sys; print(f'{min(float(sys.argv[1]), float(sys.argv[2]) - $MARGIN):.2f}')" "$1" "$(python scripts/spend.py --left $BUDGET)"; }
gate() { python -c "import sys,pandas as pd; d=pd.read_csv(sys.argv[1]); x=d.excluded.mean(); print(f'gate {sys.argv[1]}: {len(d)} cells, excluded {x:.1%}'); sys.exit(int(x > 0.15))" "$1"; }
run() {  # run PLANNED_CAP OUT ARGS...
  local planned=$1 out=$2; shift 2
  local c; c=$(cap "$planned")
  if python -c "import sys; sys.exit(0 if float(sys.argv[1]) >= 0.25 else 1)" "$c"; then
    echo "=== $out (cap \$$c)"
    python src/api_elicit.py --provider anthropic --go --resume --out "$out" --max-spend "$c" "$@"
    gate "$out"
  else
    echo "=== SKIPPED $out: \$$c left under \$$BUDGET"
  fi
}
S=claude-sonnet-5-5; H=claude-haiku-5-5; O=claude-opus-5-5; F=claude-fable-5-1
G=data/stageA_grid_reason.csv; G2=data/stageA2_grid_reason.csv
HI="--thinking on --effort high --max-tokens 16000"

# finish Sonnet's second-wording reasoning run
run 2.50 data/stageA_sonnet55_a2_reason.csv --model $S --n 3 --grid $G2 --thinking off --reason
# 1. Fable at high effort, full main grid
run 12 data/stageA_fable51_api_think_high.csv --model $F --n 3 --grid $G $HI
# 2. effort ladder
run 4  data/stageA_sonnet55_api_think_medium.csv --model $S --n 3 --grid $G --thinking on --effort medium --max-tokens 16000
run 1  data/stageA_haiku55_api_think_high.csv    --model $H --n 3 --grid $G $HI
run 8  data/stageA_opus55_api_think_high.csv     --model $O --n 3 --grid $G $HI
# 3. second wording + release domain, thinking at low effort with a one-letter answer
run 7  data/stageA_opus55_a2_think.csv  --model $O --n 3 --grid $G2 --thinking on --effort low
run 12 data/stageA_fable51_a2_think.csv --model $F --n 3 --grid $G2 --thinking on --effort low
python scripts/spend.py
