#!/bin/bash
# ARIMA(1,1,1) baseline - all experiments (exp0-exp4) x all horizons (30s, 300s, 600s)
#
# Usage: bash NuclearResearch/scripts/multi_horizon_final/train_arima_all_experiments.sh
# Optional: LAUNCHER="<prefix>" runs each job through a scheduler (see submit_slurm.sh).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
[ -f "$ROOT/NuclearResearch/src/run_loca_forecast.py" ] || ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" && [ -f NuclearResearch/src/run_loca_forecast.py ] || { echo "Run from the repository root" >&2; exit 1; }

LAUNCHER="${LAUNCHER:-}"
SEQ_LEN=30
LOG_DIR="NuclearResearch/logs"
mkdir -p "$LOG_DIR"

for EXP in exp0 exp1 exp2 exp3 exp4; do
    for PL in 3 30 60; do
        echo "ARIMA | ${EXP} | pred_len=${PL}"
        if [ -n "$LAUNCHER" ]; then
            $LAUNCHER python NuclearResearch/src/baselines/run_arima_baseline.py \
                --exp "$EXP" --seq_len $SEQ_LEN --pred_len $PL
        else
            python NuclearResearch/src/baselines/run_arima_baseline.py \
                --exp "$EXP" --seq_len $SEQ_LEN --pred_len $PL \
                2>&1 | tee "${LOG_DIR}/arima_${EXP}_pl${PL}.log"
        fi
    done
done

echo "ARIMA runs finished (or submitted): 5 experiments x 3 horizons."
