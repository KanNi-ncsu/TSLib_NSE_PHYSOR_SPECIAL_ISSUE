#!/bin/bash
# Foundation-model zero-shot inference - all experiments (exp0-exp4) x 3 horizons x 3 models
#   Chronos  - amazon/chronos-bolt-base
#   Chronos2 - amazon/chronos-2
#   TimesFM  - google/timesfm-2.0-500m-pytorch
#
# Usage: bash NuclearResearch/scripts/multi_horizon_final/run_foundation_models_all_experiments.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
[ -f "$ROOT/NuclearResearch/src/run_loca_forecast.py" ] || ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" && [ -f NuclearResearch/src/run_loca_forecast.py ] || { echo "Run from the repository root" >&2; exit 1; }

echo "================================================================================"
echo "Foundation Model Zero-Shot Inference - All Experiments"
echo "================================================================================"
echo "Models (3): Chronos, Chronos2, TimesFM"
echo "Experiments (5): exp0, exp1, exp2, exp3, exp4"
echo "Horizons (3): 30s (pl3), 300s (pl30), 600s (pl60)"
echo "Total inference runs: 45"
echo "Features: MS (74 multivariate inputs -> 1 TPCT output)"
echo "Sequence length: 30 timesteps (300 seconds)"
echo "================================================================================"

# ============================================================================
# Configuration
# ============================================================================

ROOT_PATH="./dataset"
ENC_IN=74
DEC_IN=74
C_OUT=1
SEQ_LEN=30
FEATURES="MS"
TARGET="TPCT"
BATCH_SIZE=8
GPU=0

# Experiment datasets
declare -A EXP_DATA
EXP_DATA[exp0]="LOCA_only_train_LOCA_test"
EXP_DATA[exp1]="LOCA_normal_train_LOCA_test"
EXP_DATA[exp2]="normal_LOCAC_train_LOCA_test"
EXP_DATA[exp3]="LOCA_FLB_normal_train_LOCA_test"
EXP_DATA[exp4]="normal_LOCAC_RW_LOF_SGATR_train_LOCA_test"

# Experiment descriptions
declare -A EXP_DESC
EXP_DESC[exp0]="LOCA-Only Baseline"
EXP_DESC[exp1]="LOCA + Normal Operation"
EXP_DESC[exp2]="LOCAC + Normal (Cross-Accident Transfer)"
EXP_DESC[exp3]="FLB + Normal (Cross-Accident Transfer)"
EXP_DESC[exp4]="Multi-Accident Diversity"

# Horizon configurations: pred_len label_len name
declare -A HORIZONS
HORIZONS[short]="3 2 30s"
HORIZONS[medium]="30 15 300s"
HORIZONS[long]="60 30 600s"

# Foundation models (Chronos loops over the input channels and is the slowest)
MODELS=(TimesFM Chronos2 Chronos)

LOG_DIR="NuclearResearch/logs"
TIMING_DIR="NuclearResearch/results_summary/timing"
mkdir -p "$LOG_DIR" "$TIMING_DIR"

# Timing summary files (one per experiment)
for EXP_KEY in exp0 exp1 exp2 exp3 exp4; do
    TIMING_FILE="${TIMING_DIR}/timing_${EXP_KEY}_zeroshot_summary.csv"
    echo "model,horizon,pred_len,total_seconds,total_human,status" > "$TIMING_FILE"
done

# ============================================================================
# Inference function
# ============================================================================

run_zeroshot() {
    local MODEL_NAME=$1
    local EXP_KEY=$2
    local PRED_LEN=$3
    local LABEL_LEN=$4
    local HORIZON_NAME=$5
    local DATA_PATH=${EXP_DATA[$EXP_KEY]}
    local RUN_NUM=$6
    local TOTAL_RUNS=$7

    local MODEL_ID="LOCA_TPCT_${EXP_KEY}_foundation_${MODEL_NAME}_pl${PRED_LEN}"
    local LOG_FILE="${LOG_DIR}/zeroshot_${EXP_KEY}_${MODEL_NAME}_pl${PRED_LEN}.log"

    echo "================================================================================"
    echo "[${RUN_NUM}/${TOTAL_RUNS}] ${MODEL_NAME} | ${EXP_KEY} (${EXP_DESC[$EXP_KEY]}) | ${HORIZON_NAME}"
    echo "  Dataset: ${DATA_PATH}"
    echo "  pred_len=${PRED_LEN}, label_len=${LABEL_LEN}, seq_len=${SEQ_LEN}"
    echo "  Start time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  Log: ${LOG_FILE}"
    echo "================================================================================"

    local RUN_START=$(date +%s)

    python NuclearResearch/src/run_loca_zeroshot.py \
        --model "$MODEL_NAME" \
        --model_id "$MODEL_ID" \
        --root_path "$ROOT_PATH" \
        --data_path "$DATA_PATH" \
        --task_name zero_shot_forecast \
        --is_training 0 \
        --data nuclear_accident \
        --enc_in $ENC_IN \
        --dec_in $DEC_IN \
        --c_out $C_OUT \
        --seq_len $SEQ_LEN \
        --label_len $LABEL_LEN \
        --pred_len $PRED_LEN \
        --features $FEATURES \
        --target $TARGET \
        --batch_size $BATCH_SIZE \
        --gpu $GPU \
        2>&1 | tee "$LOG_FILE"

    local EXIT_CODE=${PIPESTATUS[0]}
    local RUN_END=$(date +%s)
    local RUN_ELAPSED=$((RUN_END - RUN_START))
    local RUN_MIN=$((RUN_ELAPSED / 60))
    local RUN_SEC=$((RUN_ELAPSED % 60))

    if [ $EXIT_CODE -eq 0 ]; then
        local STATUS="OK"
    else
        local STATUS="FAILED(exit=$EXIT_CODE)"
        echo "FAILED: ${MODEL_NAME} ${EXP_KEY} ${HORIZON_NAME}" >> "${LOG_DIR}/zeroshot_failed_runs.log"
    fi

    echo "--------------------------------------------------------------------------------"
    echo "  ${MODEL_NAME} ${HORIZON_NAME} -- ${STATUS} -- ${RUN_MIN}m ${RUN_SEC}s (${RUN_ELAPSED}s total)"
    echo "  End time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "--------------------------------------------------------------------------------"

    local TIMING_FILE="${TIMING_DIR}/timing_${EXP_KEY}_zeroshot_summary.csv"
    echo "${MODEL_NAME},${HORIZON_NAME},${PRED_LEN},${RUN_ELAPSED},${RUN_MIN}m${RUN_SEC}s,${STATUS}" >> "$TIMING_FILE"

    sleep 2
}

# ============================================================================
# Main execution loop
# ============================================================================

TOTAL_RUNS=$(( ${#MODELS[@]} * 5 * 3 ))
RUN_COUNT=0

# Clear failed runs log
> "${LOG_DIR}/zeroshot_failed_runs.log"

START_TIME=$(date +%s)

for EXP_KEY in exp0 exp1 exp2 exp3 exp4; do
    echo "################################################################################"
    echo "## EXPERIMENT: ${EXP_KEY} - ${EXP_DESC[$EXP_KEY]}"
    echo "## Dataset: ${EXP_DATA[$EXP_KEY]}"
    echo "################################################################################"

    for HORIZON_KEY in short medium long; do
        HORIZON_INFO=(${HORIZONS[$HORIZON_KEY]})
        PRED_LEN=${HORIZON_INFO[0]}
        LABEL_LEN=${HORIZON_INFO[1]}
        HORIZON_NAME=${HORIZON_INFO[2]}

        echo "--------------------------------------------------------------------------------"
        echo "-- Horizon: ${HORIZON_NAME} (pred_len=${PRED_LEN}, label_len=${LABEL_LEN})"
        echo "--------------------------------------------------------------------------------"

        for MODEL in "${MODELS[@]}"; do
            RUN_COUNT=$((RUN_COUNT + 1))
            run_zeroshot "$MODEL" "$EXP_KEY" "$PRED_LEN" "$LABEL_LEN" "$HORIZON_NAME" "$RUN_COUNT" "$TOTAL_RUNS"
        done
    done
done

# ============================================================================
# Summary
# ============================================================================

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(( (ELAPSED % 3600) / 60 ))

echo "================================================================================"
echo "ZERO-SHOT INFERENCE COMPLETE"
echo "================================================================================"
echo "Total runs: ${RUN_COUNT}/${TOTAL_RUNS}"
echo "Elapsed time: ${HOURS}h ${MINUTES}m"

if [ -s "${LOG_DIR}/zeroshot_failed_runs.log" ]; then
    echo "FAILED RUNS:"
    cat "${LOG_DIR}/zeroshot_failed_runs.log"
else
    echo "All runs completed successfully."
fi

echo "Timing CSVs: ${TIMING_DIR}/timing_exp*_zeroshot_summary.csv"
echo "Results:     ./results/zero_shot_forecast_LOCA_TPCT_exp*_foundation_*/"
echo "Logs:        ${LOG_DIR}/zeroshot_*.log"
echo "================================================================================"
