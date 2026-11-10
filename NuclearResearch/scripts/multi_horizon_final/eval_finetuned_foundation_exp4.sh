#!/bin/bash
# Evaluate the fine-tuned foundation models (Chronos2_FT, Chronos_FT) on the 15 LOCA
# test scenarios at all horizons (6 runs). Requires the checkpoints written by
# finetune_foundation_exp4.sh under checkpoints_foundation/finetuned/.
#
# Usage: bash NuclearResearch/scripts/multi_horizon_final/eval_finetuned_foundation_exp4.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
[ -f "$ROOT/NuclearResearch/src/run_loca_forecast.py" ] || ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" && [ -f NuclearResearch/src/run_loca_forecast.py ] || { echo "Run from the repository root" >&2; exit 1; }

echo "================================================================================"
echo "Fine-Tuned Foundation Model Evaluation - Exp4 LOCA Test"
echo "================================================================================"
echo ""
echo "Models: Chronos2_FT (multivariate), Chronos_FT (univariate)"
echo "Horizons: 30s (pl3), 300s (pl30), 600s (pl60)"
echo "Total evaluation runs: 6"
echo "================================================================================"
echo ""

# ============================================================================
# Configuration
# ============================================================================

ROOT_PATH="./dataset"
DATA_PATH="normal_LOCAC_RW_LOF_SGATR_train_LOCA_test"
CKPT_BASE="./checkpoints_foundation/finetuned"

ENC_IN=74
DEC_IN=74
C_OUT=1
SEQ_LEN=30
FEATURES="MS"
TARGET="TPCT"
BATCH_SIZE=8
GPU=0

LOG_DIR="NuclearResearch/logs"
mkdir -p "$LOG_DIR"

mkdir -p NuclearResearch/results_summary/timing
TIMING_FILE="NuclearResearch/results_summary/timing/timing_finetune_eval_summary.csv"
echo "model,horizon,pred_len,total_seconds,total_human,status" > "$TIMING_FILE"

# Horizon configurations: pred_len label_len name
declare -A HORIZONS
HORIZONS[short]="3 2 30s"
HORIZONS[medium]="30 15 300s"
HORIZONS[long]="60 30 600s"

# ============================================================================
# Evaluation function
# ============================================================================

run_eval() {
    local MODEL_NAME=$1
    local CKPT_PATH=$2
    local PRED_LEN=$3
    local LABEL_LEN=$4
    local HORIZON_NAME=$5
    local RUN_NUM=$6
    local TOTAL=$7

    local MODEL_ID="LOCA_TPCT_exp4_finetuned_${MODEL_NAME}_pl${PRED_LEN}"
    local LOG_FILE="${LOG_DIR}/eval_finetuned_${MODEL_NAME}_pl${PRED_LEN}.log"

    # Verify checkpoint exists
    if [ ! -d "$CKPT_PATH" ]; then
        echo "WARNING: Checkpoint not found: ${CKPT_PATH}"
        echo "SKIPPED: ${MODEL_NAME} pl${PRED_LEN} (no checkpoint)" >> "${LOG_DIR}/eval_finetuned_failed.log"
        echo "${MODEL_NAME},${HORIZON_NAME},${PRED_LEN},0,0s,SKIPPED" >> "$TIMING_FILE"
        return 1
    fi

    echo "================================================================================"
    echo "[${RUN_NUM}/${TOTAL}] ${MODEL_NAME} | ${HORIZON_NAME} (pred_len=${PRED_LEN})"
    echo "  Checkpoint: ${CKPT_PATH}"
    echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  Log: ${LOG_FILE}"
    echo "================================================================================"

    local START=$(date +%s)

    python NuclearResearch/src/run_loca_zeroshot.py \
        --model "$MODEL_NAME" \
        --model_id "$MODEL_ID" \
        --finetuned_model_path "$CKPT_PATH" \
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
    local END=$(date +%s)
    local ELAPSED=$((END - START))
    local MIN=$((ELAPSED / 60))
    local SEC=$((ELAPSED % 60))

    if [ $EXIT_CODE -eq 0 ]; then
        local STATUS="OK"
    else
        local STATUS="FAILED(exit=$EXIT_CODE)"
        echo "FAILED: ${MODEL_NAME} pl${PRED_LEN}" >> "${LOG_DIR}/eval_finetuned_failed.log"
    fi

    echo "  ${MODEL_NAME} ${HORIZON_NAME} -- ${STATUS} -- ${MIN}m ${SEC}s"
    echo "${MODEL_NAME},${HORIZON_NAME},${PRED_LEN},${ELAPSED},${MIN}m${SEC}s,${STATUS}" >> "$TIMING_FILE"
    echo ""
    sleep 2
}

# ============================================================================
# Main execution
# ============================================================================

TOTAL_RUNS=6
RUN_COUNT=0

> "${LOG_DIR}/eval_finetuned_failed.log" 2>/dev/null

START_TIME=$(date +%s)

for HORIZON_KEY in short medium long; do
    HORIZON_INFO=(${HORIZONS[$HORIZON_KEY]})
    PRED_LEN=${HORIZON_INFO[0]}
    LABEL_LEN=${HORIZON_INFO[1]}
    HORIZON_NAME=${HORIZON_INFO[2]}

    echo "--------------------------------------------------------------------------------"
    echo "-- Horizon: ${HORIZON_NAME} (pred_len=${PRED_LEN}, label_len=${LABEL_LEN})"
    echo "--------------------------------------------------------------------------------"
    echo ""

    # Chronos-2 Fine-Tuned (multivariate)
    RUN_COUNT=$((RUN_COUNT + 1))
    run_eval "Chronos2_FT" \
        "${CKPT_BASE}/chronos2_multivar_exp4_pl${PRED_LEN}/finetuned-ckpt" \
        $PRED_LEN $LABEL_LEN "$HORIZON_NAME" $RUN_COUNT $TOTAL_RUNS

    # Chronos Bolt Fine-Tuned (univariate)
    RUN_COUNT=$((RUN_COUNT + 1))
    run_eval "Chronos_FT" \
        "${CKPT_BASE}/chronos_bolt_exp4_pl${PRED_LEN}" \
        $PRED_LEN $LABEL_LEN "$HORIZON_NAME" $RUN_COUNT $TOTAL_RUNS
done

# ============================================================================
# Summary
# ============================================================================

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(( (ELAPSED % 3600) / 60 ))

echo ""
echo "================================================================================"
echo "FINE-TUNED EVALUATION COMPLETE"
echo "================================================================================"
echo "Total runs: ${RUN_COUNT}/${TOTAL_RUNS}"
echo "Elapsed: ${HOURS}h ${MINUTES}m"
echo ""

if [ -s "${LOG_DIR}/eval_finetuned_failed.log" ]; then
    echo "FAILED/SKIPPED RUNS:"
    cat "${LOG_DIR}/eval_finetuned_failed.log"
    echo ""
else
    echo "All evaluation runs completed successfully."
fi

echo ""
echo "Timing summary:"
column -t -s',' "$TIMING_FILE"
echo ""
echo "Results saved to: ./results/"
echo "Logs saved to: ${LOG_DIR}/"
echo ""
echo "================================================================================"
