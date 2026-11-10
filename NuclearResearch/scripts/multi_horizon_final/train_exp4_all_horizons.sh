#!/bin/bash
# Experiment 4: Multi-Accident Diversity - All Horizons (30s, 300s, 600s)
# Dataset: dataset/normal_LOCAC_RW_LOF_SGATR_train_LOCA_test
# Train: 75 each of LOCA, LOCAC, FLB, RW, SGATR + 20 Normal = 395 | Val: 10 each of LOCAC, FLB, RW, SGATR = 40 | Test: 15 LOCA
#
# Usage: bash NuclearResearch/scripts/multi_horizon_final/train_exp4_all_horizons.sh

# Run from the repository root
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
[ -f "$ROOT/NuclearResearch/src/run_loca_forecast.py" ] || ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" && [ -f NuclearResearch/src/run_loca_forecast.py ] || { echo "Run from the repository root" >&2; exit 1; }

echo "================================================================================"
echo "Experiment 4: Multi-Accident Diversity - Multi-Horizon Training (24 models)"
echo "================================================================================"
echo ""
echo "Dataset: normal_LOCAC_RW_LOF_SGATR_train_LOCA_test"
echo "Training scenarios: 75 each of LOCA, LOCAC, FLB, RW, SGATR + 20 Normal = 395"
echo "Validation scenarios: 10 each of LOCAC, FLB, RW, SGATR = 40"
echo "Test scenarios: 15 LOCA (same for all experiments)"
echo ""
echo "Horizons (3):"
echo "  Short (30s):   seq_len=30, pred_len=3,  label_len=2"
echo "  Medium (300s): seq_len=30, pred_len=30, label_len=15"
echo "  Long (600s):   seq_len=30, pred_len=60, label_len=30"
echo ""
echo "Models (24): TFT, Transformer, Nonstationary, PatchTST, Crossformer, Pyraformer,"
echo "             Reformer, TSMixer, iTransformer, TimesNet, TimeMixer, SegRNN,"
echo "             Informer, DLinear, FEDformer, Autoformer, MICN, LSTM, GRU, Naive,"
echo "             MovingAverage, ETSformer, SCINet, FreTS"
echo "ARIMA is run separately (train_arima_all_experiments.sh)."
echo "================================================================================"
echo ""

# Common parameters
ROOT_PATH="./dataset"
DATA_PATH="normal_LOCAC_RW_LOF_SGATR_train_LOCA_test"
ENC_IN=74
DEC_IN=74
C_OUT=1
SEQ_LEN=30
TRAIN_EPOCHS=100
PATIENCE=5

# Horizon configurations: pred_len label_len name
declare -A HORIZONS
HORIZONS[short]="3 2 30s"
HORIZONS[medium]="30 15 300s"
HORIZONS[long]="60 30 600s"

LOG_DIR="NuclearResearch/logs"
TIMING_DIR="NuclearResearch/results_summary/timing"
mkdir -p "$LOG_DIR" "$TIMING_DIR"

# Timing summary file
TIMING_FILE="${TIMING_DIR}/timing_exp4_summary.csv"
echo "model,horizon,pred_len,total_seconds,total_human,status" > "$TIMING_FILE"
EXP_START_TIME=$(date +%s)

# Function to train a model
train_model() {
    local MODEL_NAME=$1
    local MODEL_ID=$2
    local PRED_LEN=$3
    local LABEL_LEN=$4
    local HORIZON_NAME=$5
    local MODEL_NUM=$6
    shift 6
    local EXTRA_ARGS="$@"

    echo "================================================================================"
    echo "Training ${MODEL_NUM}: ${MODEL_NAME} - ${HORIZON_NAME} horizon"
    echo "  Start time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================================"

    local RUN_START=$(date +%s)

    python NuclearResearch/src/run_loca_forecast.py \
        --model "$MODEL_NAME" \
        --model_id "LOCA_TPCT_exp4_multi_accident_${MODEL_ID}_pl${PRED_LEN}" \
        --root_path "$ROOT_PATH" \
        --data_path "$DATA_PATH" \
        --enc_in $ENC_IN \
        --dec_in $DEC_IN \
        --c_out $C_OUT \
        --seq_len $SEQ_LEN \
        --label_len $LABEL_LEN \
        --pred_len $PRED_LEN \
        --train_epochs $TRAIN_EPOCHS \
        --patience $PATIENCE \
        $EXTRA_ARGS \
        2>&1 | tee "${LOG_DIR}/training_exp4_${MODEL_ID}_pl${PRED_LEN}.log"

    local EXIT_CODE=${PIPESTATUS[0]}
    local RUN_END=$(date +%s)
    local RUN_ELAPSED=$((RUN_END - RUN_START))
    local RUN_MIN=$((RUN_ELAPSED / 60))
    local RUN_SEC=$((RUN_ELAPSED % 60))

    if [ $EXIT_CODE -eq 0 ]; then
        local STATUS="OK"
    else
        local STATUS="FAILED(exit=$EXIT_CODE)"
    fi

    echo "--------------------------------------------------------------------------------"
    echo "  ${MODEL_NAME} ${HORIZON_NAME} -- ${STATUS} -- ${RUN_MIN}m ${RUN_SEC}s (${RUN_ELAPSED}s total)"
    echo "  End time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "--------------------------------------------------------------------------------"

    echo "${MODEL_ID},${HORIZON_NAME},${PRED_LEN},${RUN_ELAPSED},${RUN_MIN}m${RUN_SEC}s,${STATUS}" >> "$TIMING_FILE"

    sleep 5
}

# Iterate through each horizon
for HORIZON_KEY in short medium long; do
    HORIZON_INFO=(${HORIZONS[$HORIZON_KEY]})
    PRED_LEN=${HORIZON_INFO[0]}
    LABEL_LEN=${HORIZON_INFO[1]}
    HORIZON_NAME=${HORIZON_INFO[2]}

    echo ""
    echo "################################################################################"
    echo "## STARTING HORIZON: ${HORIZON_NAME} (pred_len=${PRED_LEN}, label_len=${LABEL_LEN})"
    echo "################################################################################"
    echo ""

    # Model 1: TFT
    train_model "TemporalFusionTransformer" "TFT" $PRED_LEN $LABEL_LEN $HORIZON_NAME "Model 1/24" \
        "--d_model 128 --d_ff 512 --e_layers 2 --n_heads 4 --batch_size 16 --learning_rate 0.0001"

    # Model 2: Transformer
    train_model "Transformer" "Transformer" $PRED_LEN $LABEL_LEN $HORIZON_NAME "Model 2/24" \
        "--d_model 128 --d_ff 512 --e_layers 2 --n_heads 4 --batch_size 32"

    # Model 3: Nonstationary_Transformer
    train_model "Nonstationary_Transformer" "Nonstationary" $PRED_LEN $LABEL_LEN $HORIZON_NAME "Model 3/24" \
        "--d_model 128 --d_ff 512 --e_layers 2 --n_heads 4 --batch_size 32 --p_hidden_dims 128 128 --p_hidden_layers 2"

    # Model 4: PatchTST
    train_model "PatchTST" "PatchTST" $PRED_LEN $LABEL_LEN $HORIZON_NAME "Model 4/24" \
        "--d_model 128 --d_ff 512 --e_layers 2 --n_heads 4 --batch_size 32"

    # Model 5: Crossformer
    train_model "Crossformer" "Crossformer" $PRED_LEN $LABEL_LEN $HORIZON_NAME "Model 5/24" \
        "--d_model 128 --d_ff 512 --e_layers 2 --n_heads 4 --batch_size 32"

    # Model 6: Pyraformer
    train_model "Pyraformer" "Pyraformer" $PRED_LEN $LABEL_LEN $HORIZON_NAME "Model 6/24" \
        "--d_model 128 --d_ff 512 --e_layers 2 --n_heads 4 --batch_size 32"

    # Model 7: Reformer
    train_model "Reformer" "Reformer" $PRED_LEN $LABEL_LEN $HORIZON_NAME "Model 7/24" \
        "--d_model 128 --d_ff 512 --e_layers 2 --n_heads 4 --batch_size 32"

    # Model 8: TSMixer
    train_model "TSMixer" "TSMixer" $PRED_LEN $LABEL_LEN $HORIZON_NAME "Model 8/24" \
        "--d_model 128 --d_ff 512 --e_layers 2 --batch_size 32"

    # Model 9: iTransformer
    train_model "iTransformer" "iTransformer" $PRED_LEN $LABEL_LEN $HORIZON_NAME "Model 9/24" \
        "--d_model 512 --d_ff 512 --e_layers 3 --n_heads 4 --batch_size 16 --learning_rate 0.0005"

    # Model 10: TimesNet
    train_model "TimesNet" "TimesNet" $PRED_LEN $LABEL_LEN $HORIZON_NAME "Model 10/24" \
        "--d_model 128 --d_ff 512 --e_layers 2 --n_heads 4 --batch_size 32 --top_k 5 --num_kernels 6"

    # Model 11: TimeMixer
    train_model "TimeMixer" "TimeMixer" $PRED_LEN $LABEL_LEN $HORIZON_NAME "Model 11/24" \
        "--d_model 128 --d_ff 512 --e_layers 2 --batch_size 32 --down_sampling_layers 3 --down_sampling_window 2 --down_sampling_method avg --channel_independence 0 --decomp_method moving_avg --use_norm 1"

    # Model 12: SegRNN (adaptive seg_len based on pred_len)
    SEG_LEN_ARG="--seg_len 30"
    if [ "$PRED_LEN" -eq 3 ]; then
        SEG_LEN_ARG="--seg_len 3"
    fi
    train_model "SegRNN" "SegRNN" $PRED_LEN $LABEL_LEN $HORIZON_NAME "Model 12/24" \
        "--d_model 128 --batch_size 32 $SEG_LEN_ARG"

    # Model 13: Informer
    train_model "Informer" "Informer" $PRED_LEN $LABEL_LEN $HORIZON_NAME "Model 13/24" \
        "--d_model 128 --d_ff 512 --e_layers 2 --n_heads 4 --batch_size 32"

    # Model 14: DLinear
    train_model "DLinear" "DLinear" $PRED_LEN $LABEL_LEN $HORIZON_NAME "Model 14/24" \
        "--batch_size 32"

    # Model 15: FEDformer
    train_model "FEDformer" "FEDformer" $PRED_LEN $LABEL_LEN $HORIZON_NAME "Model 15/24" \
        "--d_model 128 --d_ff 512 --e_layers 2 --n_heads 4 --batch_size 32"

    # Model 16: Autoformer
    train_model "Autoformer" "Autoformer" $PRED_LEN $LABEL_LEN $HORIZON_NAME "Model 16/24" \
        "--d_model 128 --d_ff 512 --e_layers 2 --n_heads 4 --batch_size 32"

    # Model 17: MICN (requires label_len == seq_len)
    train_model "MICN" "MICN" $PRED_LEN $SEQ_LEN $HORIZON_NAME "Model 17/24" \
        "--d_model 128 --batch_size 32"

    # Model 18: LSTM
    train_model "LSTM" "LSTM" $PRED_LEN $LABEL_LEN $HORIZON_NAME "Model 18/24" \
        "--d_model 128 --e_layers 2 --batch_size 32"

    # Model 19: GRU
    train_model "GRU" "GRU" $PRED_LEN $LABEL_LEN $HORIZON_NAME "Model 19/24" \
        "--d_model 128 --e_layers 2 --batch_size 32"

    # Model 20: Naive (persistence baseline)
    train_model "Naive" "Naive" $PRED_LEN $LABEL_LEN $HORIZON_NAME "Model 20/24" \
        ""

    # Model 21: MovingAverage
    train_model "MovingAverage" "MovingAverage" $PRED_LEN $LABEL_LEN $HORIZON_NAME "Model 21/24" \
        ""

    # Model 22: ETSformer
    train_model "ETSformer" "ETSformer" $PRED_LEN $LABEL_LEN $HORIZON_NAME "Model 22/24" \
        "--d_model 128 --d_ff 512 --e_layers 2 --d_layers 2 --n_heads 4 --batch_size 32 --top_k 5"

    # Model 23: SCINet
    train_model "SCINet" "SCINet" $PRED_LEN $LABEL_LEN $HORIZON_NAME "Model 23/24" \
        "--d_model 128 --batch_size 32"

    # Model 24: FreTS
    train_model "FreTS" "FreTS" $PRED_LEN $LABEL_LEN $HORIZON_NAME "Model 24/24" \
        "--d_model 128 --d_ff 512 --e_layers 2 --n_heads 4 --batch_size 32"

    echo ""
    echo "################################################################################"
    echo "## COMPLETED HORIZON: ${HORIZON_NAME}"
    echo "################################################################################"
    echo ""
done

EXP_END_TIME=$(date +%s)
EXP_TOTAL=$((EXP_END_TIME - EXP_START_TIME))
EXP_HOURS=$((EXP_TOTAL / 3600))
EXP_MINS=$(( (EXP_TOTAL % 3600) / 60 ))

echo ""
echo "================================================================================"
echo "EXPERIMENT 4 TRAINING COMPLETE - ALL HORIZONS"
echo "================================================================================"
echo ""
echo "Total experiment wall time: ${EXP_HOURS}h ${EXP_MINS}m (${EXP_TOTAL}s)"
echo ""
echo "Per-model timing summary:"
echo "--------------------------------------------------------------------------------"
column -t -s',' "$TIMING_FILE"
echo "--------------------------------------------------------------------------------"
echo ""
echo "Timing CSV: $TIMING_FILE"
echo "Results:    ./results/long_term_forecast_LOCA_TPCT_exp4_multi_accident_*/"
echo "Logs:       ${LOG_DIR}/training_exp4_*.log"
echo ""
echo "Next steps:"
echo "  1. Baselines:       bash NuclearResearch/scripts/multi_horizon_final/train_arima_all_experiments.sh"
echo "                     python NuclearResearch/src/baselines/run_naive_baselines.py"
echo "  2. Metrics/plots:   python NuclearResearch/src/result_process/visualization/visualize_experiment.py --exp exp4"
echo "================================================================================"
