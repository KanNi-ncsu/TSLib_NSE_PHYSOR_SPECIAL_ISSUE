#!/bin/bash
# Inference benchmark (Experiment 4 checkpoints): 24 specialist models, fine-tuned
# foundation models and ARIMA at all horizons. Uses saved checkpoints; no retraining.
# Zero-shot foundation models: benchmark_inference_zeroshot_exp4.sh (appends to the same CSV).
#
# Usage: bash NuclearResearch/scripts/multi_horizon_final/benchmark_inference_exp4.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
[ -f "$ROOT/NuclearResearch/src/run_loca_forecast.py" ] || ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" && [ -f NuclearResearch/src/run_loca_forecast.py ] || { echo "Run from the repository root" >&2; exit 1; }

# Common parameters (identical to train_exp4_all_horizons.sh)
ROOT_PATH="./dataset"
DATA_PATH="normal_LOCAC_RW_LOF_SGATR_train_LOCA_test"
ENC_IN=74
DEC_IN=74
C_OUT=1
SEQ_LEN=30

# Benchmark parameters
N_WARMUP=3
N_RUNS=10

# Horizon configurations (identical to training)
declare -A HORIZONS
HORIZONS[short]="3 2 30s"
HORIZONS[medium]="30 15 300s"
HORIZONS[long]="60 30 600s"

# Output timing CSV
mkdir -p NuclearResearch/results_summary/timing
TIMING_CSV="NuclearResearch/results_summary/timing/inference_timing_exp4_summary.csv"

# Remove old CSV to start fresh (header written by Python script on first run)
rm -f "$TIMING_CSV"

EXP_START_TIME=$(date +%s)

# Function to benchmark a single model
benchmark_model() {
    local MODEL_NAME=$1
    local MODEL_ID=$2
    local PRED_LEN=$3
    local LABEL_LEN=$4
    local HORIZON_NAME=$5
    local MODEL_NUM=$6
    shift 6
    local EXTRA_ARGS="$@"

    echo "--------------------------------------------------------------------------------"
    echo "  Benchmarking ${MODEL_NUM}: ${MODEL_NAME} - ${HORIZON_NAME} (pred_len=${PRED_LEN})"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "--------------------------------------------------------------------------------"

    python NuclearResearch/src/run_inference_benchmark.py \
        --benchmark_mode dl \
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
        --n_warmup $N_WARMUP \
        --n_runs $N_RUNS \
        --timing_csv "$TIMING_CSV" \
        $EXTRA_ARGS

    local EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "  FAILED: ${MODEL_NAME} ${HORIZON_NAME} (exit code ${EXIT_CODE})"
    fi

    echo ""
}

# Iterate through each horizon
for HORIZON_KEY in short medium long; do
    HORIZON_INFO=(${HORIZONS[$HORIZON_KEY]})
    PRED_LEN=${HORIZON_INFO[0]}
    LABEL_LEN=${HORIZON_INFO[1]}
    HORIZON_NAME=${HORIZON_INFO[2]}

    echo ""
    echo "################################################################################"
    echo "## BENCHMARK HORIZON: ${HORIZON_NAME} (pred_len=${PRED_LEN}, label_len=${LABEL_LEN})"
    echo "################################################################################"
    echo ""

    # Model 1: TFT
    benchmark_model "TemporalFusionTransformer" "TFT" $PRED_LEN $LABEL_LEN $HORIZON_NAME "Model 1/24" \
        "--d_model 128 --d_ff 512 --e_layers 2 --n_heads 4 --batch_size 16 --learning_rate 0.0001"

    # Model 2: Transformer
    benchmark_model "Transformer" "Transformer" $PRED_LEN $LABEL_LEN $HORIZON_NAME "Model 2/24" \
        "--d_model 128 --d_ff 512 --e_layers 2 --n_heads 4 --batch_size 32"

    # Model 3: Nonstationary_Transformer
    benchmark_model "Nonstationary_Transformer" "Nonstationary" $PRED_LEN $LABEL_LEN $HORIZON_NAME "Model 3/24" \
        "--d_model 128 --d_ff 512 --e_layers 2 --n_heads 4 --batch_size 32 --p_hidden_dims 128 128 --p_hidden_layers 2"

    # Model 4: PatchTST
    benchmark_model "PatchTST" "PatchTST" $PRED_LEN $LABEL_LEN $HORIZON_NAME "Model 4/24" \
        "--d_model 128 --d_ff 512 --e_layers 2 --n_heads 4 --batch_size 32"

    # Model 5: Crossformer
    benchmark_model "Crossformer" "Crossformer" $PRED_LEN $LABEL_LEN $HORIZON_NAME "Model 5/24" \
        "--d_model 128 --d_ff 512 --e_layers 2 --n_heads 4 --batch_size 32"

    # Model 6: Pyraformer
    benchmark_model "Pyraformer" "Pyraformer" $PRED_LEN $LABEL_LEN $HORIZON_NAME "Model 6/24" \
        "--d_model 128 --d_ff 512 --e_layers 2 --n_heads 4 --batch_size 32"

    # Model 7: Reformer
    benchmark_model "Reformer" "Reformer" $PRED_LEN $LABEL_LEN $HORIZON_NAME "Model 7/24" \
        "--d_model 128 --d_ff 512 --e_layers 2 --n_heads 4 --batch_size 32"

    # Model 8: TSMixer
    benchmark_model "TSMixer" "TSMixer" $PRED_LEN $LABEL_LEN $HORIZON_NAME "Model 8/24" \
        "--d_model 128 --d_ff 512 --e_layers 2 --batch_size 32"

    # Model 9: iTransformer
    benchmark_model "iTransformer" "iTransformer" $PRED_LEN $LABEL_LEN $HORIZON_NAME "Model 9/24" \
        "--d_model 512 --d_ff 512 --e_layers 3 --n_heads 4 --batch_size 16 --learning_rate 0.0005"

    # Model 10: TimesNet
    benchmark_model "TimesNet" "TimesNet" $PRED_LEN $LABEL_LEN $HORIZON_NAME "Model 10/24" \
        "--d_model 128 --d_ff 512 --e_layers 2 --n_heads 4 --batch_size 32 --top_k 5 --num_kernels 6"

    # Model 11: TimeMixer
    benchmark_model "TimeMixer" "TimeMixer" $PRED_LEN $LABEL_LEN $HORIZON_NAME "Model 11/24" \
        "--d_model 128 --d_ff 512 --e_layers 2 --batch_size 32 --down_sampling_layers 3 --down_sampling_window 2 --down_sampling_method avg --channel_independence 0 --decomp_method moving_avg --use_norm 1"

    # Model 12: SegRNN (adaptive seg_len based on pred_len)
    SEG_LEN_ARG="--seg_len 30"
    if [ "$PRED_LEN" -eq 3 ]; then
        SEG_LEN_ARG="--seg_len 3"
    fi
    benchmark_model "SegRNN" "SegRNN" $PRED_LEN $LABEL_LEN $HORIZON_NAME "Model 12/24" \
        "--d_model 128 --batch_size 32 $SEG_LEN_ARG"

    # Model 13: Informer
    benchmark_model "Informer" "Informer" $PRED_LEN $LABEL_LEN $HORIZON_NAME "Model 13/24" \
        "--d_model 128 --d_ff 512 --e_layers 2 --n_heads 4 --batch_size 32"

    # Model 14: DLinear
    benchmark_model "DLinear" "DLinear" $PRED_LEN $LABEL_LEN $HORIZON_NAME "Model 14/24" \
        "--batch_size 32"

    # Model 15: FEDformer
    benchmark_model "FEDformer" "FEDformer" $PRED_LEN $LABEL_LEN $HORIZON_NAME "Model 15/24" \
        "--d_model 128 --d_ff 512 --e_layers 2 --n_heads 4 --batch_size 32"

    # Model 16: Autoformer
    benchmark_model "Autoformer" "Autoformer" $PRED_LEN $LABEL_LEN $HORIZON_NAME "Model 16/24" \
        "--d_model 128 --d_ff 512 --e_layers 2 --n_heads 4 --batch_size 32"

    # Model 17: MICN (requires label_len == seq_len)
    benchmark_model "MICN" "MICN" $PRED_LEN $SEQ_LEN $HORIZON_NAME "Model 17/24" \
        "--d_model 128 --batch_size 32"

    # Model 18: LSTM
    benchmark_model "LSTM" "LSTM" $PRED_LEN $LABEL_LEN $HORIZON_NAME "Model 18/24" \
        "--d_model 128 --e_layers 2 --batch_size 32"

    # Model 19: GRU
    benchmark_model "GRU" "GRU" $PRED_LEN $LABEL_LEN $HORIZON_NAME "Model 19/24" \
        "--d_model 128 --e_layers 2 --batch_size 32"

    # Model 20: Naive (persistence baseline)
    benchmark_model "Naive" "Naive" $PRED_LEN $LABEL_LEN $HORIZON_NAME "Model 20/24" \
        ""

    # Model 21: MovingAverage
    benchmark_model "MovingAverage" "MovingAverage" $PRED_LEN $LABEL_LEN $HORIZON_NAME "Model 21/24" \
        ""

    # Model 22: ETSformer
    benchmark_model "ETSformer" "ETSformer" $PRED_LEN $LABEL_LEN $HORIZON_NAME "Model 22/24" \
        "--d_model 128 --d_ff 512 --e_layers 2 --d_layers 2 --n_heads 4 --batch_size 32 --top_k 5"

    # Model 23: SCINet
    benchmark_model "SCINet" "SCINet" $PRED_LEN $LABEL_LEN $HORIZON_NAME "Model 23/24" \
        "--d_model 128 --batch_size 32"

    # Model 24: FreTS
    benchmark_model "FreTS" "FreTS" $PRED_LEN $LABEL_LEN $HORIZON_NAME "Model 24/24" \
        "--d_model 128 --d_ff 512 --e_layers 2 --n_heads 4 --batch_size 32"


    echo ""
    echo "################################################################################"
    echo "## COMPLETED HORIZON: ${HORIZON_NAME}"
    echo "################################################################################"
    echo ""
done

# ==============================================================================
# Fine-Tuned Foundation Models + ARIMA
# ==============================================================================

echo ""
echo "================================================================================"
echo "BENCHMARKING FINE-TUNED FOUNDATION MODELS + ARIMA"
echo "================================================================================"
echo ""

for HORIZON_KEY in short medium long; do
    HORIZON_INFO=(${HORIZONS[$HORIZON_KEY]})
    PRED_LEN=${HORIZON_INFO[0]}
    LABEL_LEN=${HORIZON_INFO[1]}
    HORIZON_NAME=${HORIZON_INFO[2]}

    echo "--- ${HORIZON_NAME} (pred_len=${PRED_LEN}) ---"

    # Chronos2_FT (fine-tuned multivariate)
    echo "  Benchmarking Chronos2_FT - ${HORIZON_NAME}"
    python NuclearResearch/src/run_inference_benchmark.py \
        --benchmark_mode foundation \
        --model Chronos2_FT \
        --model_id "LOCA_TPCT_exp4_finetuned_Chronos2_FT_pl${PRED_LEN}" \
        --task_name zero_shot_forecast \
        --features MS --target TPCT --des zeroshot \
        --finetuned_model_path "checkpoints_foundation/finetuned/chronos2_multivar_exp4_pl${PRED_LEN}/finetuned-ckpt" \
        --root_path "$ROOT_PATH" --data_path "$DATA_PATH" \
        --enc_in $ENC_IN --dec_in $DEC_IN --c_out $C_OUT \
        --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $PRED_LEN \
        --batch_size 32 \
        --n_warmup $N_WARMUP --n_runs $N_RUNS \
        --timing_csv "$TIMING_CSV"
    echo ""

    # Chronos_FT (fine-tuned univariate)
    echo "  Benchmarking Chronos_FT - ${HORIZON_NAME}"
    python NuclearResearch/src/run_inference_benchmark.py \
        --benchmark_mode foundation \
        --model Chronos_FT \
        --model_id "LOCA_TPCT_exp4_finetuned_Chronos_FT_pl${PRED_LEN}" \
        --task_name zero_shot_forecast \
        --features MS --target TPCT --des zeroshot \
        --finetuned_model_path "checkpoints_foundation/finetuned/chronos_bolt_exp4_pl${PRED_LEN}" \
        --root_path "$ROOT_PATH" --data_path "$DATA_PATH" \
        --enc_in $ENC_IN --dec_in $DEC_IN --c_out $C_OUT \
        --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $PRED_LEN \
        --batch_size 32 \
        --n_warmup $N_WARMUP --n_runs $N_RUNS \
        --timing_csv "$TIMING_CSV"
    echo ""

    # ARIMA (statistical baseline, CPU-only)
    echo "  Benchmarking ARIMA - ${HORIZON_NAME}"
    python NuclearResearch/src/run_inference_benchmark.py \
        --benchmark_mode arima \
        --model ARIMA \
        --pred_len $PRED_LEN --seq_len $SEQ_LEN \
        --root_path "$ROOT_PATH" --data_path "$DATA_PATH" \
        --n_warmup $N_WARMUP --n_runs $N_RUNS \
        --timing_csv "$TIMING_CSV"
    echo ""
done

EXP_END_TIME=$(date +%s)
EXP_TOTAL=$((EXP_END_TIME - EXP_START_TIME))
EXP_MIN=$((EXP_TOTAL / 60))
EXP_SEC=$((EXP_TOTAL % 60))

echo ""
echo "================================================================================"
echo "INFERENCE BENCHMARK COMPLETE - EXPERIMENT 4"
echo "================================================================================"
echo ""
echo "Total benchmark time: ${EXP_MIN}m ${EXP_SEC}s (${EXP_TOTAL}s)"
echo ""
echo "Results saved to: $TIMING_CSV"
echo ""
echo "Timing CSV contents:"
echo "--------------------------------------------------------------------------------"
column -t -s',' "$TIMING_CSV" 2>/dev/null || cat "$TIMING_CSV"
echo "--------------------------------------------------------------------------------"
echo ""
