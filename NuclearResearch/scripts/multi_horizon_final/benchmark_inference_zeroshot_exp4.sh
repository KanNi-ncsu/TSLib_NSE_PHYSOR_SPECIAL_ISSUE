#!/bin/bash
# Inference benchmark: zero-shot foundation models (TimesFM, Chronos2, Chronos) at all
# horizons. Appends rows to the CSV written by benchmark_inference_exp4.sh.
#
# Usage: bash NuclearResearch/scripts/multi_horizon_final/benchmark_inference_zeroshot_exp4.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
[ -f "$ROOT/NuclearResearch/src/run_loca_forecast.py" ] || ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" && [ -f NuclearResearch/src/run_loca_forecast.py ] || { echo "Run from the repository root" >&2; exit 1; }

ROOT_PATH="./dataset"
DATA_PATH="normal_LOCAC_RW_LOF_SGATR_train_LOCA_test"
ENC_IN=74
DEC_IN=74
C_OUT=1
SEQ_LEN=30

N_WARMUP=3
N_RUNS=10

mkdir -p NuclearResearch/results_summary/timing
TIMING_CSV="NuclearResearch/results_summary/timing/inference_timing_exp4_summary.csv"

declare -A HORIZONS
HORIZONS[short]="3 2 30s"
HORIZONS[medium]="30 15 300s"
HORIZONS[long]="60 30 600s"

for MODEL_NAME in TimesFM Chronos2 Chronos; do
    for HORIZON_KEY in short medium long; do
        HORIZON_INFO=(${HORIZONS[$HORIZON_KEY]})
        PRED_LEN=${HORIZON_INFO[0]}
        LABEL_LEN=${HORIZON_INFO[1]}
        HORIZON_NAME=${HORIZON_INFO[2]}

        echo "  Benchmarking ${MODEL_NAME} - ${HORIZON_NAME}"
        python NuclearResearch/src/run_inference_benchmark.py \
            --benchmark_mode foundation \
            --task_name zero_shot_forecast \
            --model "$MODEL_NAME" \
            --model_id "LOCA_TPCT_exp4_foundation_${MODEL_NAME}_pl${PRED_LEN}" \
            --root_path "$ROOT_PATH" --data_path "$DATA_PATH" \
            --data nuclear_accident \
            --enc_in $ENC_IN --dec_in $DEC_IN --c_out $C_OUT \
            --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $PRED_LEN \
            --features MS --target TPCT \
            --batch_size 8 \
            --des zeroshot \
            --n_warmup $N_WARMUP --n_runs $N_RUNS \
            --timing_csv "$TIMING_CSV"
    done
done

echo "Results appended to: $TIMING_CSV"
