#!/bin/bash
# Seeded reruns: Experiment 0 (LOCA-only)
# 6 models (SCINet, Crossformer, Pyraformer, LSTM, GRU, SegRNN) x 5 seeds x 3 horizons = 90 runs
#
# Usage: bash NuclearResearch/scripts/multi_horizon_final/train_exp0_seeded_rerun.sh
# Optional: LAUNCHER="<prefix>" runs each job through a scheduler (see submit_slurm.sh).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
[ -f "$ROOT/NuclearResearch/src/run_loca_forecast.py" ] || ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" && [ -f NuclearResearch/src/run_loca_forecast.py ] || { echo "Run from the repository root" >&2; exit 1; }

LAUNCHER="${LAUNCHER:-}"

# Seeds for the reproducibility study
SEEDS=(42 123 456 789 1024)

# Common parameters
ROOT_PATH="./dataset"
DATA_PATH="LOCA_only_train_LOCA_test"
ENC_IN=74
DEC_IN=74
C_OUT=1
SEQ_LEN=30
TRAIN_EPOCHS=100
PATIENCE=5

# Experiment tag
EXP_TAG="exp0_loca_only"

LOG_DIR="NuclearResearch/logs"
mkdir -p "$LOG_DIR"

JOB_COUNT=0

# Train and test one (model, horizon, seed) configuration
run_model() {
    local MODEL_NAME=$1
    local MODEL_ID=$2
    local PRED_LEN=$3
    local LABEL_LEN=$4
    local SEED=$5
    shift 5
    local EXTRA_ARGS="$@"

    JOB_COUNT=$((JOB_COUNT + 1))
    echo "[${JOB_COUNT}/90] ${MODEL_NAME} pl${PRED_LEN} seed=${SEED}"

    local CMD=(python NuclearResearch/src/run_loca_forecast.py
        --model "$MODEL_NAME"
        --model_id "LOCA_TPCT_${EXP_TAG}_${MODEL_ID}_pl${PRED_LEN}"
        --root_path "$ROOT_PATH"
        --data_path "$DATA_PATH"
        --enc_in $ENC_IN
        --dec_in $DEC_IN
        --c_out $C_OUT
        --seq_len $SEQ_LEN
        --label_len $LABEL_LEN
        --pred_len $PRED_LEN
        --train_epochs $TRAIN_EPOCHS
        --patience $PATIENCE
        --seed $SEED
        $EXTRA_ARGS)

    if [ -n "$LAUNCHER" ]; then
        $LAUNCHER "${CMD[@]}"
    else
        "${CMD[@]}" 2>&1 | tee "${LOG_DIR}/seeded_exp0_${MODEL_ID}_pl${PRED_LEN}_seed${SEED}.log"
    fi
}

echo "================================================================================"
echo "Seed-sensitivity reruns: Experiment 0 (LOCA-only, 75 training scenarios)"
echo "Models: SCINet, Crossformer, Pyraformer, LSTM, GRU, SegRNN"
echo "Seeds:  ${SEEDS[*]}"
echo "Total:  90 runs (6 models x 5 seeds x 3 horizons)"
echo "================================================================================"

# Horizon configurations: pred_len label_len
declare -A HORIZONS
HORIZONS[short]="3 2"
HORIZONS[medium]="30 15"
HORIZONS[long]="60 30"

for HORIZON_KEY in short medium long; do
    HORIZON_PARAMS=(${HORIZONS[$HORIZON_KEY]})
    PRED_LEN=${HORIZON_PARAMS[0]}
    LABEL_LEN=${HORIZON_PARAMS[1]}

    for SEED in "${SEEDS[@]}"; do

        # SCINet
        run_model "SCINet" "SCINet" $PRED_LEN $LABEL_LEN $SEED \
            "--d_model 128 --batch_size 32"

        # Crossformer
        run_model "Crossformer" "Crossformer" $PRED_LEN $LABEL_LEN $SEED \
            "--d_model 128 --d_ff 512 --e_layers 2 --n_heads 4 --batch_size 32"

        # Pyraformer (batch_size 33 at pred_len 60)
        PYRA_BATCH=32
        if [ "$PRED_LEN" -eq 60 ]; then
            PYRA_BATCH=33
        fi
        run_model "Pyraformer" "Pyraformer" $PRED_LEN $LABEL_LEN $SEED \
            "--d_model 128 --d_ff 512 --e_layers 2 --n_heads 4 --batch_size $PYRA_BATCH"

        # LSTM
        run_model "LSTM" "LSTM" $PRED_LEN $LABEL_LEN $SEED \
            "--d_model 128 --e_layers 2 --batch_size 32"

        # GRU
        run_model "GRU" "GRU" $PRED_LEN $LABEL_LEN $SEED \
            "--d_model 128 --e_layers 2 --batch_size 32"

        # SegRNN (seg_len 3 for the 30s horizon, else 30)
        SEG_LEN=30
        if [ "$PRED_LEN" -eq 3 ]; then
            SEG_LEN=3
        fi
        run_model "SegRNN" "SegRNN" $PRED_LEN $LABEL_LEN $SEED \
            "--d_model 128 --batch_size 32 --seg_len $SEG_LEN"

    done
done

echo "================================================================================"
echo "Seeded runs finished (or submitted): ${JOB_COUNT}"
echo "Next steps (after all runs complete):"
echo "  python NuclearResearch/src/result_process/visualization/visualize_seeded_rerun.py --exp exp0"
echo "  python NuclearResearch/src/result_process/result_extraction/extract_seeded_metrics.py"
echo "  python NuclearResearch/src/result_process/visualization/visualize_seeded_uncertainty.py --exp exp0"
echo "================================================================================"
