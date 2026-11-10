#!/bin/bash
# Fine-tune Chronos-2 (Chronos2_FT) and Chronos-Bolt (Chronos_FT) on the Experiment 4
# training data at all horizons (6 runs). Evaluate with eval_finetuned_foundation_exp4.sh.
#
# Usage: bash NuclearResearch/scripts/multi_horizon_final/finetune_foundation_exp4.sh
# Optional: LAUNCHER="<prefix>" runs each job through a scheduler (see submit_slurm.sh).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
[ -f "$ROOT/NuclearResearch/src/run_loca_forecast.py" ] || ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" && [ -f NuclearResearch/src/run_loca_forecast.py ] || { echo "Run from the repository root" >&2; exit 1; }

LAUNCHER="${LAUNCHER:-}"
DATASET_DIR="./dataset/normal_LOCAC_RW_LOF_SGATR_train_LOCA_test"
OUTPUT_BASE="./checkpoints_foundation/finetuned"
LOG_DIR="NuclearResearch/logs"
mkdir -p "$LOG_DIR"

run() {
    local LOG_FILE=$1
    shift
    if [ -n "$LAUNCHER" ]; then
        $LAUNCHER "$@"
    else
        "$@" 2>&1 | tee "$LOG_FILE"
    fi
}

# Chronos-2 (multivariate) for pred_len 3, 30, 60
for PL in 3 30 60; do
    run "${LOG_DIR}/finetune_chronos2_exp4_pl${PL}.log" \
        python NuclearResearch/src/finetune_chronos2.py \
            --dataset_dir "$DATASET_DIR" \
            --target TPCT \
            --model_name "amazon/chronos-2" \
            --pred_len $PL \
            --context_length 30 \
            --learning_rate 1e-5 \
            --num_steps 8000 \
            --batch_size 32 \
            --output_base "$OUTPUT_BASE"
done

# Chronos-Bolt (univariate) for pred_len 3, 30, 60
for PL in 3 30 60; do
    run "${LOG_DIR}/finetune_chronos_bolt_exp4_pl${PL}.log" \
        python NuclearResearch/src/finetune_chronos_bolt.py \
            --dataset_dir "$DATASET_DIR" \
            --target TPCT \
            --model_name "amazon/chronos-bolt-base" \
            --pred_len $PL \
            --seq_len 30 \
            --learning_rate 5e-5 \
            --num_epochs 40 \
            --batch_size 64 \
            --patience 5 \
            --output_base "$OUTPUT_BASE"
done

echo "Fine-tuning runs finished (or submitted): 3x Chronos-2, 3x Chronos-Bolt."
