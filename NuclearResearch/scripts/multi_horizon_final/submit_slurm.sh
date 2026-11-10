#!/bin/bash
# Submit one command as a Slurm batch job (for use as LAUNCHER), e.g.
#   LAUNCHER="bash NuclearResearch/scripts/multi_horizon_final/submit_slurm.sh" SBATCH_OPTS="--gres=gpu:1" \
#       bash NuclearResearch/scripts/multi_horizon_final/train_exp4_seeded_rerun.sh

set -euo pipefail

if [ "$#" -eq 0 ]; then
    echo "Usage: submit_slurm.sh <command> [args...]" >&2
    exit 1
fi

mkdir -p NuclearResearch/logs/slurm
CMD=$(printf '%q ' "$@")
# shellcheck disable=SC2086
sbatch ${SBATCH_OPTS:-} --chdir "$(pwd)" --output "NuclearResearch/logs/slurm/%j.out" --wrap "$CMD"
