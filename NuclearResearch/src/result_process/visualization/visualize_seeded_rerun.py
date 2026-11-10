#!/usr/bin/env python3
"""
Visualization Script for Seeded Rerun Experiments

Runs the same rolling-window visualization pipeline on seeded rerun results
(pred.npy/true.npy) to produce metrics_summary.txt files with per-scenario
and average MAE/RMSE, consistent with the original visualization methodology.

Outputs to NuclearResearch/results/Visualization_seeded/ (separate from
the original Visualization/ directory to preserve existing results).

Usage:
    python visualize_seeded_rerun.py --exp exp0        # Run exp0 only
    python visualize_seeded_rerun.py --exp exp0 exp4   # Run both experiments
    python visualize_seeded_rerun.py --exp all         # Run all
"""

import argparse
import subprocess
import os
import sys
from pathlib import Path
from typing import Tuple

# Derive project root from script location
# Script is at: <PROJECT_ROOT>/NuclearResearch/src/result_process/visualization/
SCRIPT_DIR = Path(__file__).parent
RESULT_PROCESS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = Path(__file__).resolve().parents[4]

# ============================================================================
# CONFIGURATION
# ============================================================================

EXPERIMENTS = {
    'exp0': {
        'name': 'LOCA-only (Seeded Rerun)',
        'pattern': 'exp0_loca_only',
        'description': '75 LOCA training - 5 seeds'
    },
    'exp4': {
        'name': 'Multi-Accident (Seeded Rerun)',
        'pattern': 'exp4_multi_accident',
        'description': '75x5 accidents + 20 Normal - 5 seeds'
    }
}

MODELS = ['SCINet', 'Crossformer', 'Pyraformer', 'LSTM', 'GRU', 'SegRNN']

SEEDS = [42, 123, 456, 789, 1024]

HORIZONS = [
    {'name': '30s', 'pl': 3, 'script': 'visualize_30s_ahead_flexible.py'},
    {'name': '300s', 'pl': 30, 'script': 'visualize_300s_ahead_flexible.py'},
    {'name': '600s', 'pl': 60, 'script': 'visualize_600s_ahead_flexible.py'}
]

RESULTS_ROOT = PROJECT_ROOT / 'results'
CENTRALIZED_OUTPUT_ROOT = PROJECT_ROOT / 'NuclearResearch' / 'results' / 'Visualization_seeded'
SEQ_LEN = 30


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def find_result_folder(exp_pattern: str, model: str, pred_len: int, seed: int) -> Path:
    """
    Find result folder for a seeded run.

    Seeded results end with _seed{N} instead of _0.

    Args:
        exp_pattern: Experiment pattern (e.g., 'exp0_loca_only')
        model: Model name (e.g., 'SCINet')
        pred_len: Prediction length (3, 30, or 60)
        seed: Random seed value

    Returns:
        Path to result folder, or None if not found
    """
    pattern = '*{}_{}_pl{}_*_seed{}'.format(exp_pattern, model, pred_len, seed)
    matches = list(RESULTS_ROOT.glob(pattern))
    if matches:
        return matches[0]
    return None


def run_visualization(result_folder: Path, viz_script: str, seq_len: int, pred_len: int,
                     output_dir: Path = None, exp_name: str = "",
                     model_name: str = "") -> Tuple[bool, str]:
    """
    Run visualization script for a single result folder.

    Args:
        result_folder: Path to result folder containing pred.npy/true.npy
        viz_script: Name of visualization script
        seq_len: Sequence length
        pred_len: Prediction length
        output_dir: Custom output directory
        exp_name: Experiment name for plot titles
        model_name: Model name for plot titles and file naming

    Returns:
        (success, message) tuple
    """
    script_path = RESULT_PROCESS_DIR / viz_script

    if not script_path.exists():
        return False, 'Script not found: {}'.format(script_path)

    env = os.environ.copy()
    env['MPLBACKEND'] = 'Agg'

    cmd = [
        'python', str(script_path),
        '--result_folder', str(result_folder),
        '--seq_len', str(seq_len),
        '--pred_len', str(pred_len)
    ]

    if output_dir:
        cmd.extend(['--output_dir', str(output_dir)])
    if exp_name:
        cmd.extend(['--exp_name', exp_name])
    if model_name:
        cmd.extend(['--model_name', model_name])

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'Average MAE' in line:
                    return True, line.strip()
            return True, 'Visualization completed'
        else:
            return False, 'Error: {}'.format(result.stderr[:200])

    except subprocess.TimeoutExpired:
        return False, 'Timeout (>5 minutes)'
    except Exception as e:
        return False, 'Exception: {}'.format(str(e)[:200])


# ============================================================================
# MAIN LOGIC
# ============================================================================

def visualize_seeded_experiment(exp_id: str, verbose: bool = True) -> dict:
    """
    Run visualizations for all models, seeds, and horizons in an experiment.

    Args:
        exp_id: Experiment ID (e.g., 'exp0')
        verbose: Print detailed progress

    Returns:
        Dictionary with results summary
    """
    exp_config = EXPERIMENTS[exp_id]
    total_runs = len(MODELS) * len(SEEDS) * len(HORIZONS)
    current_run = 0
    results = {'success': 0, 'skipped': 0, 'failed': 0}

    print('=' * 80)
    print('SEEDED RERUN VISUALIZATION: {} - {}'.format(exp_id.upper(), exp_config['name']))
    print('Description: {}'.format(exp_config['description']))
    print('Models: {}'.format(', '.join(MODELS)))
    print('Seeds: {}'.format(SEEDS))
    print('Total runs: {}'.format(total_runs))
    print('=' * 80)
    print()

    for model in MODELS:
        if verbose:
            print('\n{}'.format('_' * 80))
            print('MODEL: {}'.format(model))
            print('{}'.format('_' * 80))

        for seed in SEEDS:
            for horizon in HORIZONS:
                current_run += 1

                result_folder = find_result_folder(
                    exp_config['pattern'], model, horizon['pl'], seed
                )

                if result_folder is None:
                    if verbose:
                        print('  [{:3d}/{}] {} seed={} - SKIP: Folder not found'.format(
                            current_run, total_runs, horizon['name'], seed))
                    results['skipped'] += 1
                    continue

                # Output: Visualization_seeded/{exp}/{horizon}_ahead/seed{seed}/
                centralized_output = (
                    CENTRALIZED_OUTPUT_ROOT / exp_id
                    / '{}_ahead'.format(horizon['name'])
                    / 'seed{}'.format(seed)
                )

                if verbose:
                    print('  [{:3d}/{}] {} seed={} - Running...'.format(
                        current_run, total_runs, horizon['name'], seed), end='', flush=True)

                success, message = run_visualization(
                    result_folder,
                    horizon['script'],
                    SEQ_LEN,
                    horizon['pl'],
                    output_dir=centralized_output,
                    exp_name='{}: {} seed={}'.format(exp_id.upper(), exp_config['name'], seed),
                    model_name=model
                )

                if success:
                    if verbose:
                        print(' OK - {}'.format(message))
                    results['success'] += 1
                else:
                    if verbose:
                        print(' FAIL - {}'.format(message))
                    results['failed'] += 1

    print()
    print('=' * 80)
    print('{} SEEDED VISUALIZATION COMPLETE'.format(exp_id.upper()))
    print('=' * 80)
    print('  Success: {:3d}/{}'.format(results['success'], total_runs))
    print('  Skipped: {:3d}/{}'.format(results['skipped'], total_runs))
    print('  Failed:  {:3d}/{}'.format(results['failed'], total_runs))
    print('=' * 80)
    print()

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Run visualizations for seeded rerun experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_seeded_rerun.py --exp exp0           # Run exp0 only
  python visualize_seeded_rerun.py --exp exp0 exp4      # Run both
  python visualize_seeded_rerun.py --exp all            # Run all
  python visualize_seeded_rerun.py --exp exp0 --quiet   # Minimal output
        """
    )

    parser.add_argument(
        '--exp',
        nargs='+',
        required=True,
        choices=list(EXPERIMENTS.keys()) + ['all'],
        help='Experiment(s) to visualize'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Minimal output (only show summaries)'
    )

    args = parser.parse_args()

    if 'all' in args.exp:
        experiments = list(EXPERIMENTS.keys())
    else:
        experiments = args.exp

    verbose = not args.quiet
    all_results = {}

    for exp_id in experiments:
        exp_results = visualize_seeded_experiment(exp_id, verbose=verbose)
        all_results[exp_id] = exp_results

    if len(experiments) > 1:
        print('\n' + '=' * 80)
        print('FINAL SUMMARY - ALL SEEDED EXPERIMENTS')
        print('=' * 80)
        for exp_id, exp_results in all_results.items():
            total = sum(exp_results.values())
            print('{:5s}: {} success, {} skipped, {} failed (total: {})'.format(
                exp_id, exp_results['success'], exp_results['skipped'],
                exp_results['failed'], total))
        print('=' * 80)


if __name__ == '__main__':
    main()
