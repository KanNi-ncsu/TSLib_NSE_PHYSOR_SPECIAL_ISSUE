#!/usr/bin/env python3
"""
Multi-Seed Uncertainty Overlay Visualization

For each (experiment, model, horizon, scenario), overlays all 5 seeds'
prediction curves on a single plot to demonstrate prediction stability
across random initializations.

Plot elements:
  - Blue solid: Actual TPCT
  - Light red shaded band: min-max range across 5 seeds
  - Thin red lines: individual seed predictions (semi-transparent)
  - Bold red dashed: mean prediction across seeds
  - Title shows MAE/RMSE as mean +/- std

Usage:
    python visualize_seeded_uncertainty.py --exp exp0 --model SCINet --horizon 300s
    python visualize_seeded_uncertainty.py --exp all   # all experiments, models, horizons
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Derive project root from script location
# Script is at: <PROJECT_ROOT>/NuclearResearch/src/result_process/visualization/
PROJECT_ROOT = Path(__file__).resolve().parents[4]

# ============================================================================
# CONFIGURATION
# ============================================================================

EXPERIMENTS = {
    'exp0': {
        'name': 'LOCA-only',
        'pattern': 'exp0_loca_only',
    },
    'exp4': {
        'name': 'Multi-Accident',
        'pattern': 'exp4_multi_accident',
    }
}

MODELS = ['SCINet', 'Crossformer', 'Pyraformer', 'LSTM', 'GRU', 'SegRNN']

SEEDS = [42, 123, 456, 789, 1024]

HORIZON_CONFIG = {
    '30s':  {'pl': 3,  'extract_idx': 2,  'label': '30s-ahead (3-step)'},
    '300s': {'pl': 30, 'extract_idx': 29, 'label': '300s-ahead (30-step)'},
    '600s': {'pl': 60, 'extract_idx': 59, 'label': '600s-ahead (60-step)'},
}

RESULTS_ROOT = PROJECT_ROOT / 'results'
OUTPUT_ROOT = PROJECT_ROOT / 'NuclearResearch' / 'results' / 'Visualization_seeded'
RAW_DATA_ROOT = PROJECT_ROOT / 'NuclearResearch' / 'Raw_data' / 'DONOTMODIFY' / 'Operation_csv_data' / 'LOCA'
DATASET_ROOT = PROJECT_ROOT / 'dataset'
SEQ_LEN = 30


# ============================================================================
# DATA LOADING
# ============================================================================

def get_metadata_path(exp_pattern):
    """Get metadata path based on experiment pattern."""
    if 'exp4' in exp_pattern or 'multi_accident' in exp_pattern:
        return DATASET_ROOT / 'normal_LOCAC_RW_LOF_SGATR_train_LOCA_test' / 'split_metadata_lhs.json'
    elif 'exp0' in exp_pattern or 'loca_only' in exp_pattern:
        return DATASET_ROOT / 'LOCA_only_train_LOCA_test' / 'split_metadata_lhs.json'
    else:
        return DATASET_ROOT / 'LOCA_only_train_LOCA_test' / 'split_metadata_lhs.json'


def load_test_metadata(metadata_path):
    """Load test scenario IDs from split metadata."""
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata['test_scenarios']


def load_scenario_data(scenario_id):
    """Load raw scenario CSV data."""
    csv_path = RAW_DATA_ROOT / '{}.csv'.format(scenario_id)
    return pd.read_csv(csv_path)


def find_result_folder(exp_pattern, model, pred_len, seed):
    """Find result folder for a seeded run."""
    pattern = '*{}_{}_pl{}_*_seed{}'.format(exp_pattern, model, pred_len, seed)
    matches = list(RESULTS_ROOT.glob(pattern))
    if matches:
        return matches[0]
    return None


def reconstruct_scenario_predictions(scenario_idx, pred, true, test_scenarios,
                                     seq_len, pred_len, extract_idx):
    """
    Reconstruct predictions for a single scenario using rolling-window extraction.

    Extracts the last-step prediction from each sliding window, consistent
    with the methodology in visualize_30s/300s/600s_ahead.py.

    Args:
        scenario_idx: Index in test set (0-14)
        pred: Full prediction array (n_windows, pred_len, 1)
        true: Full ground truth array (n_windows, pred_len, 1)
        test_scenarios: List of test scenario IDs
        seq_len: Input sequence length
        pred_len: Prediction length
        extract_idx: Index to extract (pred_len - 1 for last-step)

    Returns:
        predicted_sequence, actual_sequence, scenario_id
    """
    scenario_id = test_scenarios[scenario_idx]
    scenario_df = load_scenario_data(scenario_id)
    scenario_length = len(scenario_df)

    num_windows = scenario_length - seq_len - pred_len + 1

    # Calculate starting window index by summing previous scenarios' windows
    window_start_idx = 0
    for i in range(scenario_idx):
        prev_id = test_scenarios[i]
        prev_df = load_scenario_data(prev_id)
        prev_windows = len(prev_df) - seq_len - pred_len + 1
        window_start_idx += prev_windows

    scenario_pred = pred[window_start_idx:window_start_idx + num_windows]
    scenario_true = true[window_start_idx:window_start_idx + num_windows]

    predicted_sequence = np.zeros(num_windows)
    actual_sequence = np.zeros(num_windows)

    for i in range(num_windows):
        predicted_sequence[i] = scenario_pred[i, extract_idx, 0]
        actual_sequence[i] = scenario_true[i, extract_idx, 0]

    return predicted_sequence, actual_sequence, scenario_id


# ============================================================================
# PLOTTING
# ============================================================================

def plot_scenario_uncertainty(scenario_idx, all_seed_preds, all_seed_trues,
                              test_scenarios, seq_len, pred_len, extract_idx,
                              save_dir, horizon_label, exp_name='', model_name=''):
    """
    Plot a single scenario with all seed predictions overlaid.

    Args:
        scenario_idx: Test scenario index (0-14)
        all_seed_preds: List of pred arrays, one per seed
        all_seed_trues: List of true arrays, one per seed
        test_scenarios: List of test scenario IDs
        seq_len: Sequence length
        pred_len: Prediction length
        extract_idx: Index for last-step extraction
        save_dir: Output directory
        horizon_label: e.g., '300s'
        exp_name: Experiment name for title
        model_name: Model name for title and filename

    Returns:
        (save_path, mean_mae, std_mae, mean_rmse, std_rmse) or None on error
    """
    # Reconstruct predictions for each seed
    seed_sequences = []
    actual_seq = None
    scenario_id = None

    for seed_pred, seed_true in zip(all_seed_preds, all_seed_trues):
        pred_seq, act_seq, s_id = reconstruct_scenario_predictions(
            scenario_idx, seed_pred, seed_true, test_scenarios,
            seq_len, pred_len, extract_idx
        )
        seed_sequences.append(pred_seq)
        if actual_seq is None:
            actual_seq = act_seq
            scenario_id = s_id

    # Stack into (n_seeds, num_windows)
    seed_stack = np.stack(seed_sequences, axis=0)
    pred_mean = np.mean(seed_stack, axis=0)
    pred_min = np.min(seed_stack, axis=0)
    pred_max = np.max(seed_stack, axis=0)

    # Per-seed MAE and RMSE
    seed_maes = [np.mean(np.abs(s - actual_seq)) for s in seed_sequences]
    seed_rmses = [np.sqrt(np.mean((s - actual_seq) ** 2)) for s in seed_sequences]
    mean_mae = np.mean(seed_maes)
    std_mae = np.std(seed_maes)
    mean_rmse = np.mean(seed_rmses)
    std_rmse = np.std(seed_rmses)

    # Time axis
    time_steps = np.arange(
        seq_len + pred_len - 1,
        seq_len + pred_len - 1 + len(pred_mean)
    ) * 10  # seconds

    # Load full actual TPCT
    scenario_df = load_scenario_data(scenario_id)
    full_actual_tpct = scenario_df['TPCT'].values
    full_time = scenario_df['TIME'].values

    break_size_pct = scenario_id

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Layer 1: Uncertainty band (min-max)
    ax.fill_between(time_steps, pred_min, pred_max,
                    alpha=0.2, color='#D32F2F',
                    label='Min-Max Range ({} seeds)'.format(len(seed_sequences)))

    # Layer 2: Individual seed curves (thin, semi-transparent)
    for i, seq in enumerate(seed_sequences):
        label = 'Individual Seeds' if i == 0 else None
        ax.plot(time_steps, seq, color='#EF5350', linewidth=0.8, alpha=0.4,
                label=label)

    # Layer 3: Mean prediction (bold dashed)
    ax.plot(time_steps, pred_mean, 'r--', linewidth=2.0, alpha=0.9,
            label='Mean Prediction ({} seeds)'.format(len(seed_sequences)))

    # Layer 4: Actual TPCT (bold solid black)
    ax.plot(full_time, full_actual_tpct, 'k-', linewidth=2, alpha=0.7,
            label='Actual TPCT')

    # Annotations
    ax.axvspan(0, seq_len * 10, alpha=0.1, color='gray',
               label='First Input Sequence')
    ax.axvline((seq_len + pred_len - 1) * 10, color='green', linestyle=':',
               linewidth=2, label='Onset of Prediction')

    ax.set_xlabel('Time (seconds)', fontsize=13)
    ax.set_ylabel('TPCT (\u00B0C)', fontsize=13)

    ax.set_title(
        '{} {} - {}% Break'.format(
            model_name, horizon_label, break_size_pct),
        fontsize=14, fontweight='bold'
    )
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    filename = '{}_scenario_{:03d}_{}_ahead_uncertainty.png'.format(
        model_name, scenario_id, horizon_label) if model_name else \
        'scenario_{:03d}_{}_ahead_uncertainty.png'.format(scenario_id, horizon_label)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return save_path, mean_mae, std_mae, mean_rmse, std_rmse


def create_uncertainty_summary_grid(all_metrics, all_seed_preds, all_seed_trues,
                                     test_scenarios, seq_len, pred_len, extract_idx,
                                     save_dir, horizon_label, exp_name='', model_name=''):
    """
    Create 3x5 grid of all 15 scenarios with uncertainty bands.

    Shows actual (blue solid), mean prediction (red dashed), and min-max band
    (light red fill). No individual seed lines (too cluttered at small scale).
    """
    n_scenarios = len(test_scenarios)
    fig, axes = plt.subplots(3, 5, figsize=(24, 14))
    axes = axes.flatten()

    for idx in range(n_scenarios):
        ax = axes[idx]
        scenario_id = test_scenarios[idx]

        # Reconstruct per-seed predictions
        seed_sequences = []
        actual_seq = None

        for seed_pred, seed_true in zip(all_seed_preds, all_seed_trues):
            pred_seq, act_seq, _ = reconstruct_scenario_predictions(
                idx, seed_pred, seed_true, test_scenarios,
                seq_len, pred_len, extract_idx
            )
            seed_sequences.append(pred_seq)
            if actual_seq is None:
                actual_seq = act_seq

        seed_stack = np.stack(seed_sequences, axis=0)
        pred_mean = np.mean(seed_stack, axis=0)
        pred_min = np.min(seed_stack, axis=0)
        pred_max = np.max(seed_stack, axis=0)

        # Per-seed MAEs for this scenario
        seed_maes = [np.mean(np.abs(s - actual_seq)) for s in seed_sequences]
        mean_mae = np.mean(seed_maes)
        std_mae = np.std(seed_maes)

        # Time axes
        pred_time = np.arange(
            seq_len + pred_len - 1,
            seq_len + pred_len - 1 + len(pred_mean)
        ) * 10

        scenario_df = load_scenario_data(scenario_id)
        full_tpct = scenario_df['TPCT'].values
        full_time = scenario_df['TIME'].values

        # Plot
        ax.plot(full_time, full_tpct, 'k-', linewidth=1.2, alpha=0.7, label='Actual')
        ax.fill_between(pred_time, pred_min, pred_max,
                        alpha=0.2, color='#D32F2F', label='Range')
        ax.plot(pred_time, pred_mean, 'r--', linewidth=1.2, alpha=0.8, label='Mean')

        ax.set_title('{}% Break'.format(scenario_id), fontsize=10)
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.set_ylabel('TPCT (\u00B0C)', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

        if idx == 0:
            ax.legend(fontsize=8, loc='best')

    # Suptitle
    title_parts = []
    if exp_name:
        title_parts.append(exp_name)
    if model_name:
        title_parts.append(model_name)
    title_prefix = ' - '.join(title_parts)
    if title_prefix:
        title_prefix = '{}\n'.format(title_prefix)

    plt.suptitle(
        '{}All Test Scenarios - {} Predictions with Seed Uncertainty'.format(
            title_prefix, horizon_label),
        fontsize=16, fontweight='bold'
    )
    plt.tight_layout()

    filename = '{}_overview_{}_ahead_uncertainty.png'.format(
        model_name, horizon_label) if model_name else \
        'overview_{}_ahead_uncertainty.png'.format(horizon_label)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print('Saved uncertainty overview: {}'.format(save_path))
    return save_path


# ============================================================================
# MAIN LOGIC
# ============================================================================

def visualize_model_uncertainty(exp_id, exp_pattern, model, horizon_name,
                                 verbose=True):
    """
    Generate uncertainty overlay plots for one (experiment, model, horizon).

    Returns:
        dict with success/failure info
    """
    hcfg = HORIZON_CONFIG[horizon_name]
    pred_len = hcfg['pl']
    extract_idx = hcfg['extract_idx']
    horizon_label = horizon_name

    metadata_path = get_metadata_path(exp_pattern)
    if not metadata_path.exists():
        if verbose:
            print('  SKIP: Metadata not found: {}'.format(metadata_path))
        return {'status': 'skipped', 'reason': 'no metadata'}

    test_scenarios = load_test_metadata(metadata_path)

    # Load pred/true from all seed folders
    all_seed_preds = []
    all_seed_trues = []
    found_seeds = []

    for seed in SEEDS:
        result_folder = find_result_folder(exp_pattern, model, pred_len, seed)
        if result_folder is None:
            if verbose:
                print('  SKIP seed={}: result folder not found'.format(seed))
            continue

        pred_path = result_folder / 'pred.npy'
        true_path = result_folder / 'true.npy'
        if not pred_path.exists() or not true_path.exists():
            if verbose:
                print('  SKIP seed={}: pred.npy/true.npy missing'.format(seed))
            continue

        all_seed_preds.append(np.load(str(pred_path)))
        all_seed_trues.append(np.load(str(true_path)))
        found_seeds.append(seed)

    if len(found_seeds) < 2:
        if verbose:
            print('  SKIP: Need at least 2 seeds, found {}'.format(len(found_seeds)))
        return {'status': 'skipped', 'reason': 'insufficient seeds'}

    if verbose:
        print('  Found {} seeds: {}'.format(len(found_seeds), found_seeds))

    # Output directory
    output_dir = OUTPUT_ROOT / exp_id / '{}_ahead'.format(horizon_name) / 'uncertainty_plots'
    os.makedirs(str(output_dir), exist_ok=True)

    exp_name = '{}: {}'.format(exp_id.upper(), EXPERIMENTS[exp_id]['name'])

    # Plot each scenario
    all_metrics = []
    for idx in range(len(test_scenarios)):
        scenario_id = test_scenarios[idx]
        if verbose:
            print('    Scenario {} ({}/{})...'.format(
                scenario_id, idx + 1, len(test_scenarios)), end='')

        try:
            result = plot_scenario_uncertainty(
                idx, all_seed_preds, all_seed_trues,
                test_scenarios, SEQ_LEN, pred_len, extract_idx,
                str(output_dir), horizon_label,
                exp_name=exp_name, model_name=model
            )
            save_path, mean_mae, std_mae, mean_rmse, std_rmse = result
            all_metrics.append((scenario_id, mean_mae, std_mae, mean_rmse, std_rmse))
            if verbose:
                print(' MAE: {:.2f}+/-{:.2f}'.format(mean_mae, std_mae))
        except Exception as e:
            if verbose:
                print(' ERROR: {}'.format(e))
            import traceback
            traceback.print_exc()

    # Summary grid
    if all_metrics:
        if verbose:
            print('    Creating summary grid...')
        create_uncertainty_summary_grid(
            all_metrics, all_seed_preds, all_seed_trues,
            test_scenarios, SEQ_LEN, pred_len, extract_idx,
            str(output_dir), horizon_label,
            exp_name=exp_name, model_name=model
        )

    # Save metrics summary text file
    if all_metrics:
        summary_filename = '{}_{}_ahead_uncertainty_metrics.txt'.format(
            model, horizon_name)
        summary_path = output_dir / summary_filename
        with open(str(summary_path), 'w') as f:
            f.write('{} Forecast Uncertainty Metrics ({} seeds)\n'.format(
                horizon_label, len(found_seeds)))
            f.write('=' * 75 + '\n\n')
            f.write('{:<15}{:<12}{:<18}{:<18}\n'.format(
                'Scenario ID', 'Break Size',
                'MAE (mean+/-std)', 'RMSE (mean+/-std)'))
            f.write('-' * 75 + '\n')
            for s_id, m_mae, s_mae, m_rmse, s_rmse in all_metrics:
                f.write('{:<15}{}%{:<10}{:.2f}+/-{:<10.2f}{:.2f}+/-{:<10.2f}\n'.format(
                    s_id, s_id, '', m_mae, s_mae, m_rmse, s_rmse))

            avg_mean_mae = np.mean([m[1] for m in all_metrics])
            avg_std_mae = np.mean([m[2] for m in all_metrics])
            avg_mean_rmse = np.mean([m[3] for m in all_metrics])
            avg_std_rmse = np.mean([m[4] for m in all_metrics])
            f.write('\n' + '=' * 75 + '\n')
            f.write('Average MAE:  {:.2f} +/- {:.2f} K\n'.format(
                avg_mean_mae, avg_std_mae))
            f.write('Average RMSE: {:.2f} +/- {:.2f} K\n'.format(
                avg_mean_rmse, avg_std_rmse))
            f.write('\nSeeds used: {}\n'.format(found_seeds))

        if verbose:
            print('    Saved metrics: {}'.format(summary_path))

    return {
        'status': 'success',
        'n_seeds': len(found_seeds),
        'n_scenarios': len(all_metrics),
        'output_dir': str(output_dir)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Multi-seed uncertainty overlay visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_seeded_uncertainty.py --exp exp0 --model SCINet --horizon 300s
  python visualize_seeded_uncertainty.py --exp exp0 --model all --horizon all
  python visualize_seeded_uncertainty.py --exp all
        """
    )

    parser.add_argument(
        '--exp', nargs='+', required=True,
        choices=list(EXPERIMENTS.keys()) + ['all'],
        help='Experiment(s) to visualize'
    )
    parser.add_argument(
        '--model', nargs='+', default=['all'],
        help='Model(s) to visualize (default: all)'
    )
    parser.add_argument(
        '--horizon', nargs='+', default=['all'],
        choices=list(HORIZON_CONFIG.keys()) + ['all'],
        help='Horizon(s) to visualize (default: all)'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Minimal output'
    )

    args = parser.parse_args()

    # Resolve 'all' selections
    experiments = list(EXPERIMENTS.keys()) if 'all' in args.exp else args.exp
    models = MODELS if 'all' in args.model else args.model
    horizons = list(HORIZON_CONFIG.keys()) if 'all' in args.horizon else args.horizon

    verbose = not args.quiet

    print('=' * 80)
    print('MULTI-SEED UNCERTAINTY VISUALIZATION')
    print('=' * 80)
    print('Experiments: {}'.format(', '.join(experiments)))
    print('Models: {}'.format(', '.join(models)))
    print('Horizons: {}'.format(', '.join(horizons)))
    print('Seeds: {}'.format(SEEDS))
    print('=' * 80)
    print()

    summary = {}

    for exp_id in experiments:
        exp_config = EXPERIMENTS[exp_id]
        if verbose:
            print('\n{}'.format('=' * 80))
            print('EXPERIMENT: {} - {}'.format(exp_id.upper(), exp_config['name']))
            print('{}'.format('=' * 80))

        for model in models:
            for horizon in horizons:
                if verbose:
                    print('\n  {} / {} / {}:'.format(exp_id, model, horizon))

                result = visualize_model_uncertainty(
                    exp_id, exp_config['pattern'], model, horizon,
                    verbose=verbose
                )
                key = '{}/{}/{}'.format(exp_id, model, horizon)
                summary[key] = result

    # Final summary
    print('\n' + '=' * 80)
    print('SUMMARY')
    print('=' * 80)
    success_count = sum(1 for r in summary.values() if r['status'] == 'success')
    skip_count = sum(1 for r in summary.values() if r['status'] == 'skipped')
    print('  Success: {}'.format(success_count))
    print('  Skipped: {}'.format(skip_count))
    print('  Total:   {}'.format(len(summary)))
    print('=' * 80)


if __name__ == '__main__':
    main()
