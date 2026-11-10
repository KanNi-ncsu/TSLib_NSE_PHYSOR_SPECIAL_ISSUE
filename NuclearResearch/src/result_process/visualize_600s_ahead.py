"""
Per-Scenario Visualization for LOCA 600s-ahead Forecasting

Plots each test scenario separately showing 600s-ahead predictions (60-step ahead).
Uses pred[:, 59, 0] to extract consistent 600s-ahead forecast at each timestep.

Author: Nuclear Research Team
Date: 2025-10-07
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import argparse
from pathlib import Path

# Derive project root from script location
# Script is at: <PROJECT_ROOT>/NuclearResearch/src/result_process/
_PROJECT_ROOT = Path(os.path.abspath(__file__)).parents[3]
_DEFAULT_RAW_DATA = str(_PROJECT_ROOT / 'NuclearResearch' / 'Raw_data' / 'DONOTMODIFY' / 'Operation_csv_data' / 'LOCA')
_DEFAULT_METADATA = str(_PROJECT_ROOT / 'dataset' / 'LOCA_only_train_LOCA_test' / 'split_metadata_lhs.json')


def load_scenario_data(scenario_id, data_root=_DEFAULT_RAW_DATA):
    """
    Load raw scenario CSV data

    Args:
        scenario_id: Scenario number (e.g., 1, 2, 4, ...)
        data_root: Path to raw data

    Returns:
        df: DataFrame with TIME and all features including TF
    """
    csv_path = os.path.join(data_root, f'{scenario_id}.csv')
    df = pd.read_csv(csv_path)
    return df


def load_test_metadata(metadata_path=_DEFAULT_METADATA):
    """Load test scenario IDs from split metadata"""
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata['test_scenarios']


def reconstruct_scenario_predictions(scenario_idx, pred, true, seq_len=30, pred_len=30):
    """
    Reconstruct full scenario predictions from windows (600s-ahead).

    For each timestep, we use the 60-step ahead prediction (600s ahead) from the window
    that starts at (timestep - 29). This maintains consistent 300s forecast horizon.

    Args:
        scenario_idx: Index in test set (0-14)
        pred: Full prediction array (n_windows, 30, 1)
        true: Full ground truth array (n_windows, 30, 1)
        seq_len: Sequence length used in training (30)
        pred_len: Prediction length (30)

    Returns:
        predicted_sequence: 1D array of 600s-ahead predictions for this scenario
        actual_sequence: 1D array of actual values for this scenario
        scenario_length: Total length of this scenario
    """
    # Each scenario has: scenario_length - seq_len - pred_len + 1 windows

    # The windows are ordered: all windows from scenario 0, then scenario 1, etc.
    # We need to figure out which windows belong to this scenario

    # Load actual scenario to get its length
    test_scenarios = load_test_metadata()
    scenario_id = test_scenarios[scenario_idx]
    scenario_df = load_scenario_data(scenario_id)
    scenario_length = len(scenario_df)

    # Calculate number of windows for this scenario
    num_windows = scenario_length - seq_len - pred_len + 1

    # Calculate starting index for this scenario's windows
    # We need to sum up windows from all previous scenarios
    window_start_idx = 0
    for i in range(scenario_idx):
        prev_scenario_id = test_scenarios[i]
        prev_df = load_scenario_data(prev_scenario_id)
        prev_length = len(prev_df)
        prev_windows = prev_length - seq_len - pred_len + 1
        window_start_idx += prev_windows

    # Extract windows for this scenario
    scenario_pred = pred[window_start_idx:window_start_idx + num_windows]
    scenario_true = true[window_start_idx:window_start_idx + num_windows]

    # Reconstruct sequence using 60-step ahead predictions (600s ahead)
    # Position i in output = 30-step prediction from window i
    predicted_sequence = np.zeros(num_windows)
    actual_sequence = np.zeros(num_windows)

    for i in range(num_windows):
        predicted_sequence[i] = scenario_pred[i, 59, 0]  # 60-step ahead prediction (index 59)
        actual_sequence[i] = scenario_true[i, 59, 0]      # 60-step ahead actual

    return predicted_sequence, actual_sequence, scenario_length, scenario_id


def plot_scenario(scenario_idx, pred, true, save_dir, seq_len=30, pred_len=30, exp_name="", model_name=""):
    """
    Plot single scenario with 600s-ahead predicted vs actual TF

    Args:
        scenario_idx: Test scenario index (0-14)
        pred: Full prediction array
        true: Full ground truth array
        save_dir: Directory to save plot
        seq_len: Sequence length (30)
        pred_len: Prediction length (30)
        exp_name: Experiment name for plot titles
        model_name: Model name for filename and title
    """
    # Reconstruct predictions for this scenario
    predicted_seq, actual_seq, scenario_length, scenario_id = reconstruct_scenario_predictions(
        scenario_idx, pred, true, seq_len, pred_len
    )

    # Time axis (10-second intervals)
    # For 600s-ahead: predicted_seq[i] = pred[i, 59, 0] corresponds to timestep (seq_len + i + pred_len - 1)
    # This is because window i predicts timesteps [i+30, ..., i+89], and we use the last one (index 59)
    time_steps = np.arange(seq_len + pred_len - 1, seq_len + pred_len - 1 + len(predicted_seq)) * 10  # seconds

    # Load full actual TPCT from raw data for reference
    scenario_df = load_scenario_data(scenario_id)
    full_actual_tpct = scenario_df['TPCT'].values
    full_time = scenario_df['TIME'].values

    # Calculate error metrics
    mae = np.mean(np.abs(predicted_seq - actual_seq))
    rmse = np.sqrt(np.mean((predicted_seq - actual_seq)**2))

    # Determine break size from scenario ID (1-100 corresponds to 1%-100% break)
    break_size_pct = scenario_id

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot full actual TPCT (including input sequence part)
    ax.plot(full_time, full_actual_tpct, 'k-', label='Actual TPCT', linewidth=2, alpha=0.7)

    # Overlay predicted TPCT (600s-ahead predictions)
    ax.plot(time_steps, predicted_seq, 'r--', label='Predicted TPCT',
            linewidth=2, alpha=0.8)

    # Mark the first input sequence region
    ax.axvspan(0, seq_len * 10, alpha=0.1, color='gray', label='First Input Sequence')

    # Add vertical line at first prediction time
    ax.axvline((seq_len + pred_len - 1) * 10, color='green', linestyle=':', linewidth=2,
               label='Onset of Prediction')

    ax.set_xlabel('Time (seconds)', fontsize=13)
    ax.set_ylabel('TPCT (\u00B0C)', fontsize=13)

    ax.set_title(f'{model_name} 600s - {break_size_pct}% Break',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure with model name prefix
    filename = f'{model_name}_scenario_{scenario_id:03d}_600s_ahead.png' if model_name else f'scenario_{scenario_id:03d}_600s_ahead.png'
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return save_path, mae, rmse


def create_summary_grid(all_metrics, pred, true, save_dir, seq_len=30, pred_len=30, exp_name="", model_name=""):
    """
    Create grid of all scenarios (3x5 layout for 15 scenarios)

    Args:
        all_metrics: List of (scenario_id, mae, rmse) tuples
        pred: Full prediction array (n_windows, pred_len, 1)
        true: Full ground truth array (n_windows, pred_len, 1)
        save_dir: Directory to save plot
        seq_len: Sequence length used in training (30)
        pred_len: Prediction length (30)
        exp_name: Experiment name for plot titles
        model_name: Model name for filename and title
    """
    test_scenarios = load_test_metadata()

    fig, axes = plt.subplots(3, 5, figsize=(24, 14))
    axes = axes.flatten()

    for idx, (scenario_id, mae, rmse) in enumerate(all_metrics):
        ax = axes[idx]

        # Load scenario data
        scenario_df = load_scenario_data(scenario_id)
        tpct_values = scenario_df['TPCT'].values
        time_values = scenario_df['TIME'].values

        # Reconstruct predictions for this scenario
        predicted_seq, actual_seq, scenario_length, _ = reconstruct_scenario_predictions(
            idx, pred, true, seq_len, pred_len
        )

        # Time axis for predictions (600s-ahead)
        pred_time_steps = np.arange(seq_len + pred_len - 1, seq_len + pred_len - 1 + len(predicted_seq)) * 10

        # Plot actual (full scenario)
        ax.plot(time_values, tpct_values, 'k-', linewidth=1.2, alpha=0.7, label='Actual')

        # Plot predictions (600s-ahead)
        ax.plot(pred_time_steps, predicted_seq, 'r--', linewidth=1.2, alpha=0.8, label='Predicted')

        ax.set_title(f'{scenario_id}% Break',
                    fontsize=10)
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.set_ylabel('TPCT (\u00B0C)', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

        # Add legend only to first subplot
        if idx == 0:
            ax.legend(fontsize=8, loc='best')

    # Build suptitle with exp_name and model_name
    title_parts = []
    if exp_name:
        title_parts.append(exp_name)
    if model_name:
        title_parts.append(model_name)
    title_prefix = " - ".join(title_parts)
    if title_prefix:
        title_prefix = f'{title_prefix}\n'

    plt.suptitle(f'{title_prefix}All Test Scenarios - TPCT (Actual vs 600s-ahead Predictions)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    filename = f'{model_name}_overview_600s_ahead.png' if model_name else 'all_scenarios_overview.png'
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved overview: {save_path}")
    return save_path


def main():
    parser = argparse.ArgumentParser(description='600s-ahead LOCA Visualization')
    parser.add_argument('--result_folder', type=str, required=True,
                       help='Path to results folder')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: result_folder/600s_ahead_plots)')
    parser.add_argument('--seq_len', type=int, default=30,
                       help='Sequence length used in training')
    parser.add_argument('--pred_len', type=int, default=60,
                       help='Prediction length (60 for 600s-ahead)')
    parser.add_argument('--exp_name', type=str, default="",
                       help='Experiment name for plot titles')
    parser.add_argument('--model_name', type=str, default="",
                       help='Model name for plot titles and file naming')

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        output_dir = os.path.join(args.result_folder, '600s_ahead_plots')
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    print("="*80)
    print("300s-AHEAD LOCA VISUALIZATION")
    print("="*80)
    print()

    # Load results
    print(f"Loading results from: {args.result_folder}")
    pred = np.load(os.path.join(args.result_folder, 'pred.npy'))
    true = np.load(os.path.join(args.result_folder, 'true.npy'))
    print(f"  Predictions shape: {pred.shape}")
    print(f"  Ground truth shape: {true.shape}")
    print()

    # Get test scenarios
    test_scenarios = load_test_metadata()
    print(f"Test scenarios: {test_scenarios}")
    print(f"Number of scenarios: {len(test_scenarios)}")
    print()

    # Plot each scenario
    print("Generating per-scenario plots...")
    all_metrics = []

    for idx in range(len(test_scenarios)):
        scenario_id = test_scenarios[idx]
        print(f"  Processing scenario {scenario_id} ({idx+1}/{len(test_scenarios)})...")

        try:
            save_path, mae, rmse = plot_scenario(
                idx, pred, true, output_dir,
                seq_len=args.seq_len, pred_len=args.pred_len,
                exp_name=args.exp_name, model_name=args.model_name
            )
            all_metrics.append((scenario_id, mae, rmse))
            print(f"    Saved: {os.path.basename(save_path)} (MAE: {mae:.2f} K)")
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Create summary grid
    print("\nCreating summary overview...")
    create_summary_grid(all_metrics, pred, true, output_dir, args.seq_len, args.pred_len,
                       args.exp_name, args.model_name)

    # Save metrics summary
    print("\nSaving metrics summary...")
    metrics_filename = f'{args.model_name}_600s_ahead_metrics_summary.txt' if args.model_name else '600s_ahead_metrics_summary.txt'
    summary_path = os.path.join(output_dir, metrics_filename)
    with open(summary_path, 'w') as f:
        f.write("600s-ahead Forecast Performance Metrics (60-step ahead)\n")
        f.write("="*60 + "\n\n")
        f.write(f"{'Scenario ID':<15}{'Break Size':<15}{'MAE (K)':<15}{'RMSE (K)':<15}\n")
        f.write("-"*60 + "\n")
        for scenario_id, mae, rmse in all_metrics:
            f.write(f"{scenario_id:<15}{scenario_id}%{'':<13}{mae:<15.2f}{rmse:<15.2f}\n")

        f.write("\n" + "="*60 + "\n")
        f.write(f"Average MAE:  {np.mean([m[1] for m in all_metrics]):.2f} K\n")
        f.write(f"Average RMSE: {np.mean([m[2] for m in all_metrics]):.2f} K\n")
        f.write("\nNote: All predictions are consistent 600s-ahead (60-step) forecasts.\n")

    print(f"  Saved: {summary_path}")

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print(f"All plots saved to: {output_dir}")
    print(f"  - {len(test_scenarios)} individual scenario plots")
    print(f"  - 1 overview grid")
    print(f"  - 1 metrics summary")
    print("="*80)


if __name__ == '__main__':
    main()
