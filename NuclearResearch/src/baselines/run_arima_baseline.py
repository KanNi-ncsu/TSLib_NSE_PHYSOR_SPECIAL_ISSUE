"""
Standalone ARIMA Baseline for Nuclear LOCA Forecasting

Runs ARIMA(1,1,1) baseline separately from TSLib to avoid gradient corruption.
Matches TSLib preprocessing and outputs results in TSLib-compatible format.

Usage:
    python run_arima_baseline.py --exp exp0 --seq_len 30 --pred_len 3
    python run_arima_baseline.py --exp exp1 --seq_len 30 --pred_len 30
    python run_arima_baseline.py --exp exp2 --seq_len 30 --pred_len 60
    python run_arima_baseline.py --exp exp3 --seq_len 30 --pred_len 30
    python run_arima_baseline.py --exp exp4 --seq_len 30 --pred_len 30

Author: Nuclear Research Team
Date: 2025
"""

import argparse
import os
import sys
import time
import numpy as np

# Add baselines directory to path
sys.path.append(os.path.dirname(__file__))

from arima_standalone import ARIMAForecaster
from arima_data_loader import NuclearAccidentDataLoader


def calculate_metrics(predictions, ground_truth):
    """
    Calculate forecasting metrics

    Parameters:
    -----------
    predictions : np.ndarray, shape [n_samples, pred_len]
        Predicted values

    ground_truth : np.ndarray, shape [n_samples, pred_len]
        True values

    Returns:
    --------
    metrics : dict
        Dictionary containing MAE, MSE, RMSE, MAPE
    """
    mae = np.mean(np.abs(predictions - ground_truth))
    mse = np.mean((predictions - ground_truth) ** 2)
    rmse = np.sqrt(mse)

    # MAPE (handle division by zero)
    epsilon = 1e-8
    mape = np.mean(np.abs((ground_truth - predictions) / (ground_truth + epsilon))) * 100

    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape
    }


def run_arima_forecast(exp='exp1', seq_len=30, pred_len=30, arima_order=(1, 1, 1)):
    """
    Run ARIMA baseline forecast for nuclear LOCA data

    Parameters:
    -----------
    exp : str
        Experiment name ('exp1' or 'exp2')

    seq_len : int
        Sequence length (history window)

    pred_len : int
        Prediction length (forecast horizon)

    arima_order : tuple (p, d, q)
        ARIMA order (default: (1,1,1) - standard baseline)

    Returns:
    --------
    results : dict
        Dictionary containing predictions, ground truth, and metrics
    """
    print("="*80)
    print(f"ARIMA Baseline - Nuclear LOCA Forecasting ({exp.upper()})")
    print("="*80)
    print()

    # Experiment configurations
    # 'pattern' matches what visualize_experiment.py uses in find_result_folder()
    exp_configs = {
        'exp0': {
            'data_path': 'LOCA_only_train_LOCA_test',
            'pattern': 'exp0_loca_only',
            'description': 'LOCA-only (75 scenarios) train, LOCA (15 scenarios) test'
        },
        'exp1': {
            'data_path': 'LOCA_normal_train_LOCA_test',
            'pattern': 'exp1_loca_normal',
            'description': 'LOCA + Normal (95 scenarios) train, LOCA (15 scenarios) test'
        },
        'exp2': {
            'data_path': 'normal_LOCAC_train_LOCA_test',
            'pattern': 'exp2_locac_normal',
            'description': 'LOCAC + Normal (95 scenarios) train, LOCA (15 scenarios) test'
        },
        'exp3': {
            'data_path': 'LOCA_FLB_normal_train_LOCA_test',
            'pattern': 'exp3_flb_normal',
            'description': 'FLB + Normal (95 scenarios) train, LOCA (15 scenarios) test'
        },
        'exp4': {
            'data_path': 'normal_LOCAC_RW_LOF_SGATR_train_LOCA_test',
            'pattern': 'exp4_multi_accident',
            'description': 'Multi-accident (395 scenarios) train, LOCA (15 scenarios) test'
        }
    }

    if exp not in exp_configs:
        raise ValueError(f"Invalid experiment: {exp}. Choose 'exp0', 'exp1', 'exp2', 'exp3', or 'exp4'")

    config = exp_configs[exp]

    print(f"Configuration:")
    print(f"  Experiment: {exp}")
    print(f"  Dataset: {config['description']}")
    print(f"  Data path: {config['data_path']}")
    print(f"  Sequence length: {seq_len} timesteps ({seq_len * 10} seconds)")
    print(f"  Prediction length: {pred_len} timesteps ({pred_len * 10} seconds)")
    print(f"  ARIMA order: {arima_order} (Box-Jenkins standard baseline)")
    print(f"  Target: TPCT (Peak Clad Temperature)")
    print()

    # Load data
    print("Step 1: Loading data")
    print("-" * 80)

    loader = NuclearAccidentDataLoader(
        data_path=config['data_path'],
        target='TPCT',
        scale=True
    )

    # Fit scaler on training data
    loader.fit_scaler()

    # Load test data
    test_data = loader.load_test_data(seq_len=seq_len, pred_len=pred_len)

    print(f"Loaded {len(test_data)} test scenarios")
    print()

    # Initialize ARIMA forecaster
    print("Step 2: Initializing ARIMA forecaster")
    print("-" * 80)

    forecaster = ARIMAForecaster(order=arima_order)
    print(f"ARIMA forecaster: {forecaster}")
    print()

    # Run forecasts for each scenario
    print("Step 3: Running forecasts")
    print("-" * 80)

    all_predictions_scaled = []
    all_ground_truth_scaled = []
    all_predictions_original = []
    all_ground_truth_original = []

    start_time = time.time()

    for idx, scenario_data in enumerate(test_data):
        scenario_id = scenario_data['scenario_id']
        scaled_series = scenario_data['data']
        target_idx = scenario_data['target_idx']

        # Extract target column (TPCT - last column)
        target_series = scaled_series[:, target_idx]  # [timesteps]

        # Number of windows we can extract
        n_timesteps = len(target_series)
        n_windows = n_timesteps - seq_len - pred_len + 1

        if n_windows <= 0:
            print(f"  Scenario {scenario_id}: Too short, skipping")
            continue

        # Collect predictions for this scenario
        # Use MULTIPLE windows to match TSLib's sliding window approach
        scenario_predictions = []
        scenario_ground_truth = []

        # Extract windows (like TSLib does)
        for window_start in range(n_windows):
            # Historical window
            history = target_series[window_start:window_start + seq_len]

            # Ground truth future
            future_start = window_start + seq_len
            future_end = future_start + pred_len
            future_truth = target_series[future_start:future_end]

            # ARIMA forecast
            forecast = forecaster.fit_and_forecast(history, steps=pred_len)

            scenario_predictions.append(forecast)
            scenario_ground_truth.append(future_truth)

        # Convert to arrays
        scenario_predictions = np.array(scenario_predictions)  # [n_windows, pred_len]
        scenario_ground_truth = np.array(scenario_ground_truth)  # [n_windows, pred_len]

        # Store scaled predictions
        all_predictions_scaled.append(scenario_predictions)
        all_ground_truth_scaled.append(scenario_ground_truth)

        # Inverse transform to original scale
        # Need to expand to full feature dimension for inverse transform
        n_features = scaled_series.shape[1]

        # Create full arrays with target column
        # IMPORTANT: Need all features for inverse transform, but only target has values
        pred_full = np.zeros((scenario_predictions.shape[0], pred_len, n_features))
        truth_full = np.zeros((scenario_ground_truth.shape[0], pred_len, n_features))

        pred_full[:, :, target_idx] = scenario_predictions
        truth_full[:, :, target_idx] = scenario_ground_truth

        # Reshape for inverse transform: [n_windows * pred_len, n_features]
        pred_reshaped = pred_full.reshape(-1, n_features)
        truth_reshaped = truth_full.reshape(-1, n_features)

        # Clip extreme values before inverse transform (handle ARIMA numerical instability)
        # ARIMA sometimes produces extreme outliers due to convergence issues
        pred_reshaped = np.clip(pred_reshaped, -10, 10)  # In scaled space, typically [-3, 3]

        # Inverse transform
        pred_original = loader.inverse_transform(pred_reshaped)
        truth_original = loader.inverse_transform(truth_reshaped)

        # Extract target column only and reshape back to [n_windows, pred_len]
        pred_original = pred_original[:, target_idx].reshape(scenario_predictions.shape)
        truth_original = truth_original[:, target_idx].reshape(scenario_ground_truth.shape)

        all_predictions_original.append(pred_original)
        all_ground_truth_original.append(truth_original)

        print(f"  Scenario {scenario_id}: {n_windows} windows forecasted")

    elapsed_time = time.time() - start_time
    print(f"\nForecasting completed in {elapsed_time:.2f} seconds")
    print()

    # Combine all predictions
    all_predictions_scaled = np.vstack(all_predictions_scaled)  # [total_windows, pred_len]
    all_ground_truth_scaled = np.vstack(all_ground_truth_scaled)
    all_predictions_original = np.vstack(all_predictions_original)
    all_ground_truth_original = np.vstack(all_ground_truth_original)

    # Reshape to TSLib format: [n_windows, pred_len, 1] (add feature dimension)
    all_predictions_scaled = all_predictions_scaled[:, :, np.newaxis]
    all_ground_truth_scaled = all_ground_truth_scaled[:, :, np.newaxis]
    all_predictions_original = all_predictions_original[:, :, np.newaxis]
    all_ground_truth_original = all_ground_truth_original[:, :, np.newaxis]

    print(f"Total windows: {all_predictions_scaled.shape[0]}")
    print(f"Prediction shape: {all_predictions_scaled.shape}")
    print()

    # Calculate metrics on ORIGINAL scale (like TSLib)
    print("Step 4: Evaluating performance")
    print("-" * 80)

    metrics = calculate_metrics(all_predictions_original, all_ground_truth_original)

    print("Performance Metrics (Original Scale - TPCT in Kelvin):")
    print(f"  MAE:  {metrics['MAE']:.4f} K")
    print(f"  MSE:  {metrics['MSE']:.2f}")
    print(f"  RMSE: {metrics['RMSE']:.4f} K")
    print(f"  MAPE: {metrics['MAPE']:.4f} %")
    print()

    # Save results
    print("Step 5: Saving results")
    print("-" * 80)

    # Output directory naming matches viz pipeline pattern: *{exp_pattern}_ARIMA_pl{pred_len}_*
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    results_dir = os.path.join(project_root, 'results')
    output_dir = os.path.join(results_dir, f'arima_LOCA_TPCT_{config["pattern"]}_ARIMA_pl{pred_len}_standalone')
    os.makedirs(output_dir, exist_ok=True)

    # Save predictions and ground truth (TSLib format)
    np.save(os.path.join(output_dir, 'pred.npy'), all_predictions_original)
    np.save(os.path.join(output_dir, 'true.npy'), all_ground_truth_original)

    print(f"Saved predictions: {output_dir}/pred.npy")
    print(f"Saved ground truth: {output_dir}/true.npy")

    # Save metrics
    metrics_file = os.path.join(output_dir, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("ARIMA Baseline - Nuclear LOCA Forecasting\n")
        f.write(f"Experiment: {exp}\n")
        f.write(f"ARIMA Order: {arima_order}\n")
        f.write(f"\n")
        f.write("Performance Metrics:\n")
        f.write(f"  MAE:  {metrics['MAE']:.4f} K\n")
        f.write(f"  MSE:  {metrics['MSE']:.2f}\n")
        f.write(f"  RMSE: {metrics['RMSE']:.4f} K\n")
        f.write(f"  MAPE: {metrics['MAPE']:.4f} %\n")
        f.write(f"\n")
        f.write(f"Total windows: {all_predictions_scaled.shape[0]}\n")
        f.write(f"Prediction length: {pred_len}\n")
        f.write(f"Forecasting time: {elapsed_time:.2f} seconds\n")

    print(f"Saved metrics: {metrics_file}")
    print()

    print("="*80)
    print(f"ARIMA Baseline Completed - MAE: {metrics['MAE']:.4f} K")
    print("="*80)

    return {
        'predictions': all_predictions_original,
        'ground_truth': all_ground_truth_original,
        'metrics': metrics,
        'output_dir': output_dir
    }


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='ARIMA Baseline for Nuclear LOCA Forecasting')

    parser.add_argument('--exp', type=str, default='exp1',
                        choices=['exp0', 'exp1', 'exp2', 'exp3', 'exp4'],
                        help='Experiment name (exp0, exp1, exp2, exp3, or exp4)')

    parser.add_argument('--seq_len', type=int, default=30,
                        help='Sequence length (default: 30)')

    parser.add_argument('--pred_len', type=int, default=30,
                        help='Prediction length (default: 30)')

    parser.add_argument('--arima_order', type=str, default='1,1,1',
                        help='ARIMA order as p,d,q (default: 1,1,1)')

    args = parser.parse_args()

    # Parse ARIMA order
    arima_order = tuple(map(int, args.arima_order.split(',')))

    # Run forecast
    results = run_arima_forecast(
        exp=args.exp,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        arima_order=arima_order
    )

    print(f"\nResults saved to: {results['output_dir']}")


if __name__ == "__main__":
    main()
