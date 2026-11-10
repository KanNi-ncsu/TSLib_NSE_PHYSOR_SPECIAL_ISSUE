"""
Inference Benchmark Script for Nuclear LOCA Forecasting Models

Measures pure forward pass time only, excluding data loading, checkpoint loading,
and post-processing. Designed for nuclear safety analysis where demonstrating
fast inference time is critical.

This script is standalone -- it does NOT modify any existing TSLib files.
It imports from TSLib to build models and load data, reads checkpoints,
and writes timing results to a CSV file.

Supports two modes:
  --benchmark_mode dl         : Deep learning models with saved checkpoints
  --benchmark_mode foundation : Foundation models (Chronos, Chronos2, TimesFM)

Usage:
  python NuclearResearch/src/run_inference_benchmark.py \
      --benchmark_mode dl \
      --model Transformer \
      --model_id LOCA_TPCT_exp0_loca_only_Transformer_pl3 \
      --pred_len 3 --label_len 2 \
      --d_model 128 --d_ff 512 --e_layers 2 --n_heads 4 --batch_size 32 \
      --n_warmup 3 --n_runs 10 \
      --timing_csv NuclearResearch/results_summary/timing/inference_timing_exp0_summary.csv

Author: Nuclear Research Team
Date: 2026-03-25
"""

import os
import sys
import argparse
import time
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import torch
from data_provider.data_factory import data_provider


def create_benchmark_args():
    """
    Create argument parser for inference benchmarking.

    Argument defaults are copied directly from run_loca_forecast.py to ensure
    identical setting strings and correct checkpoint path resolution.
    Additional benchmark-specific args are added at the end.
    """
    parser = argparse.ArgumentParser(description='Nuclear LOCA Inference Benchmark')

    # =========================================================================
    # Arguments copied from run_loca_forecast.py (identical defaults)
    # =========================================================================

    # Basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                       help='Task name: long_term_forecast or zero_shot_forecast')
    parser.add_argument('--is_training', type=int, default=0,
                       help='Status: always 0 for benchmark (inference only)')
    parser.add_argument('--model_id', type=str, default='LOCA_300s_ahead',
                       help='Model ID for checkpoint path')
    parser.add_argument('--model', type=str, default='Transformer',
                       help='Model name')

    # Data config
    parser.add_argument('--data', type=str, default='nuclear_accident',
                       help='Dataset name: nuclear_accident')
    parser.add_argument('--root_path', type=str,
                       default=str(project_root / 'dataset'),
                       help='Root path to dataset')
    parser.add_argument('--data_path', type=str, default='LOCA_only_train_LOCA_test',
                       help='Dataset folder under --root_path')
    parser.add_argument('--accident_type', type=str, default='LOCA',
                       help='Accident type: LOCA, SGTR, etc.')
    parser.add_argument('--features', type=str, default='MS',
                       help='Features: M (MIMO), MS (MISO), S (SISO)')
    parser.add_argument('--target', type=str, default='TPCT',
                       help='Target feature name')
    parser.add_argument('--freq', type=str, default='s',
                       help='Frequency: s for seconds')
    parser.add_argument('--checkpoints', type=str,
                       default='./checkpoints',
                       help='Checkpoint directory')

    # Forecasting task parameters
    parser.add_argument('--seq_len', type=int, default=30,
                       help='Input sequence length')
    parser.add_argument('--label_len', type=int, default=15,
                       help='Decoder start token length')
    parser.add_argument('--pred_len', type=int, default=30,
                       help='Prediction length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly',
                       help='Seasonal patterns (required by TSLib)')
    parser.add_argument('--inverse', type=int, default=1,
                       help='Inverse output')

    # Model architecture
    parser.add_argument('--enc_in', type=int, default=74,
                       help='Encoder input size')
    parser.add_argument('--dec_in', type=int, default=74,
                       help='Decoder input size')
    parser.add_argument('--c_out', type=int, default=1,
                       help='Output size')
    parser.add_argument('--d_model', type=int, default=128,
                       help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=4,
                       help='Number of attention heads')
    parser.add_argument('--e_layers', type=int, default=2,
                       help='Number of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1,
                       help='Number of decoder layers')
    parser.add_argument('--d_ff', type=int, default=512,
                       help='Feed-forward dimension')
    parser.add_argument('--moving_avg', type=int, default=25,
                       help='Moving average window size')
    parser.add_argument('--factor', type=int, default=1,
                       help='Attention factor')
    parser.add_argument('--distil', action='store_false', default=True,
                       help='Use distilling in encoder')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--embed', type=str, default='fixed',
                       help='Time features encoding: timeF, fixed, learned')
    parser.add_argument('--activation', type=str, default='gelu',
                       help='Activation function')
    parser.add_argument('--output_attention', action='store_true',
                       default=False,
                       help='Output attention in encoder')
    parser.add_argument('--expand', type=int, default=2,
                       help='Expansion factor')
    parser.add_argument('--d_conv', type=int, default=4,
                       help='Convolution kernel size')

    # Optimization (needed for arg compatibility)
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Data loader workers')
    parser.add_argument('--train_epochs', type=int, default=20,
                       help='Training epochs (not used)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience (not used)')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                       help='Learning rate (not used)')
    parser.add_argument('--loss', type=str, default='MSE',
                       help='Loss function (not used)')
    parser.add_argument('--lradj', type=str, default='type1',
                       help='Learning rate adjustment (not used)')
    parser.add_argument('--use_amp', action='store_true', default=False,
                       help='Use automatic mixed precision')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True,
                       help='Use GPU')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--gpu_type', type=str, default='cuda',
                       help='GPU type: cuda or mps')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False,
                       help='Use multiple GPUs')
    parser.add_argument('--devices', type=str, default='0,1,2,3',
                       help='Device IDs for multi-GPU')

    # Additional TSLib required arguments
    parser.add_argument('--des', type=str, default='LOCA_forecast',
                       help='Experiment description')
    parser.add_argument('--itr', type=int, default=1,
                       help='Experiment iterations')
    parser.add_argument('--use_dtw', action='store_true', default=False,
                       help='Use DTW metric (not used)')

    # Model-specific parameters
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                       help='Hidden dims for Nonstationary_Transformer projector')
    parser.add_argument('--p_hidden_layers', type=int, default=2,
                       help='Hidden layers for Nonstationary_Transformer projector')
    parser.add_argument('--top_k', type=int, default=5,
                       help='For TimesNet: top frequencies in FFT')
    parser.add_argument('--num_kernels', type=int, default=6,
                       help='For TimesNet: kernels in Inception block')
    parser.add_argument('--down_sampling_layers', type=int, default=3,
                       help='For TimeMixer: downsampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=2,
                       help='For TimeMixer: downsampling window size')
    parser.add_argument('--down_sampling_method', type=str, default='avg',
                       choices=['max', 'avg', 'conv'],
                       help='For TimeMixer: downsampling method')
    parser.add_argument('--channel_independence', type=int, default=0,
                       help='For TimeMixer: channel independence')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                       choices=['moving_avg', 'dft_decomp'],
                       help='For TimeMixer: decomposition method')
    parser.add_argument('--use_norm', type=int, default=1,
                       help='For TimeMixer: use normalization')
    parser.add_argument('--seg_len', type=int, default=30,
                       help='For SegRNN: segment length')
    parser.add_argument('--patch_len', type=int, default=16,
                       help='For PatchTST/MultiPatchFormer: patch length')

    # =========================================================================
    # Benchmark-specific arguments
    # =========================================================================
    parser.add_argument('--benchmark_mode', type=str, default='dl',
                       choices=['dl', 'foundation', 'arima'],
                       help='dl: DL models with checkpoints, foundation: zero-shot models, arima: ARIMA statistical baseline')
    parser.add_argument('--finetuned_model_path', type=str, default=None,
                       help='Path to fine-tuned model checkpoint (for Chronos_FT/Chronos2_FT)')
    parser.add_argument('--n_warmup', type=int, default=3,
                       help='Number of warm-up forward passes (discarded)')
    parser.add_argument('--n_runs', type=int, default=10,
                       help='Number of timed forward passes')
    parser.add_argument('--timing_csv', type=str,
                       default='NuclearResearch/results_summary/timing/inference_timing_exp4_summary.csv',
                       help='Path to output timing CSV')

    args = parser.parse_args()
    return args


def build_setting_string(args):
    """
    Construct the setting string exactly as in run_loca_forecast.py and
    run_loca_zeroshot.py. This determines the checkpoint path.
    """
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_0'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.expand,
        args.d_conv,
        args.factor,
        args.embed,
        args.distil,
        args.des
    )
    return setting


def count_parameters(model):
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def time_forward_pass(model, batch_x, batch_x_mark, dec_inp, batch_y_mark,
                      use_amp, device, n_warmup, n_runs):
    """
    Time the pure forward pass of a model.

    Performs warm-up runs first to eliminate CUDA cold-start overhead
    (kernel compilation, memory allocation, GPU frequency ramp-up),
    then measures n_runs forward passes with proper GPU synchronization.

    Returns:
        list of float: Per-run forward pass times in milliseconds
    """
    use_cuda = device.type == 'cuda'

    model.eval()
    with torch.no_grad():
        # Warm-up: discard these timings
        for _ in range(n_warmup):
            if use_amp:
                with torch.cuda.amp.autocast():
                    _ = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                _ = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            if use_cuda:
                torch.cuda.synchronize()

        # Measured runs
        times_ms = []
        for _ in range(n_runs):
            if use_cuda:
                torch.cuda.synchronize()

            t0 = time.perf_counter()

            if use_amp:
                with torch.cuda.amp.autocast():
                    _ = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                _ = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            if use_cuda:
                torch.cuda.synchronize()

            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    return times_ms


def run_arima_benchmark(args):
    """
    Benchmark ARIMA statistical baseline (CPU-only, no PyTorch).

    Times fit_and_forecast() on a single sliding window of TPCT data,
    matching the warm-up + N timed runs methodology used for DL models.
    """
    # Import ARIMA modules
    baselines_dir = os.path.join(project_root, 'NuclearResearch', 'src', 'baselines')
    sys.path.insert(0, baselines_dir)
    from arima_standalone import ARIMAForecaster
    from arima_data_loader import NuclearAccidentDataLoader

    print(f"Loading test data for ARIMA benchmark...")
    loader = NuclearAccidentDataLoader(
        data_path=args.data_path,
        target='TPCT',
        scale=True
    )
    loader.fit_scaler()
    test_data = loader.load_test_data(seq_len=args.seq_len, pred_len=args.pred_len)

    # Extract one sliding window from the first test scenario
    scenario = test_data[0]
    target_series = scenario['data'][:, scenario['target_idx']]
    history = target_series[:args.seq_len]

    forecaster = ARIMAForecaster(order=(1, 1, 1))
    print(f"ARIMA benchmark: seq_len={args.seq_len}, pred_len={args.pred_len}")
    print(f"  History shape: {history.shape}")

    # Warm-up
    print(f"  Warm-up: {args.n_warmup} runs...")
    for _ in range(args.n_warmup):
        forecaster.fit_and_forecast(history, steps=args.pred_len)

    # Timed runs
    print(f"  Timing: {args.n_runs} runs...")
    times_ms = []
    for _ in range(args.n_runs):
        t0 = time.perf_counter()
        forecaster.fit_and_forecast(history, steps=args.pred_len)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    batch_mean = np.mean(times_ms)
    batch_std = np.std(times_ms)
    batch_min = np.min(times_ms)
    batch_max = np.max(times_ms)

    # ARIMA processes one sample at a time, so single = batch
    single_mean, single_std, single_min, single_max = batch_mean, batch_std, batch_min, batch_max

    print(f"  Forward pass: {batch_mean:.3f} +/- {batch_std:.3f} ms "
          f"(min={batch_min:.3f}, max={batch_max:.3f})")

    # Real-time factor
    sampling_interval_ms = 10 * 1000  # 10 seconds in ms
    realtime_factor = single_mean / sampling_interval_ms
    speedup = sampling_interval_ms / single_mean if single_mean > 0 else float('inf')
    print(f"  Real-time factor: {realtime_factor:.6f} ({speedup:.0f}x faster than real-time)")

    # Write CSV row
    csv_path = args.timing_csv
    write_header = not os.path.exists(csv_path)
    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    horizon_map = {3: '30s', 30: '300s', 60: '600s'}
    horizon = horizon_map.get(args.pred_len, f'{args.pred_len * 10}s')

    with open(csv_path, 'a') as f:
        if write_header:
            f.write("model,horizon,pred_len,batch_size,"
                    "batch_fwd_mean_ms,batch_fwd_std_ms,batch_fwd_min_ms,batch_fwd_max_ms,"
                    "single_fwd_mean_ms,single_fwd_std_ms,single_fwd_min_ms,single_fwd_max_ms,"
                    "realtime_factor,speedup_vs_realtime,"
                    "n_params,n_warmup,n_runs,device,gpu_name,status\n")

        f.write(f"ARIMA,{horizon},{args.pred_len},1,"
                f"{batch_mean:.4f},{batch_std:.4f},{batch_min:.4f},{batch_max:.4f},"
                f"{single_mean:.4f},{single_std:.4f},{single_min:.4f},{single_max:.4f},"
                f"{realtime_factor:.8f},{speedup:.1f},"
                f"3,{args.n_warmup},{args.n_runs},cpu,CPU,OK\n")

    print(f"\nResults appended to: {csv_path}")

    return {
        'model': 'ARIMA',
        'batch_fwd_mean_ms': batch_mean,
        'single_fwd_mean_ms': single_mean,
        'realtime_factor': realtime_factor,
        'n_params': 3,
    }


def run_benchmark(args):
    """
    Main benchmark logic:
    1. Build model and load checkpoint
    2. Get one test batch and prepare inputs
    3. Time forward pass (batch and single-sample)
    4. Write results to CSV
    """
    # ARIMA has its own benchmark path (CPU-only, no PyTorch)
    if args.benchmark_mode == 'arima':
        return run_arima_benchmark(args)

    setting = build_setting_string(args)

    # -- Pre-flight: verify checkpoint exists (DL mode only) --
    if args.benchmark_mode == 'dl':
        ckpt_path = os.path.join(args.checkpoints, setting, 'checkpoint.pth')
        if not os.path.exists(ckpt_path):
            print(f"ERROR: Checkpoint not found: {ckpt_path}")
            print(f"  Setting string: {setting}")
            print(f"  This likely means CLI arguments do not match training args.")
            print(f"  Verify: --model_id, --d_model, --d_ff, --e_layers, --d_layers, "
                  f"--n_heads, --label_len, --pred_len match the training script.")
            return None

    # -- Build model --
    print(f"Building model: {args.model}")
    if args.benchmark_mode == 'dl':
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
        exp = Exp_Long_Term_Forecast(args)
    else:
        from exp.exp_zero_shot_forecasting import Exp_Zero_Shot_Forecast
        exp = Exp_Zero_Shot_Forecast(args)

    model = exp.model
    device = exp.device

    # -- Load checkpoint (DL mode only) --
    if args.benchmark_mode == 'dl':
        print(f"Loading checkpoint: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")

    # -- Load one test batch --
    print("Loading test data...")
    test_data, test_loader = data_provider(args, 'test')

    # Get first batch
    batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(test_loader))
    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float().to(device)
    batch_x_mark = batch_x_mark.float().to(device)
    batch_y_mark = batch_y_mark.float().to(device)

    # Prepare decoder input (same as exp_long_term_forecasting.py lines 189-191)
    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
    dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

    actual_batch_size = batch_x.shape[0]
    print(f"Batch shape: {list(batch_x.shape)} (batch_size={actual_batch_size})")

    # -- Benchmark: full batch --
    print(f"\nBenchmarking full batch (n_warmup={args.n_warmup}, n_runs={args.n_runs})...")
    batch_times = time_forward_pass(
        model, batch_x, batch_x_mark, dec_inp, batch_y_mark,
        args.use_amp, device, args.n_warmup, args.n_runs
    )

    batch_mean = np.mean(batch_times)
    batch_std = np.std(batch_times)
    batch_min = np.min(batch_times)
    batch_max = np.max(batch_times)

    print(f"  Batch forward pass: {batch_mean:.3f} +/- {batch_std:.3f} ms "
          f"(min={batch_min:.3f}, max={batch_max:.3f})")

    # -- Benchmark: single sample (batch_size=1) --
    print(f"Benchmarking single sample...")
    single_x = batch_x[:1]
    single_x_mark = batch_x_mark[:1]
    single_y = batch_y[:1]
    single_y_mark = batch_y_mark[:1]
    single_dec = torch.zeros_like(single_y[:, -args.pred_len:, :]).float()
    single_dec = torch.cat([single_y[:, :args.label_len, :], single_dec], dim=1).float().to(device)

    single_times = time_forward_pass(
        model, single_x, single_x_mark, single_dec, single_y_mark,
        args.use_amp, device, args.n_warmup, args.n_runs
    )

    single_mean = np.mean(single_times)
    single_std = np.std(single_times)
    single_min = np.min(single_times)
    single_max = np.max(single_times)

    print(f"  Single sample forward pass: {single_mean:.3f} +/- {single_std:.3f} ms "
          f"(min={single_min:.3f}, max={single_max:.3f})")

    # -- Nuclear safety metric --
    sampling_interval_ms = 10 * 1000  # 10 seconds in ms
    realtime_factor = single_mean / sampling_interval_ms
    speedup = sampling_interval_ms / single_mean if single_mean > 0 else float('inf')
    print(f"\n  Real-time factor: {realtime_factor:.6f} "
          f"({speedup:.0f}x faster than real-time)")

    # -- Device info --
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(device)
    else:
        gpu_name = 'CPU'

    # -- Write CSV row --
    csv_path = args.timing_csv
    write_header = not os.path.exists(csv_path)

    # Ensure output directory exists
    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    with open(csv_path, 'a') as f:
        if write_header:
            f.write("model,horizon,pred_len,batch_size,"
                    "batch_fwd_mean_ms,batch_fwd_std_ms,batch_fwd_min_ms,batch_fwd_max_ms,"
                    "single_fwd_mean_ms,single_fwd_std_ms,single_fwd_min_ms,single_fwd_max_ms,"
                    "realtime_factor,speedup_vs_realtime,"
                    "n_params,n_warmup,n_runs,device,gpu_name,status\n")

        # Determine horizon name from pred_len
        horizon_map = {3: '30s', 30: '300s', 60: '600s', 8: '80s', 32: '320s', 64: '640s'}
        horizon = horizon_map.get(args.pred_len, f'{args.pred_len * 10}s')

        f.write(f"{args.model},{horizon},{args.pred_len},{actual_batch_size},"
                f"{batch_mean:.4f},{batch_std:.4f},{batch_min:.4f},{batch_max:.4f},"
                f"{single_mean:.4f},{single_std:.4f},{single_min:.4f},{single_max:.4f},"
                f"{realtime_factor:.8f},{speedup:.1f},"
                f"{n_params},{args.n_warmup},{args.n_runs},{device},{gpu_name},OK\n")

    print(f"\nResults appended to: {csv_path}")

    return {
        'model': args.model,
        'batch_fwd_mean_ms': batch_mean,
        'single_fwd_mean_ms': single_mean,
        'realtime_factor': realtime_factor,
        'n_params': n_params,
    }


def main():
    print("=" * 80)
    print("Nuclear LOCA Inference Benchmark")
    print("=" * 80)

    args = create_benchmark_args()

    print(f"\nConfiguration:")
    print(f"  Mode: {args.benchmark_mode}")
    print(f"  Model: {args.model}")
    print(f"  Model ID: {args.model_id}")
    print(f"  pred_len: {args.pred_len} ({args.pred_len * 10}s)")
    print(f"  Warm-up runs: {args.n_warmup}")
    print(f"  Measurement runs: {args.n_runs}")
    print(f"  Output CSV: {args.timing_csv}")

    result = run_benchmark(args)

    if result is None:
        print("\nBenchmark FAILED -- see errors above.")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print(f"  {result['model']}: "
          f"batch={result['batch_fwd_mean_ms']:.3f}ms, "
          f"single={result['single_fwd_mean_ms']:.3f}ms, "
          f"realtime_factor={result['realtime_factor']:.6f}")
    print("=" * 80)


if __name__ == '__main__':
    main()
