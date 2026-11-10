"""
LOCA Forecasting Training Script

Trains transformer models on nuclear LOCA accident scenarios.
Uses TSLib infrastructure with Dataset_Nuclear_Accident.

Author: Nuclear Research Team
Date: 2025-10-06
"""

import os
import sys
import argparse
import random
from pathlib import Path

import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


def create_loca_args():
    """
    Create argument parser and set defaults for LOCA forecasting

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Nuclear LOCA Forecasting')

    # Basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                       help='Task name: long_term_forecast')
    parser.add_argument('--is_training', type=int, default=1,
                       help='Status: 1 for train, 0 for test')
    parser.add_argument('--model_id', type=str, default='LOCA_300s_ahead',
                       help='Model ID for saving (300s_ahead for 300s rolling forecast)')
    parser.add_argument('--model', type=str, default='Transformer',
                       help='Model name: Transformer, DLinear, iTransformer, etc.')

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
                       help='Target feature name (e.g., TPCT for peak cladding temp)')
    parser.add_argument('--freq', type=str, default='s',
                       help='Frequency: s for seconds (10-second sampling interval)')
    parser.add_argument('--checkpoints', type=str,
                       default='./checkpoints',
                       help='Checkpoint save location')

    # Forecasting task parameters
    parser.add_argument('--seq_len', type=int, default=30,
                       help='Input sequence length (30 = 300 seconds = 5 minutes lookback)')
    parser.add_argument('--label_len', type=int, default=15,
                       help='Decoder start token length (15 = 150 seconds)')
    parser.add_argument('--pred_len', type=int, default=30,
                       help='Prediction length (30 = 300 seconds = 5 minutes forecast, 30-step ahead)')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly',
                       help='Seasonal patterns (required by TSLib, not used)')
    parser.add_argument('--inverse', type=int, default=1,
                       help='Inverse output for visualization (1=True, 0=False)')

    # Model architecture
    parser.add_argument('--enc_in', type=int, default=74,
                       help='Encoder input size (74 features after exclusions)')
    parser.add_argument('--dec_in', type=int, default=74,
                       help='Decoder input size (74 features after exclusions)')
    parser.add_argument('--c_out', type=int, default=1,
                       help='Output size (1 for MS mode predicting TPCT)')
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
                       help='Expansion factor for TimesNet')
    parser.add_argument('--d_conv', type=int, default=4,
                       help='Convolution kernel size')

    # Optimization
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Data loader workers')
    parser.add_argument('--train_epochs', type=int, default=20,
                       help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size (8 for 60-timestep total context)')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--loss', type=str, default='MSE',
                       help='Loss function')
    parser.add_argument('--lradj', type=str, default='type1',
                       help='Learning rate adjustment')
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

    # Additional required arguments for TSLib
    parser.add_argument('--des', type=str, default='LOCA_forecast',
                       help='Experiment description')
    parser.add_argument('--itr', type=int, default=1,
                       help='Experiment iterations')
    parser.add_argument('--use_dtw', action='store_true', default=False,
                       help='Use DTW metric for evaluation')

    # Model-specific parameters
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                       help='Hidden dimensions for Nonstationary_Transformer projector')
    parser.add_argument('--p_hidden_layers', type=int, default=2,
                       help='Hidden layers for Nonstationary_Transformer projector')

    # TimesNet parameters
    parser.add_argument('--top_k', type=int, default=5,
                       help='For TimesNet: number of top frequencies to use in FFT')
    parser.add_argument('--num_kernels', type=int, default=6,
                       help='For TimesNet: number of kernels in Inception block')

    # TimeMixer parameters
    parser.add_argument('--down_sampling_layers', type=int, default=3,
                       help='For TimeMixer: number of downsampling layers for multi-scale')
    parser.add_argument('--down_sampling_window', type=int, default=2,
                       help='For TimeMixer: downsampling window size (stride)')
    parser.add_argument('--down_sampling_method', type=str, default='avg',
                       choices=['max', 'avg', 'conv'],
                       help='For TimeMixer: downsampling method')
    parser.add_argument('--channel_independence', type=int, default=0,
                       help='For TimeMixer: 0=shared across channels, 1=independent per channel')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                       choices=['moving_avg', 'dft_decomp'],
                       help='For TimeMixer: decomposition method for trend-season')
    parser.add_argument('--use_norm', type=int, default=1,
                       help='For TimeMixer: 1=use normalization, 0=no normalization')

    # SegRNN parameters
    parser.add_argument('--seg_len', type=int, default=30,
                       help='For SegRNN: segment length (must divide seq_len evenly)')

    # PatchTST and MultiPatchFormer parameters
    parser.add_argument('--patch_len', type=int, default=16,
                       help='For PatchTST/MultiPatchFormer: patch length')

    # Reproducibility
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility (sets random, numpy, torch, torch.cuda)')

    args = parser.parse_args()
    return args


def main():
    """Main execution function"""
    print("=" * 80)
    print("Nuclear LOCA Forecasting - Multi-Scenario Training")
    print("=" * 80)

    # Parse arguments
    args = create_loca_args()

    # Set random seeds for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"  Random seed: {args.seed}")

    # Display configuration
    print(f"\nConfiguration:")
    print(f"  Dataset: {args.data}")
    print(f"  Accident type: {args.accident_type}")
    print(f"  Model: {args.model}")
    print(f"  Sequence length: {args.seq_len} timesteps ({args.seq_len * 10} seconds)")
    print(f"  Prediction length: {args.pred_len} timesteps ({args.pred_len * 10} seconds)")
    print(f"  Features mode: {args.features}")
    print(f"  Target: {args.target}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.train_epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Model dimension: {args.d_model}")
    print(f"  Attention heads: {args.n_heads}")
    print(f"  Encoder layers: {args.e_layers}")

    # Create checkpoint directory
    os.makedirs(args.checkpoints, exist_ok=True)

    # Create experiment setting string (matching TSLib convention)
    # Use seed-based suffix when --seed is provided, otherwise default _0 for backward compatibility
    run_suffix = 'seed{}'.format(args.seed) if args.seed is not None else '0'
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
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
        args.des,
        run_suffix
    )

    # Initialize experiment
    print(f"\nInitializing experiment: {setting}")
    exp = Exp_Long_Term_Forecast(args)

    if args.is_training:
        print("\n" + "=" * 80)
        print("Starting training...")
        print("=" * 80)
        exp.train(setting=setting)

        print("\n" + "=" * 80)
        print("Training complete! Running final test...")
        print("=" * 80)
        exp.test(setting=setting)
    else:
        print("\n" + "=" * 80)
        print("Running test only...")
        print("=" * 80)
        exp.test(setting=setting, test=1)

    print("\n" + "=" * 80)
    print("Experiment complete!")
    print("=" * 80)
    print(f"Results saved to: {args.checkpoints}/{setting}")


if __name__ == '__main__':
    main()
