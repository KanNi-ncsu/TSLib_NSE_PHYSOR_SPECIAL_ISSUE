"""
LOCA Zero-Shot Forecasting Script

Runs foundation models (Chronos, Chronos2, Moirai, TimesFM, TiRex, Sundial, TimeMoE)
in zero-shot inference mode on nuclear LOCA accident scenarios.
No training -- uses pretrained weights only.

Uses TSLib infrastructure with Dataset_Nuclear_Accident and Exp_Zero_Shot_Forecast.

Author: Nuclear Research Team
Date: 2026-03-20
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from exp.exp_zero_shot_forecasting import Exp_Zero_Shot_Forecast


def create_zeroshot_args():
    """
    Create argument parser for zero-shot LOCA forecasting.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Nuclear LOCA Zero-Shot Forecasting')

    # Basic config
    parser.add_argument('--task_name', type=str, default='zero_shot_forecast',
                       help='Task name: zero_shot_forecast')
    parser.add_argument('--is_training', type=int, default=0,
                       help='Status: 0 for zero-shot inference only')
    parser.add_argument('--model_id', type=str, default='LOCA_zeroshot',
                       help='Model ID for saving')
    parser.add_argument('--model', type=str, default='Chronos2',
                       help='Foundation model: Chronos, Chronos2, Moirai, TimesFM, TiRex, Sundial, TimeMoE, Chronos_FT, Chronos2_FT')
    parser.add_argument('--finetuned_model_path', type=str, default=None,
                       help='Path to fine-tuned model checkpoint (required for Chronos_FT/Chronos2_FT)')

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
                       help='Prediction length (30 = 300 seconds = 5 minutes forecast)')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly',
                       help='Seasonal patterns (required by TSLib, not used)')
    parser.add_argument('--inverse', type=int, default=1,
                       help='Inverse output for visualization (1=True, 0=False)')

    # Model architecture (needed by exp_basic.py even for zero-shot)
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
                       help='Expansion factor')
    parser.add_argument('--d_conv', type=int, default=4,
                       help='Convolution kernel size')

    # Additional model-specific params (required by arg parser)
    parser.add_argument('--top_k', type=int, default=5, help='For TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='For Inception')
    parser.add_argument('--down_sampling_layers', type=int, default=3, help='For TimeMixer')
    parser.add_argument('--down_sampling_window', type=int, default=2, help='For TimeMixer')
    parser.add_argument('--down_sampling_method', type=str, default='avg', help='For TimeMixer')
    parser.add_argument('--channel_independence', type=int, default=0, help='For TimeMixer')
    parser.add_argument('--decomp_method', type=str, default='moving_avg', help='Decomposition method')
    parser.add_argument('--use_norm', type=int, default=1, help='Use normalization')
    parser.add_argument('--seg_len', type=int, default=30, help='For SegRNN')
    parser.add_argument('--patch_len', type=int, default=16, help='Patch length')
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                       help='Hidden dimensions for projector')
    parser.add_argument('--p_hidden_layers', type=int, default=2,
                       help='Hidden layers for projector')

    # Optimization (minimal for zero-shot)
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Data loader workers')
    parser.add_argument('--itr', type=int, default=1,
                       help='Experiment iterations')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for inference')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                       help='Learning rate (not used in zero-shot)')
    parser.add_argument('--train_epochs', type=int, default=0,
                       help='Training epochs (0 for zero-shot)')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience (not used)')
    parser.add_argument('--loss', type=str, default='MSE', help='Loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='LR adjustment')
    parser.add_argument('--des', type=str, default='zeroshot', help='Experiment description')
    parser.add_argument('--use_amp', action='store_true', default=False,
                       help='Use automatic mixed precision')
    parser.add_argument('--use_dtw', action='store_true', default=False,
                       help='Use DTW metric')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='Use GPU')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='GPU type: cuda or mps')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False,
                       help='Use multiple GPUs')
    parser.add_argument('--devices', type=str, default='0,1,2,3',
                       help='Device IDs for multi-GPU')

    args = parser.parse_args()
    return args


def main():
    """Main execution function"""
    print("=" * 80)
    print("Nuclear LOCA Zero-Shot Forecasting - Foundation Models")
    print("=" * 80)

    args = create_zeroshot_args()

    print(f"\nConfiguration:")
    print(f"  Task: {args.task_name}")
    print(f"  Dataset: {args.data}")
    print(f"  Data path: {args.data_path}")
    print(f"  Model: {args.model}")
    print(f"  Sequence length: {args.seq_len} timesteps ({args.seq_len * 10} seconds)")
    print(f"  Prediction length: {args.pred_len} timesteps ({args.pred_len * 10} seconds)")
    print(f"  Features mode: {args.features}")
    print(f"  Target: {args.target}")
    print(f"  Batch size: {args.batch_size}")

    # Create setting string (matching TSLib convention)
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

    # Initialize experiment
    print(f"\nInitializing zero-shot experiment: {setting}")
    exp = Exp_Zero_Shot_Forecast(args)

    print("\n" + "=" * 80)
    print("Running zero-shot inference (no training)...")
    print("=" * 80)
    exp.test(setting=setting, test=1)

    print("\n" + "=" * 80)
    print("Zero-shot inference complete!")
    print("=" * 80)
    print(f"Results saved to: ./results/{setting}/")


if __name__ == '__main__':
    main()
