#!/usr/bin/env python3
"""
Flexible 300s-ahead visualization that works with any dataset.

This script automatically detects the dataset path from result folder name
and uses the appropriate metadata and raw data paths.

Usage:
    python visualize_300s_ahead_flexible.py --result_folder <result_folder>
"""

import os
import sys
import argparse
from pathlib import Path

# Derive project root from script location
# Script is at: <PROJECT_ROOT>/NuclearResearch/src/result_process/
PROJECT_ROOT = Path(os.path.abspath(__file__)).parents[3]

# Import the main visualization script
sys.path.append(os.path.dirname(__file__))
import visualize_300s_ahead as viz


def detect_dataset_from_result_folder(result_folder):
    """
    Detect dataset type from result folder name.

    Returns:
        (data_root, metadata_path) tuple
    """
    folder_name = os.path.basename(result_folder)
    dataset_dir = str(PROJECT_ROOT / 'dataset')
    raw_loca_dir = str(PROJECT_ROOT / 'NuclearResearch' / 'Raw_data' / 'DONOTMODIFY' / 'Operation_csv_data' / 'LOCA')

    # Check for exp4 (multi-accident)
    if 'exp4' in folder_name.lower() or 'multi_accident' in folder_name.lower():
        data_root = os.path.join(dataset_dir, 'normal_LOCAC_RW_LOF_SGATR_train_LOCA_test')
        metadata_path = os.path.join(data_root, 'split_metadata_lhs.json')
        return raw_loca_dir, metadata_path

    # Check for exp3 (FLB + normal)
    elif 'exp3' in folder_name.lower() or 'flb_normal' in folder_name.lower():
        data_root = os.path.join(dataset_dir, 'LOCA_FLB_normal_train_LOCA_test')
        metadata_path = os.path.join(data_root, 'split_metadata_lhs.json')
        return raw_loca_dir, metadata_path

    # Check for exp2 (LOCAC + normal - same as exp1 dataset)
    elif 'exp2' in folder_name.lower() or 'locac_normal' in folder_name.lower():
        data_root = os.path.join(dataset_dir, 'normal_LOCAC_train_LOCA_test')
        metadata_path = os.path.join(data_root, 'split_metadata_lhs.json')
        return raw_loca_dir, metadata_path

    # Check for exp1 (LOCA + normal)
    elif 'exp1' in folder_name.lower() or 'loca_normal' in folder_name.lower():
        data_root = os.path.join(dataset_dir, 'normal_LOCAC_train_LOCA_test')
        metadata_path = os.path.join(data_root, 'split_metadata_lhs.json')
        return raw_loca_dir, metadata_path

    # Check for exp0 (LOCA_only)
    elif 'exp0' in folder_name.lower() or 'loca_only' in folder_name.lower():
        data_root = os.path.join(dataset_dir, 'LOCA_only_train_LOCA_test')
        metadata_path = os.path.join(data_root, 'split_metadata_lhs.json')
        return raw_loca_dir, metadata_path

    # Default: exp0 split
    else:
        data_root = os.path.join(dataset_dir, 'LOCA_only_train_LOCA_test')
        metadata_path = os.path.join(data_root, 'split_metadata_lhs.json')
        return raw_loca_dir, metadata_path


def main():
    parser = argparse.ArgumentParser(description='300s-ahead LOCA Visualization (Flexible)')
    parser.add_argument('--result_folder', type=str, required=True,
                       help='Path to results folder')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: result_folder/300s_ahead_plots)')
    parser.add_argument('--seq_len', type=int, default=30,
                       help='Sequence length used in training')
    parser.add_argument('--pred_len', type=int, default=30,
                       help='Prediction length (30 for 300s-ahead)')
    parser.add_argument('--exp_name', type=str, default="",
                       help='Experiment name for plot titles')
    parser.add_argument('--model_name', type=str, default="",
                       help='Model name for plot titles and file naming')

    args = parser.parse_args()

    # Detect dataset paths
    raw_data_root, metadata_path = detect_dataset_from_result_folder(args.result_folder)

    print(f"Detected dataset paths:")
    print(f"  Raw data root: {raw_data_root}")
    print(f"  Metadata path: {metadata_path}")
    print()

    # Check if paths exist
    if not os.path.exists(metadata_path):
        print(f"ERROR: Metadata file not found: {metadata_path}")
        sys.exit(1)

    if not os.path.exists(raw_data_root):
        print(f"ERROR: Raw data root not found: {raw_data_root}")
        sys.exit(1)

    # Monkey-patch the visualization module to use custom paths
    def patched_load_scenario_data(scenario_id, data_root=raw_data_root):
        csv_path = os.path.join(data_root, f'{scenario_id}.csv')
        import pandas as pd
        df = pd.read_csv(csv_path)
        return df

    def patched_load_test_metadata(meta_path=metadata_path):
        import json
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        return metadata['test_scenarios']

    # Replace functions in viz module
    viz.load_scenario_data = patched_load_scenario_data
    viz.load_test_metadata = patched_load_test_metadata

    # Modify sys.argv to pass args to viz.main()
    # The core script will re-parse these
    sys.argv = [
        'visualize_300s_ahead.py',
        '--result_folder', args.result_folder,
        '--seq_len', str(args.seq_len),
        '--pred_len', str(args.pred_len)
    ]

    if args.output_dir:
        sys.argv.extend(['--output_dir', args.output_dir])
    if args.exp_name:
        sys.argv.extend(['--exp_name', args.exp_name])
    if args.model_name:
        sys.argv.extend(['--model_name', args.model_name])

    # Call the original main
    viz.main()


if __name__ == '__main__':
    main()
