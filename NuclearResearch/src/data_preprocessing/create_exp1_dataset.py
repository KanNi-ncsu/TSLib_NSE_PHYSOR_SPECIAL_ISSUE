#!/usr/bin/env python3
"""
Experiment 1 Dataset Creation: LOCA + Normal Operation

This script creates the exp1 dataset for multi-horizon nuclear LOCA forecasting.

Training Strategy:
- Train on: 75 LOCA + 20 Normal = 95 scenarios
- Validate on: 10 LOCA = 10 scenarios (NO Normal - matches test distribution!)
- Test on: 15 LOCA = 15 scenarios (NO Normal - pure LOCA prediction)

Scientific Question: Does normal operation data improve LOCA prediction?

Dataset Structure:
- LOCA scenarios: Use ALL 75 from exp0 train split for training
- LOCA scenarios: Use exp0 val split (10 scenarios) for validation
- LOCA scenarios: Use exp0 test split (15 scenarios) for testing
- Normal scenarios: Slice Normal/1.csv into 20 windows (IDs 601-620)
  - Train: 601-620 (20 scenarios)
  - Val: NONE (validation must match test distribution - LOCA only!)
  - Test: NONE (testing LOCA prediction accuracy only!)

Total files: 120 (95 train, 10 val, 15 test)
"""

import os
import pandas as pd
import json
import argparse
import shutil
from pathlib import Path

# Default locations relative to the repository root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_ROOT = str(PROJECT_ROOT / 'NuclearResearch' / 'Raw_data' / 'DONOTMODIFY' / 'Operation_csv_data')
DATASET_ROOT = PROJECT_ROOT / 'dataset'


def create_exp1_dataset(
    raw_data_root=RAW_DATA_ROOT,
    output_root=str(DATASET_ROOT / 'LOCA_normal_train_LOCA_test'),
    loca_metadata_path=str(DATASET_ROOT / 'LOCA_only_train_LOCA_test' / 'split_metadata_lhs.json'),
    num_normal_samples=20,
    normal_window_size=250
):
    """
    Create exp1 dataset with 75 LOCA + 20 Normal for training

    Args:
        raw_data_root: Root directory of all accident type data
        output_root: Output directory for exp1 dataset
        loca_metadata_path: Path to exp0 LOCA split metadata
        num_normal_samples: Number of normal scenarios to create (all go to training!)
        normal_window_size: Length of each normal sample (timesteps)
    """

    # Create output directory
    os.makedirs(output_root, exist_ok=True)

    print("=" * 80)
    print("EXPERIMENT 1 DATASET CREATION: LOCA + Normal Operation")
    print("=" * 80)
    print()

    # Load LOCA metadata from exp0
    with open(loca_metadata_path, 'r') as f:
        loca_metadata = json.load(f)

    loca_train = loca_metadata['train_scenarios']  # 75 scenarios (ALL for training!)
    loca_val = loca_metadata['val_scenarios']      # 10 scenarios (exp0 val split)
    loca_test = loca_metadata['test_scenarios']    # 15 scenarios (exp0 test split)

    print(f"LOCA split (from exp0 metadata):")
    print(f"  Train: {len(loca_train)} scenarios (ALL 75 from exp0 train)")
    print(f"  Val:   {len(loca_val)} scenarios (exp0 val split - LOCA only!)")
    print(f"  Test:  {len(loca_test)} scenarios (exp0 test split)")
    print()

    # Copy LOCA scenarios
    print("Copying LOCA scenarios...")
    print("-" * 80)

    loca_source_path = os.path.join(raw_data_root, 'LOCA')
    all_train = []
    all_val = []
    all_test = []
    scenario_info = {}

    for scenario_id in loca_train:
        source_file = os.path.join(loca_source_path, f'{scenario_id}.csv')
        df = pd.read_csv(source_file)
        output_file = os.path.join(output_root, f'{scenario_id}.csv')
        df.to_csv(output_file, index=False)

        all_train.append(scenario_id)
        scenario_info[scenario_id] = {
            'accident_type': 'LOCA',
            'original_id': scenario_id,
            'split': 'train',
            'timesteps': len(df)
        }

    for scenario_id in loca_val:
        source_file = os.path.join(loca_source_path, f'{scenario_id}.csv')
        df = pd.read_csv(source_file)
        output_file = os.path.join(output_root, f'{scenario_id}.csv')
        df.to_csv(output_file, index=False)

        all_val.append(scenario_id)
        scenario_info[scenario_id] = {
            'accident_type': 'LOCA',
            'original_id': scenario_id,
            'split': 'val',
            'timesteps': len(df)
        }

    for scenario_id in loca_test:
        source_file = os.path.join(loca_source_path, f'{scenario_id}.csv')
        df = pd.read_csv(source_file)
        output_file = os.path.join(output_root, f'{scenario_id}.csv')
        df.to_csv(output_file, index=False)

        all_test.append(scenario_id)
        scenario_info[scenario_id] = {
            'accident_type': 'LOCA',
            'original_id': scenario_id,
            'split': 'test',
            'timesteps': len(df)
        }

    print(f"Copied {len(loca_train)} LOCA train scenarios")
    print(f"Copied {len(loca_val)} LOCA val scenarios")
    print(f"Copied {len(loca_test)} LOCA test scenarios")
    print()

    # Add normal operation samples
    print("Creating normal operation samples...")
    print("-" * 80)

    normal_data_path = os.path.join(raw_data_root, 'Normal', '1.csv')
    normal_df = pd.read_csv(normal_data_path)
    print(f"Normal data source: {normal_data_path}")
    print(f"  Available timesteps: {len(normal_df)}")
    print()

    # Normal samples: 20 train, 0 val, 0 test (ALL for training augmentation!)
    n_train_normal = 20
    n_val_normal = 0
    n_test_normal = 0
    num_normal_samples = n_train_normal  # 20 total

    # Generate normal scenario IDs (601-620)
    normal_start_id = 601
    normal_scenario_ids = list(range(normal_start_id, normal_start_id + num_normal_samples))

    train_normal = normal_scenario_ids  # ALL 20 for training (601-620)
    val_normal = []  # NO Normal in validation!

    # Extract normal windows with stride
    max_start = len(normal_df) - normal_window_size
    stride = max(1, max_start // (num_normal_samples - 1))

    for i, scenario_id in enumerate(normal_scenario_ids):
        start_idx = min(i * stride, max_start)
        end_idx = start_idx + normal_window_size

        normal_window = normal_df.iloc[start_idx:end_idx].copy()
        normal_window['TIME'] = normal_window['TIME'] - normal_window['TIME'].iloc[0]

        output_path = os.path.join(output_root, f'{scenario_id}.csv')
        normal_window.to_csv(output_path, index=False)

        # All Normal scenarios go to training (augmentation only!)
        split = 'train'
        all_train.append(scenario_id)

        scenario_info[scenario_id] = {
            'accident_type': 'Normal',
            'original_id': 1,
            'split': split,
            'timesteps': len(normal_window),
            'source_range': f'{start_idx}-{end_idx}'
        }

    print(f"Created {num_normal_samples} normal samples (IDs {normal_start_id}-{normal_start_id + num_normal_samples - 1})")
    print(f"  Train: {n_train_normal} (IDs 601-620 - ALL for training augmentation!)")
    print(f"  Val:   0 (NO Normal - validation must match test distribution!)")
    print(f"  Test:  0 (NO Normal - testing LOCA prediction accuracy only!)")
    print()

    # Create metadata
    print("Creating metadata...")
    print("-" * 80)

    metadata = {
        'experiment_name': 'exp1_LOCA_normal_train_LOCA_test',
        'dataset_type': 'LOCA_with_normal_operation',
        'accident_types': ['LOCA', 'Normal'],
        'total_scenarios': len(scenario_info),
        'n_train': len(all_train),
        'n_val': len(all_val),
        'n_test': len(all_test),
        'train_scenarios': sorted(all_train),
        'val_scenarios': sorted(all_val),
        'test_scenarios': sorted(all_test),
        'scenario_specs': {
            'target_feature': 'TPCT',
            'target_column_index': 61,
            'num_features': 94
        },
        'raw_data_location': '.',
        'accident_type_details': {
            'LOCA': {
                'description': 'Loss of Coolant Accident (Hot Leg)',
                'num_scenarios': len(loca_train) + len(loca_val) + len(loca_test),
                'id_range': '1-100',
                'train': len(loca_train),
                'val': len(loca_val),
                'test': len(loca_test)
            },
            'Normal': {
                'description': 'Normal Operation',
                'num_scenarios': num_normal_samples,
                'id_range': f'{normal_start_id}-{normal_start_id + num_normal_samples - 1}',
                'train': n_train_normal,
                'val': n_val_normal,
                'test': n_test_normal
            }
        },
        'scientific_question': 'Does normal operation data improve LOCA prediction?',
        'comparison_baseline': 'exp0 (LOCA-only)',
        'rationale': [
            'Test if normal operation data improves LOCA accident prediction',
            'Training: 75 LOCA + 20 Normal = 95 scenarios',
            'Validation: 10 LOCA (NO Normal - matches test distribution!)',
            'Test: 15 LOCA (NO Normal - pure LOCA prediction accuracy)',
            'Normal data only for training augmentation, NOT for validation/test',
            'Validation set now properly represents test distribution',
            'Compare performance with exp0 (LOCA-only baseline)'
        ]
    }

    # Save metadata
    metadata_output = os.path.join(output_root, 'split_metadata_lhs.json')
    with open(metadata_output, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata: {metadata_output}")
    print()

    # Create summary file
    summary_path = os.path.join(output_root, 'dataset_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Experiment 1 Dataset Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write("Scientific Question: Does normal operation data improve LOCA prediction?\n\n")

        f.write("Training Strategy:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Train: {len(loca_train)} LOCA + {n_train_normal} Normal = {len(all_train)} scenarios\n")
        f.write(f"  Val:   {len(loca_val)} LOCA ONLY (NO Normal - matches test distribution!)\n")
        f.write(f"  Test:  {len(loca_test)} LOCA ONLY (NO Normal - pure LOCA prediction accuracy)\n\n")

        f.write("Key Design Decisions:\n")
        f.write("  - LOCA: ALL 75 from exp0 train split for training\n")
        f.write("  - LOCA val: 10 from exp0 val split (LOCA only - matches test!)\n")
        f.write("  - LOCA test: 15 from exp0 test split\n")
        f.write("  - Normal: 20 windows from Normal/1.csv (ALL for training augmentation)\n")
        f.write("  - Normal IDs: 601-620 (training only!)\n")
        f.write("  - CRITICAL: No Normal in val/test - validation must match test distribution!\n\n")

        f.write("Expected Benefits:\n")
        f.write("  - Normal operation data may provide baseline behavior patterns\n")
        f.write("  - Help models distinguish normal vs accident conditions\n")
        f.write("  - Improve generalization through diverse operational states\n\n")

    print(f"Saved summary: {summary_path}")
    print()

    # Print final summary
    print("=" * 80)
    print("EXPERIMENT 1 DATASET CREATION COMPLETE")
    print("=" * 80)
    print()
    print(f"Output directory: {output_root}")
    print(f"Total scenarios: {len(scenario_info)}")
    print()
    print("Split distribution:")
    print(f"  Train: {len(all_train)} scenarios (75 LOCA + 20 Normal)")
    print(f"  Val:   {len(all_val)} scenarios (10 LOCA ONLY - NO Normal!)")
    print(f"  Test:  {len(all_test)} scenarios (15 LOCA ONLY - NO Normal!)")
    print()
    print("CRITICAL: Validation set now matches test distribution (LOCA only)!")
    print()
    print("Ready for training!")
    print()

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description='Create exp1 dataset: LOCA + Normal Operation'
    )
    parser.add_argument(
        '--raw_data_root',
        type=str,
        default=RAW_DATA_ROOT,
        help='Root directory of all accident type data'
    )
    parser.add_argument(
        '--output_root',
        type=str,
        default=str(DATASET_ROOT / 'LOCA_normal_train_LOCA_test'),
        help='Output directory for exp1 dataset'
    )
    parser.add_argument(
        '--loca_metadata',
        type=str,
        default=str(DATASET_ROOT / 'LOCA_only_train_LOCA_test' / 'split_metadata_lhs.json'),
        help='Path to exp0 LOCA split metadata'
    )
    parser.add_argument(
        '--num_normal_samples',
        type=int,
        default=20,
        help='Number of normal operation scenarios to create'
    )
    parser.add_argument(
        '--normal_window_size',
        type=int,
        default=250,
        help='Length of each normal sample (timesteps)'
    )

    args = parser.parse_args()

    create_exp1_dataset(
        raw_data_root=args.raw_data_root,
        output_root=args.output_root,
        loca_metadata_path=args.loca_metadata,
        num_normal_samples=args.num_normal_samples,
        normal_window_size=args.normal_window_size
    )


if __name__ == '__main__':
    main()
