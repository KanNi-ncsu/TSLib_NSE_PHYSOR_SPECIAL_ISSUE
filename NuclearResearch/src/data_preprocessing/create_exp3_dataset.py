#!/usr/bin/env python3
"""
Experiment 3 Dataset Creation: FLB + Normal (NO LOCA in training!)

This script creates the exp3 dataset for multi-horizon nuclear LOCA forecasting.

Training Strategy:
- Train on: 75 FLB (matched IDs) + 20 Normal = 95 scenarios (NO LOCA!)
- Validate on: 10 FLB (matched IDs) = 10 scenarios (NO LOCA, NO Normal!)
- Test on: 15 LOCA = 15 scenarios (same test set as all experiments)

Scientific Question: Does FLB cross-accident transfer (different physics) help LOCA prediction?

CRITICAL: This experiment tests cross-accident transfer with DIFFERENT physics.
- NO LOCA scenarios in training data
- FLB (Feedline Break) has different thermal-hydraulic characteristics than LOCA
- Uses SAME scenario IDs as exp0's 75 LOCA for controlled comparison
- Tests if model can transfer knowledge across dissimilar accident types

Dataset Structure:
- FLB scenarios: Use SAME IDs as exp0's LOCA split (controlled transfer)
  - Train: FLB scenarios 1-75 (IDs offset to 101-175)
  - Val: FLB scenarios 76-85 (IDs offset to 176-185)
- LOCA scenarios: exp0 test split (15 scenarios) for testing ONLY
- Normal scenarios: Slice Normal/1.csv into 20 windows (IDs 601-620)
  - Train: 601-620 (20 scenarios)
  - Val: NONE (validation uses FLB only!)
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


def create_exp3_dataset(
    raw_data_root=RAW_DATA_ROOT,
    output_root=str(DATASET_ROOT / 'LOCA_FLB_normal_train_LOCA_test'),
    loca_metadata_path=str(DATASET_ROOT / 'LOCA_only_train_LOCA_test' / 'split_metadata_lhs.json'),
    num_normal_samples=20,
    normal_window_size=250
):
    """
    Create exp3 dataset with 75 FLB (matched IDs) + 20 Normal (NO LOCA in training!)

    Args:
        raw_data_root: Root directory of all accident type data
        output_root: Output directory for exp3 dataset
        loca_metadata_path: Path to exp0 LOCA split metadata (for LOCA test split)
        num_normal_samples: Number of normal scenarios to create (all go to training!)
        normal_window_size: Length of each normal sample (timesteps)
    """

    # Create output directory
    os.makedirs(output_root, exist_ok=True)

    print("=" * 80)
    print("EXPERIMENT 3 DATASET CREATION: FLB + Normal (NO LOCA in training!)")
    print("=" * 80)
    print()
    print("CRITICAL: This tests cross-accident transfer with DIFFERENT physics")
    print("          ZERO LOCA scenarios in training data")
    print()

    # Load LOCA metadata from exp0 for test split ONLY
    with open(loca_metadata_path, 'r') as f:
        loca_metadata = json.load(f)

    loca_train_ids = loca_metadata['train_scenarios']  # 75 scenarios (for ID matching!)
    loca_val_ids = loca_metadata['val_scenarios']      # 10 scenarios (for ID matching!)
    loca_test = loca_metadata['test_scenarios']        # 15 scenarios (for testing)

    print(f"LOCA splits (from exp0 - for reference and test):")
    print(f"  Train IDs: {len(loca_train_ids)} (will use SAME IDs from FLB)")
    print(f"  Val IDs:   {len(loca_val_ids)} (will use SAME IDs from FLB)")
    print(f"  Test:      {len(loca_test)} scenarios (for testing LOCA prediction)")
    print()
    print("CRITICAL: ZERO LOCA in training/validation - pure cross-accident transfer!")
    print()

    all_train = []
    all_val = []
    all_test = []
    scenario_info = {}

    # Copy FLB scenarios matching LOCA IDs (controlled transfer learning)
    print("Copying FLB scenarios (SAME IDs as exp0 LOCA for controlled transfer)...")
    print("-" * 80)

    flb_source_path = os.path.join(raw_data_root, 'FLB')
    flb_offset = 100  # Map FLB IDs to 101-200 range

    # Use SAME scenario IDs as exp0's LOCA split
    flb_train_ids = loca_train_ids  # Use exact same IDs (1-75)
    flb_val_ids = loca_val_ids      # Use exact same IDs (76-85)

    print(f"FLB train: Using scenarios {min(flb_train_ids)}-{max(flb_train_ids)} (matched to LOCA)")
    print(f"FLB val:   Using scenarios {min(flb_val_ids)}-{max(flb_val_ids)} (matched to LOCA)")

    # Copy FLB training scenarios
    for source_id in flb_train_ids:
        source_file = os.path.join(flb_source_path, f'{source_id}.csv')
        df = pd.read_csv(source_file)
        new_id = source_id + flb_offset  # 1 -> 101, 75 -> 175
        output_file = os.path.join(output_root, f'{new_id}.csv')
        df.to_csv(output_file, index=False)

        all_train.append(new_id)
        scenario_info[new_id] = {
            'accident_type': 'FLB',
            'original_id': source_id,
            'split': 'train',
            'timesteps': len(df)
        }

    # Copy FLB validation scenarios
    for source_id in flb_val_ids:
        source_file = os.path.join(flb_source_path, f'{source_id}.csv')
        df = pd.read_csv(source_file)
        new_id = source_id + flb_offset  # 76 -> 176, 85 -> 185
        output_file = os.path.join(output_root, f'{new_id}.csv')
        df.to_csv(output_file, index=False)

        all_val.append(new_id)
        scenario_info[new_id] = {
            'accident_type': 'FLB',
            'original_id': source_id,
            'split': 'val',
            'timesteps': len(df)
        }

    print(f"Copied {len(flb_train_ids)} FLB train scenarios (IDs {min(flb_train_ids)+flb_offset}-{max(flb_train_ids)+flb_offset})")
    print(f"Copied {len(flb_val_ids)} FLB val scenarios (IDs {min(flb_val_ids)+flb_offset}-{max(flb_val_ids)+flb_offset})")
    print()

    # Copy LOCA scenarios for TESTING ONLY (NO LOCA in training/validation!)
    print("Copying LOCA scenarios (for TESTING ONLY - NO LOCA in training/val!)...")
    print("-" * 80)

    loca_source_path = os.path.join(raw_data_root, 'LOCA')

    # LOCA only for test split
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

    print(f"Copied {len(loca_test)} LOCA test scenarios (TESTING ONLY!)")
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
    print(f"  Val:   0 (NO Normal - validation uses FLB only!)")
    print(f"  Test:  0 (NO Normal - testing LOCA prediction accuracy only!)")
    print()

    # Create metadata
    print("Creating metadata...")
    print("-" * 80)

    metadata = {
        'experiment_name': 'exp3_FLB_normal_train_LOCA_test',
        'dataset_type': 'cross_accident_transfer_different_physics',
        'accident_types': ['FLB', 'LOCA', 'Normal'],
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
            'FLB': {
                'description': 'Feedline Break - Matched IDs to LOCA (Different Physics)',
                'num_scenarios': len(flb_train_ids) + len(flb_val_ids),
                'id_range': f'{min(flb_train_ids)+flb_offset}-{max(flb_val_ids)+flb_offset} (offset from {min(flb_train_ids)}-{max(flb_val_ids)})',
                'train': len(flb_train_ids),
                'val': len(flb_val_ids),
                'test': 0
            },
            'LOCA': {
                'description': 'Loss of Coolant Accident (Hot Leg) - Testing Only',
                'num_scenarios': len(loca_test),
                'id_range': '1-100',
                'train': 0,
                'val': 0,
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
        'scientific_question': 'Does FLB cross-accident transfer (different physics) help LOCA prediction?',
        'comparison_baseline': 'exp0, exp1, and exp2 (LOCAC has similar physics)',
        'critical_note': 'ZERO LOCA in training/validation - tests transfer from DIFFERENT accident physics',
        'rationale': [
            'Test if FLB (feedline break) physics transfers to LOCA prediction',
            'Training: 75 FLB (matched IDs) + 20 Normal = 95 scenarios (NO LOCA!)',
            'Validation: 10 FLB (matched IDs - monitors cross-accident learning!)',
            'Test: 15 LOCA (same test set as all experiments)',
            'FLB uses SAME scenario IDs as exp0 LOCA for controlled comparison',
            'FLB has DIFFERENT thermal-hydraulic characteristics than LOCA',
            'Validation set matches training distribution (FLB) not test (LOCA)',
            'Compare with exp2 (LOCAC - similar physics) to assess physics-based transfer',
            'If FLB performs similarly to LOCAC, suggests generic accident modeling'
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
        f.write("Experiment 3 Dataset Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write("Scientific Question: Does FLB transfer (different physics) help LOCA?\n\n")
        f.write("CRITICAL: ZERO LOCA in training - tests transfer from DIFFERENT physics\n\n")

        f.write("Training Strategy:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Train: {len(flb_train_ids)} FLB + {n_train_normal} Normal = {len(all_train)} scenarios\n")
        f.write(f"  Val:   {len(flb_val_ids)} FLB ONLY (NO LOCA, NO Normal!)\n")
        f.write(f"  Test:  {len(loca_test)} LOCA ONLY (NO Normal - pure LOCA prediction accuracy)\n\n")

        f.write("Key Design Decisions:\n")
        f.write(f"  - FLB: {len(flb_train_ids)} train + {len(flb_val_ids)} val (SAME IDs as exp0 LOCA)\n")
        f.write(f"  - FLB train IDs: {min(flb_train_ids)+flb_offset}-{max(flb_train_ids)+flb_offset}\n")
        f.write(f"  - FLB val IDs: {min(flb_val_ids)+flb_offset}-{max(flb_val_ids)+flb_offset}\n")
        f.write("  - LOCA: ZERO in training/validation, exp0 test split for testing only\n")
        f.write("  - Normal: 20 windows from Normal/1.csv (ALL for training augmentation)\n")
        f.write("  - Normal IDs: 601-620 (training only!)\n")
        f.write("  - CRITICAL: Validation uses FLB (matches training), test uses LOCA!\n\n")

        f.write("Expected Insights:\n")
        f.write("  - Can model transfer from FLB (different physics) to LOCA?\n")
        f.write("  - Compare with exp2 (LOCAC similar physics) to assess physics similarity\n")
        f.write("  - If FLB ~ LOCAC performance: generic accident modeling capability\n")
        f.write("  - If FLB << LOCAC performance: physics-specific features dominate\n")
        f.write("  - Quantify value of physics-based vs random transfer learning\n\n")

    print(f"Saved summary: {summary_path}")
    print()

    # Print final summary
    print("=" * 80)
    print("EXPERIMENT 3 DATASET CREATION COMPLETE")
    print("=" * 80)
    print()
    print(f"Output directory: {output_root}")
    print(f"Total scenarios: {len(scenario_info)}")
    print()
    print("Split distribution:")
    print(f"  Train: {len(all_train)} scenarios ({len(flb_train_ids)} FLB + {n_train_normal} Normal - NO LOCA!)")
    print(f"  Val:   {len(all_val)} scenarios ({len(flb_val_ids)} FLB ONLY - NO LOCA, NO Normal!)")
    print(f"  Test:  {len(all_test)} scenarios (15 LOCA ONLY - NO Normal!)")
    print()
    print("CRITICAL NOTES:")
    print("  - ZERO LOCA in training/validation (pure cross-accident transfer)")
    print("  - FLB IDs matched to exp0 LOCA IDs (controlled comparison)")
    print("  - Validation monitors FLB learning, test evaluates LOCA transfer")
    print("  - FLB has DIFFERENT physics than LOCA (unlike LOCAC)")
    print()
    print("Ready for training!")
    print()

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description='Create exp3 dataset: FLB + Normal (NO LOCA in training!)'
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
        default=str(DATASET_ROOT / 'LOCA_FLB_normal_train_LOCA_test'),
        help='Output directory for exp3 dataset'
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

    create_exp3_dataset(
        raw_data_root=args.raw_data_root,
        output_root=args.output_root,
        loca_metadata_path=args.loca_metadata,
        num_normal_samples=args.num_normal_samples,
        normal_window_size=args.normal_window_size
    )


if __name__ == '__main__':
    main()
