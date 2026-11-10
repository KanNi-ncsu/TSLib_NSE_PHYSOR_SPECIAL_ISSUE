#!/usr/bin/env python3
"""
Experiment 2 Dataset Creation: LOCAC + Normal (NO LOCA in training!)

This script creates the exp2 dataset for multi-horizon nuclear LOCA forecasting.

Training Strategy:
- Train on: 75 LOCAC (matched IDs) + 20 Normal = 95 scenarios (NO LOCA!)
- Validate on: 10 LOCAC (matched IDs) = 10 scenarios (NO LOCA, NO Normal!)
- Test on: 15 LOCA = 15 scenarios (same test set as all experiments)

Scientific Question: Does cross-accident transfer learning (LOCAC) help LOCA prediction?

CRITICAL: This experiment tests PURE cross-accident transfer learning.
- NO LOCA scenarios in training data
- LOCAC (cold leg) provides similar but distinct physics
- Uses SAME scenario IDs as exp0's 75 LOCA for controlled comparison
- Tests if model can transfer knowledge from LOCAC to predict LOCA

Dataset Structure:
- LOCAC scenarios: Use SAME IDs as exp0's LOCA split (controlled transfer)
  - Train: LOCAC scenarios 1-75 (IDs offset to 201-275)
  - Val: LOCAC scenarios 76-85 (IDs offset to 276-285)
- LOCA scenarios: exp0 test split (15 scenarios) for testing ONLY
- Normal scenarios: Slice Normal/1.csv into 20 windows (IDs 601-620)
  - Train: 601-620 (20 scenarios)
  - Val: NONE (validation uses LOCAC only!)
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


def create_exp2_dataset(
    raw_data_root=RAW_DATA_ROOT,
    output_root=str(DATASET_ROOT / 'normal_LOCAC_train_LOCA_test'),
    loca_metadata_path=str(DATASET_ROOT / 'LOCA_only_train_LOCA_test' / 'split_metadata_lhs.json'),
    num_normal_samples=20,
    normal_window_size=250
):
    """
    Create exp2 dataset with 75 LOCAC (matched IDs) + 20 Normal (NO LOCA in training!)

    Args:
        raw_data_root: Root directory of all accident type data
        output_root: Output directory for exp2 dataset
        loca_metadata_path: Path to exp0 LOCA split metadata (for LOCA test split)
        num_normal_samples: Number of normal scenarios to create (all go to training!)
        normal_window_size: Length of each normal sample (timesteps)
    """

    # Create output directory
    os.makedirs(output_root, exist_ok=True)

    print("=" * 80)
    print("EXPERIMENT 2 DATASET CREATION: LOCAC + Normal (NO LOCA in training!)")
    print("=" * 80)
    print()
    print("CRITICAL: This tests pure cross-accident transfer learning")
    print("          ZERO LOCA scenarios in training data")
    print()

    # Load LOCA metadata from exp0 for test split ONLY
    with open(loca_metadata_path, 'r') as f:
        loca_metadata = json.load(f)

    loca_train_ids = loca_metadata['train_scenarios']  # 75 scenarios (for ID matching!)
    loca_val_ids = loca_metadata['val_scenarios']      # 10 scenarios (for ID matching!)
    loca_test = loca_metadata['test_scenarios']        # 15 scenarios (for testing)

    print(f"LOCA splits (from exp0 - for reference and test):")
    print(f"  Train IDs: {len(loca_train_ids)} (will use SAME IDs from LOCAC)")
    print(f"  Val IDs:   {len(loca_val_ids)} (will use SAME IDs from LOCAC)")
    print(f"  Test:      {len(loca_test)} scenarios (for testing LOCA prediction)")
    print()
    print("CRITICAL: ZERO LOCA in training/validation - pure cross-accident transfer!")
    print()

    all_train = []
    all_val = []
    all_test = []
    scenario_info = {}

    # Copy LOCAC scenarios matching LOCA IDs (controlled transfer learning)
    print("Copying LOCAC scenarios (SAME IDs as exp0 LOCA for controlled transfer)...")
    print("-" * 80)

    locac_source_path = os.path.join(raw_data_root, 'LOCAC')
    locac_offset = 200  # Map LOCAC IDs to 201-300 range

    # Use SAME scenario IDs as exp0's LOCA split
    locac_train_ids = loca_train_ids  # Use exact same IDs (1-75)
    locac_val_ids = loca_val_ids      # Use exact same IDs (76-85)

    print(f"LOCAC train: Using scenarios {min(locac_train_ids)}-{max(locac_train_ids)} (matched to LOCA)")
    print(f"LOCAC val:   Using scenarios {min(locac_val_ids)}-{max(locac_val_ids)} (matched to LOCA)")

    # Copy LOCAC training scenarios
    for source_id in locac_train_ids:
        source_file = os.path.join(locac_source_path, f'{source_id}.csv')
        df = pd.read_csv(source_file)
        new_id = source_id + locac_offset  # 1 -> 201, 75 -> 275
        output_file = os.path.join(output_root, f'{new_id}.csv')
        df.to_csv(output_file, index=False)

        all_train.append(new_id)
        scenario_info[new_id] = {
            'accident_type': 'LOCAC',
            'original_id': source_id,
            'split': 'train',
            'timesteps': len(df)
        }

    # Copy LOCAC validation scenarios
    for source_id in locac_val_ids:
        source_file = os.path.join(locac_source_path, f'{source_id}.csv')
        df = pd.read_csv(source_file)
        new_id = source_id + locac_offset  # 76 -> 276, 85 -> 285
        output_file = os.path.join(output_root, f'{new_id}.csv')
        df.to_csv(output_file, index=False)

        all_val.append(new_id)
        scenario_info[new_id] = {
            'accident_type': 'LOCAC',
            'original_id': source_id,
            'split': 'val',
            'timesteps': len(df)
        }

    print(f"Copied {len(locac_train_ids)} LOCAC train scenarios (IDs {min(locac_train_ids)+locac_offset}-{max(locac_train_ids)+locac_offset})")
    print(f"Copied {len(locac_val_ids)} LOCAC val scenarios (IDs {min(locac_val_ids)+locac_offset}-{max(locac_val_ids)+locac_offset})")
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
    print(f"  Val:   0 (NO Normal - validation uses LOCAC only!)")
    print(f"  Test:  0 (NO Normal - testing LOCA prediction accuracy only!)")
    print()

    # Create metadata
    print("Creating metadata...")
    print("-" * 80)

    metadata = {
        'experiment_name': 'exp2_LOCAC_normal_train_LOCA_test',
        'dataset_type': 'cross_accident_transfer_learning',
        'accident_types': ['LOCAC', 'LOCA', 'Normal'],
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
            'LOCAC': {
                'description': 'Loss of Coolant Accident (Cold Leg) - Matched IDs to LOCA',
                'num_scenarios': len(locac_train_ids) + len(locac_val_ids),
                'id_range': f'{min(locac_train_ids)+locac_offset}-{max(locac_val_ids)+locac_offset} (offset from {min(locac_train_ids)}-{max(locac_val_ids)})',
                'train': len(locac_train_ids),
                'val': len(locac_val_ids),
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
        'scientific_question': 'Does cross-accident transfer learning (LOCAC) help LOCA prediction?',
        'comparison_baseline': 'exp0 (LOCA-only) and exp1 (LOCA + Normal)',
        'critical_note': 'ZERO LOCA in training/validation - pure cross-accident transfer learning test',
        'rationale': [
            'Test if LOCAC (cold leg) physics transfers to LOCA (hot leg) prediction',
            'Training: 75 LOCAC (matched IDs) + 20 Normal = 95 scenarios (NO LOCA!)',
            'Validation: 10 LOCAC (matched IDs - monitors cross-accident learning!)',
            'Test: 15 LOCA (same test set as all experiments)',
            'LOCAC uses SAME scenario IDs as exp0 LOCA for controlled comparison',
            'LOCAC provides similar but distinct thermal-hydraulic behavior',
            'Validation set matches training distribution (LOCAC) not test (LOCA)',
            'Compare with exp0 and exp1 to quantify value of cross-accident transfer'
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
        f.write("Experiment 2 Dataset Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write("Scientific Question: Does cross-accident transfer learning (LOCAC) help?\n\n")
        f.write("CRITICAL: ZERO LOCA in training - pure cross-accident transfer test\n\n")

        f.write("Training Strategy:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Train: {len(locac_train_ids)} LOCAC + {n_train_normal} Normal = {len(all_train)} scenarios\n")
        f.write(f"  Val:   {len(locac_val_ids)} LOCAC ONLY (NO LOCA, NO Normal!)\n")
        f.write(f"  Test:  {len(loca_test)} LOCA ONLY (NO Normal - pure LOCA prediction accuracy)\n\n")

        f.write("Key Design Decisions:\n")
        f.write(f"  - LOCAC: {len(locac_train_ids)} train + {len(locac_val_ids)} val (SAME IDs as exp0 LOCA)\n")
        f.write(f"  - LOCAC train IDs: {min(locac_train_ids)+locac_offset}-{max(locac_train_ids)+locac_offset}\n")
        f.write(f"  - LOCAC val IDs: {min(locac_val_ids)+locac_offset}-{max(locac_val_ids)+locac_offset}\n")
        f.write("  - LOCA: ZERO in training/validation, exp0 test split for testing only\n")
        f.write("  - Normal: 20 windows from Normal/1.csv (ALL for training augmentation)\n")
        f.write("  - Normal IDs: 601-620 (training only!)\n")
        f.write("  - CRITICAL: Validation uses LOCAC (matches training), test uses LOCA!\n\n")

        f.write("Expected Insights:\n")
        f.write("  - Can model learn LOCA physics from LOCAC data alone?\n")
        f.write("  - How much performance degrades without direct LOCA training?\n")
        f.write("  - Compare with exp0 (LOCA baseline) and exp1 (LOCA + Normal)\n")
        f.write("  - Quantify value of cross-accident transfer learning\n\n")

    print(f"Saved summary: {summary_path}")
    print()

    # Print final summary
    print("=" * 80)
    print("EXPERIMENT 2 DATASET CREATION COMPLETE")
    print("=" * 80)
    print()
    print(f"Output directory: {output_root}")
    print(f"Total scenarios: {len(scenario_info)}")
    print()
    print("Split distribution:")
    print(f"  Train: {len(all_train)} scenarios ({len(locac_train_ids)} LOCAC + {n_train_normal} Normal - NO LOCA!)")
    print(f"  Val:   {len(all_val)} scenarios ({len(locac_val_ids)} LOCAC ONLY - NO LOCA, NO Normal!)")
    print(f"  Test:  {len(all_test)} scenarios (15 LOCA ONLY - NO Normal!)")
    print()
    print("CRITICAL NOTES:")
    print("  - ZERO LOCA in training/validation (pure cross-accident transfer)")
    print("  - LOCAC IDs matched to exp0 LOCA IDs (controlled comparison)")
    print("  - Validation monitors LOCAC learning, test evaluates LOCA transfer")
    print()
    print("Ready for training!")
    print()

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description='Create exp2 dataset: LOCAC + Normal (NO LOCA in training!)'
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
        default=str(DATASET_ROOT / 'normal_LOCAC_train_LOCA_test'),
        help='Output directory for exp2 dataset'
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

    create_exp2_dataset(
        raw_data_root=args.raw_data_root,
        output_root=args.output_root,
        loca_metadata_path=args.loca_metadata,
        num_normal_samples=args.num_normal_samples,
        normal_window_size=args.normal_window_size
    )


if __name__ == '__main__':
    main()
