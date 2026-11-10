#!/usr/bin/env python3
"""
Experiment 4 Dataset Creation: Maximum BALANCED Accident Diversity

This script creates the exp4 dataset for multi-horizon nuclear LOCA forecasting.

Training Strategy:
- Train on: 75 LOCA + 75 LOCAC + 75 FLB + 75 RW + 75 SGATR + 20 Normal = 395 scenarios
- Validate on: 10 LOCAC + 10 FLB + 10 RW + 10 SGATR = 40 scenarios (NO LOCA, NO Normal!)
- Test on: 15 LOCA = 15 scenarios (same test set as all experiments)

Scientific Question: Does maximum BALANCED accident diversity improve LOCA prediction?

CRITICAL: This experiment uses BALANCED representation (75 each) for fair comparison.
- ALL accident types use SAME scenario IDs (1-75 for train, 76-85 for val)
- Enables controlled assessment of individual accident type contributions
- Validation monitors cross-accident learning (NO LOCA!)

Dataset Structure:
- LOCA scenarios: exp0 splits (75 train, 0 val, 15 test) - IDs unchanged
- LOCAC scenarios: Matched IDs (75 train, 10 val) - IDs offset to 201-285
- FLB scenarios: Matched IDs (75 train, 10 val) - IDs offset to 101-185
- RW scenarios: Matched IDs (75 train, 10 val) - IDs offset to 301-385
- SGATR scenarios: Matched IDs (75 train, 10 val) - IDs offset to 501-585
- Normal scenarios: IDs 601-620 (20 train, 0 val, 0 test)

Total files: 450 (395 train, 40 val, 15 test)
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


def create_exp4_dataset(
    raw_data_root=RAW_DATA_ROOT,
    output_root=str(DATASET_ROOT / 'normal_LOCAC_RW_LOF_SGATR_train_LOCA_test'),
    loca_metadata_path=str(DATASET_ROOT / 'LOCA_only_train_LOCA_test' / 'split_metadata_lhs.json'),
    num_normal_samples=20,
    normal_window_size=250
):
    """
    Create exp4 dataset with maximum BALANCED accident diversity

    Args:
        raw_data_root: Root directory of all accident type data
        output_root: Output directory for exp4 dataset
        loca_metadata_path: Path to exp0 LOCA split metadata (for ID matching)
        num_normal_samples: Number of normal scenarios to create (all go to training!)
        normal_window_size: Length of each normal sample (timesteps)
    """

    # Create output directory
    os.makedirs(output_root, exist_ok=True)

    print("=" * 80)
    print("EXPERIMENT 4 DATASET CREATION: Maximum BALANCED Accident Diversity")
    print("=" * 80)
    print()
    print("Goal: Test if BALANCED diversity (75 each) of accident types improves LOCA")
    print()

    # Load LOCA metadata from exp0 for ID matching
    with open(loca_metadata_path, 'r') as f:
        loca_metadata = json.load(f)

    loca_train_ids = loca_metadata['train_scenarios']  # 75 scenarios (for LOCA train & ID matching!)
    loca_val_ids = loca_metadata['val_scenarios']      # 10 scenarios (for cross-accident val matching!)
    loca_test = loca_metadata['test_scenarios']        # 15 scenarios (for testing)

    print(f"LOCA splits (from exp0 metadata):")
    print(f"  Train IDs: {len(loca_train_ids)} (ALL for LOCA training + used for matching other accidents)")
    print(f"  Val IDs:   {len(loca_val_ids)} (used for matching cross-accident validation)")
    print(f"  Test:      {len(loca_test)} scenarios (for testing LOCA prediction)")
    print()
    print("CRITICAL: ALL accident types use SAME scenario IDs for balanced comparison!")
    print()

    all_train = []
    all_val = []
    all_test = []
    scenario_info = {}

    # Define accident type configurations (ALL use SAME IDs for balanced comparison!)
    accident_types = {
        'LOCA': {
            'path': os.path.join(raw_data_root, 'LOCA'),
            'id_offset': 0,  # Keep original IDs
            'train_ids': loca_train_ids,  # 75 scenarios
            'val_ids': [],  # NO LOCA in validation
            'test_ids': loca_test  # 15 scenarios
        },
        'LOCAC': {
            'path': os.path.join(raw_data_root, 'LOCAC'),
            'id_offset': 200,  # IDs 201-285
            'train_ids': loca_train_ids,  # SAME 75 IDs as LOCA!
            'val_ids': loca_val_ids,  # SAME 10 IDs for val
            'test_ids': []  # NO LOCAC in test
        },
        'FLB': {
            'path': os.path.join(raw_data_root, 'FLB'),
            'id_offset': 100,  # IDs 101-185
            'train_ids': loca_train_ids,  # SAME 75 IDs as LOCA!
            'val_ids': loca_val_ids,  # SAME 10 IDs for val
            'test_ids': []  # NO FLB in test
        },
        'RW': {
            'path': os.path.join(raw_data_root, 'RW'),
            'id_offset': 300,  # IDs 301-385
            'train_ids': loca_train_ids,  # SAME 75 IDs as LOCA!
            'val_ids': loca_val_ids,  # SAME 10 IDs for val
            'test_ids': []  # NO RW in test
        },
        'SGATR': {
            'path': os.path.join(raw_data_root, 'SGATR'),
            'id_offset': 500,  # IDs 501-585
            'train_ids': loca_train_ids,  # SAME 75 IDs as LOCA!
            'val_ids': loca_val_ids,  # SAME 10 IDs for val
            'test_ids': []  # NO SGATR in test
        }
    }

    # Copy accident scenarios (ALL with BALANCED representation!)
    print("Copying accident scenarios (BALANCED: 75 train + 10 val each)...")
    print("-" * 80)

    for acc_type, config in accident_types.items():
        print(f"\n{acc_type}:")
        print(f"  Source: {config['path']}")
        print(f"  Train IDs: {min(config['train_ids']) if config['train_ids'] else 'N/A'}-{max(config['train_ids']) if config['train_ids'] else 'N/A'}")
        print(f"  Val IDs:   {min(config['val_ids']) if config['val_ids'] else 'N/A'}-{max(config['val_ids']) if config['val_ids'] else 'N/A'}")
        print(f"  Test IDs:  {min(config['test_ids']) if config['test_ids'] else 'N/A'}-{max(config['test_ids']) if config['test_ids'] else 'N/A'}")

        # Copy training scenarios
        for source_id in config['train_ids']:
            source_file = os.path.join(config['path'], f'{source_id}.csv')
            df = pd.read_csv(source_file)
            new_id = source_id + config['id_offset']
            output_file = os.path.join(output_root, f'{new_id}.csv')
            df.to_csv(output_file, index=False)

            all_train.append(new_id)
            scenario_info[new_id] = {
                'accident_type': acc_type,
                'original_id': source_id,
                'split': 'train',
                'timesteps': len(df)
            }

        # Copy validation scenarios
        for source_id in config['val_ids']:
            source_file = os.path.join(config['path'], f'{source_id}.csv')
            df = pd.read_csv(source_file)
            new_id = source_id + config['id_offset']
            output_file = os.path.join(output_root, f'{new_id}.csv')
            df.to_csv(output_file, index=False)

            all_val.append(new_id)
            scenario_info[new_id] = {
                'accident_type': acc_type,
                'original_id': source_id,
                'split': 'val',
                'timesteps': len(df)
            }

        # Copy test scenarios
        for source_id in config['test_ids']:
            source_file = os.path.join(config['path'], f'{source_id}.csv')
            df = pd.read_csv(source_file)
            new_id = source_id + config['id_offset']
            output_file = os.path.join(output_root, f'{new_id}.csv')
            df.to_csv(output_file, index=False)

            all_test.append(new_id)
            scenario_info[new_id] = {
                'accident_type': acc_type,
                'original_id': source_id,
                'split': 'test',
                'timesteps': len(df)
            }

        print(f"  Copied: {len(config['train_ids'])} train, {len(config['val_ids'])} val, {len(config['test_ids'])} test")

    print()
    print(f"Total accident scenarios copied: {len(scenario_info)}")
    print(f"  Train: {len(all_train)} (75 each from 5 accident types)")
    print(f"  Val:   {len(all_val)} (10 each from 4 cross-accident types - NO LOCA!)")
    print(f"  Test:  {len(all_test)} (15 LOCA only)")
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
    print(f"  Val:   0 (NO Normal - validation uses cross-accident types only!)")
    print(f"  Test:  0 (NO Normal - testing LOCA prediction accuracy only!)")
    print()

    # Create metadata
    print("Creating metadata...")
    print("-" * 80)

    metadata = {
        'experiment_name': 'exp4_maximum_accident_diversity',
        'dataset_type': 'maximum_diversity_multi_accident',
        'accident_types': ['LOCA', 'LOCAC', 'RW', 'SGATR', 'FLB', 'Normal'],
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
                'num_scenarios': len(loca_train_ids) + len(loca_test),
                'id_range': '1-100',
                'train': len(loca_train_ids),
                'val': 0,
                'test': len(loca_test)
            },
            'LOCAC': {
                'description': 'Loss of Coolant Accident (Cold Leg) - Matched IDs',
                'num_scenarios': len(loca_train_ids) + len(loca_val_ids),
                'id_range': '201-285',
                'train': len(loca_train_ids),
                'val': len(loca_val_ids),
                'test': 0
            },
            'FLB': {
                'description': 'Feedline Break - Matched IDs',
                'num_scenarios': len(loca_train_ids) + len(loca_val_ids),
                'id_range': '101-185',
                'train': len(loca_train_ids),
                'val': len(loca_val_ids),
                'test': 0
            },
            'RW': {
                'description': 'Rod Withdrawal - Matched IDs',
                'num_scenarios': len(loca_train_ids) + len(loca_val_ids),
                'id_range': '301-385',
                'train': len(loca_train_ids),
                'val': len(loca_val_ids),
                'test': 0
            },
            'SGATR': {
                'description': 'Steam Generator A Tube Rupture - Matched IDs',
                'num_scenarios': len(loca_train_ids) + len(loca_val_ids),
                'id_range': '501-585',
                'train': len(loca_train_ids),
                'val': len(loca_val_ids),
                'test': 0
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
        'scientific_question': 'Does maximum BALANCED accident diversity improve LOCA prediction?',
        'comparison_baseline': 'All previous experiments (exp0-exp3)',
        'rationale': [
            'Test if maximum BALANCED diversity improves LOCA prediction',
            'Training: 75 LOCA + 75 LOCAC + 75 FLB + 75 RW + 75 SGATR + 20 Normal = 395 scenarios',
            'Validation: 10 LOCAC + 10 FLB + 10 RW + 10 SGATR = 40 scenarios (NO LOCA, NO Normal!)',
            'Test: 15 LOCA (same test set as all experiments)',
            'ALL accident types use SAME scenario IDs for controlled comparison',
            'Balanced representation (75 each) enables fair contribution assessment',
            'Validation monitors cross-accident learning without LOCA leakage',
            'Compare with exp2 (LOCAC only) and exp3 (FLB only) to quantify diversity benefit',
            'Tests if combining multiple accident physics improves generalization'
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
        f.write("Experiment 4 Dataset Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write("Scientific Question: Does maximum BALANCED accident diversity improve LOCA?\n\n")

        f.write("Training Strategy:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Train: 75 LOCA + 75 LOCAC + 75 FLB + 75 RW + 75 SGATR + {n_train_normal} Normal = {len(all_train)} scenarios\n")
        f.write(f"  Val:   10 LOCAC + 10 FLB + 10 RW + 10 SGATR = {len(all_val)} scenarios (NO LOCA, NO Normal!)\n")
        f.write(f"  Test:  {len([s for s in scenario_info.values() if s['split'] == 'test'])} LOCA ONLY (NO Normal - pure LOCA prediction accuracy)\n\n")

        f.write("Accident Types (BALANCED representation):\n")
        f.write("-" * 80 + "\n")
        for acc_type, details in metadata['accident_type_details'].items():
            f.write(f"\n{acc_type}:\n")
            f.write(f"  Description: {details['description']}\n")
            f.write(f"  ID Range: {details['id_range']}\n")
            f.write(f"  Train: {details['train']}\n")
            f.write(f"  Val:   {details['val']}\n")
            f.write(f"  Test:  {details['test']}\n")

        f.write("\n\nKey Design Decisions:\n")
        f.write("  - BALANCED representation: 75 scenarios per accident type (train)\n")
        f.write("  - ALL accident types use SAME scenario IDs (1-75 train, 76-85 val)\n")
        f.write("  - LOCA: 75 train, 0 val, 15 test (only accident type in test!)\n")
        f.write("  - LOCAC: 75 train, 10 val (matched IDs to LOCA)\n")
        f.write("  - FLB: 75 train, 10 val (matched IDs to LOCA)\n")
        f.write("  - RW: 75 train, 10 val (matched IDs to LOCA)\n")
        f.write("  - SGATR: 75 train, 10 val (matched IDs to LOCA)\n")
        f.write("  - Normal: 20 windows (ALL for training augmentation)\n")
        f.write("  - CRITICAL: Validation uses cross-accident types, test uses LOCA!\n\n")

        f.write("Expected Insights:\n")
        f.write("  - Does BALANCED diversity improve over single cross-accident (exp2, exp3)?\n")
        f.write("  - Can model learn generalizable accident physics from multiple types?\n")
        f.write("  - Compare with exp0 (LOCA only) to quantify benefit of diversity\n")
        f.write("  - Fair assessment enabled by balanced representation (75 each)\n")
        f.write("  - Identify if combining multiple physics domains improves transfer learning\n\n")

    print(f"Saved summary: {summary_path}")
    print()

    # Print final summary
    print("=" * 80)
    print("EXPERIMENT 4 DATASET CREATION COMPLETE")
    print("=" * 80)
    print()
    print(f"Output directory: {output_root}")
    print(f"Total scenarios: {len(scenario_info)}")
    print()
    print("Split distribution (BALANCED representation):")
    print(f"  Train: {len(all_train)} scenarios")
    for acc_type in ['LOCA', 'LOCAC', 'FLB', 'RW', 'SGATR', 'Normal']:
        n = len([s for s in scenario_info.values() if s['accident_type'] == acc_type and s['split'] == 'train'])
        if n > 0:
            print(f"    - {acc_type}: {n}")
    print(f"  Val:   {len(all_val)} scenarios (NO LOCA, NO Normal!)")
    for acc_type in ['LOCAC', 'FLB', 'RW', 'SGATR']:
        n = len([s for s in scenario_info.values() if s['accident_type'] == acc_type and s['split'] == 'val'])
        if n > 0:
            print(f"    - {acc_type}: {n}")
    print(f"  Test:  {len(all_test)} scenarios (15 LOCA ONLY - NO Normal!)")
    print()
    print("CRITICAL NOTES:")
    print("  - BALANCED representation: 75 scenarios per accident type")
    print("  - ALL accident types use SAME scenario IDs (controlled comparison)")
    print("  - Validation monitors cross-accident learning (NO LOCA leakage!)")
    print("  - Test evaluates LOCA prediction from diverse multi-accident training")
    print()
    print("Ready for training!")
    print()

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description='Create exp4 dataset: Maximum Accident Diversity'
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
        default=str(DATASET_ROOT / 'normal_LOCAC_RW_LOF_SGATR_train_LOCA_test'),
        help='Output directory for exp4 dataset'
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

    create_exp4_dataset(
        raw_data_root=args.raw_data_root,
        output_root=args.output_root,
        loca_metadata_path=args.loca_metadata,
        num_normal_samples=args.num_normal_samples,
        normal_window_size=args.normal_window_size
    )


if __name__ == '__main__':
    main()
