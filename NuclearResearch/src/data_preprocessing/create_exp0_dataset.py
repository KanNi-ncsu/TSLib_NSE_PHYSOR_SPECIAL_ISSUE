#!/usr/bin/env python3
"""
Experiment 0 Dataset Creation: Pure LOCA Baseline

This script creates the exp0 dataset for multi-horizon nuclear LOCA forecasting.

Training Strategy:
- Train on: 75 LOCA scenarios
- Validate on: 10 LOCA scenarios
- Test on: 15 LOCA scenarios

Scientific Question: Can models achieve competitive performance with only LOCA data?

Dataset Structure:
- LOCA scenarios: Use Latin Hypercube Sampling to split 100 scenarios
  - Train: 75 scenarios
  - Val: 10 scenarios
  - Test: 15 scenarios

Total files: 100 (75 train, 10 val, 15 test)
"""

import os
import pandas as pd
import json
import argparse
import shutil
from pathlib import Path
import numpy as np
from scipy.stats import qmc

# Default locations relative to the repository root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_ROOT = str(PROJECT_ROOT / 'NuclearResearch' / 'Raw_data' / 'DONOTMODIFY' / 'Operation_csv_data')
DATASET_ROOT = PROJECT_ROOT / 'dataset'


def create_exp0_dataset(
    raw_data_root=RAW_DATA_ROOT,
    output_root=str(DATASET_ROOT / 'LOCA_only_train_LOCA_test'),
    random_seed=42
):
    """
    Create exp0 dataset with 75 LOCA train / 10 val / 15 test

    Args:
        raw_data_root: Root directory of all accident type data
        output_root: Output directory for exp0 dataset
        random_seed: Random seed for reproducible LHS sampling
    """

    # Create output directory
    os.makedirs(output_root, exist_ok=True)

    print("=" * 80)
    print("EXPERIMENT 0 DATASET CREATION: Pure LOCA Baseline")
    print("=" * 80)
    print()

    # LOCA source path
    loca_source_path = os.path.join(raw_data_root, 'LOCA')

    # Get all LOCA scenario IDs (1-100)
    all_loca_ids = list(range(1, 101))
    total_scenarios = len(all_loca_ids)

    print(f"Found {total_scenarios} LOCA scenarios in {loca_source_path}")
    print()

    # Split using Latin Hypercube Sampling for representative distribution
    print("Performing Latin Hypercube Sampling split...")
    print("-" * 80)

    n_train = 75
    n_val = 10
    n_test = 15

    # Use LHS to get representative samples
    np.random.seed(random_seed)
    sampler = qmc.LatinHypercube(d=1, seed=random_seed)
    samples = sampler.random(n=total_scenarios)

    # Convert LHS samples to indices
    indices = (samples * total_scenarios).astype(int).flatten()

    # Ensure no duplicates and correct range
    indices = np.clip(indices, 0, total_scenarios - 1)
    unique_indices = []
    for idx in indices:
        if idx not in unique_indices:
            unique_indices.append(idx)

    # If we have duplicates, fill with remaining indices
    while len(unique_indices) < total_scenarios:
        for i in range(total_scenarios):
            if i not in unique_indices:
                unique_indices.append(i)
                if len(unique_indices) == total_scenarios:
                    break

    # Convert indices to scenario IDs
    shuffled_ids = [all_loca_ids[i] for i in unique_indices[:total_scenarios]]

    # Split into train/val/test
    train_scenarios = sorted(shuffled_ids[:n_train])
    val_scenarios = sorted(shuffled_ids[n_train:n_train + n_val])
    test_scenarios = sorted(shuffled_ids[n_train + n_val:])

    print(f"Split distribution (using LHS with seed={random_seed}):")
    print(f"  Train: {len(train_scenarios)} scenarios")
    print(f"  Val:   {len(val_scenarios)} scenarios")
    print(f"  Test:  {len(test_scenarios)} scenarios")
    print()

    # Copy LOCA scenarios
    print("Copying LOCA scenarios...")
    print("-" * 80)

    scenario_info = {}

    for scenario_id in train_scenarios:
        source_file = os.path.join(loca_source_path, f'{scenario_id}.csv')
        df = pd.read_csv(source_file)
        output_file = os.path.join(output_root, f'{scenario_id}.csv')
        df.to_csv(output_file, index=False)

        scenario_info[scenario_id] = {
            'accident_type': 'LOCA',
            'original_id': scenario_id,
            'split': 'train',
            'timesteps': len(df)
        }

    for scenario_id in val_scenarios:
        source_file = os.path.join(loca_source_path, f'{scenario_id}.csv')
        df = pd.read_csv(source_file)
        output_file = os.path.join(output_root, f'{scenario_id}.csv')
        df.to_csv(output_file, index=False)

        scenario_info[scenario_id] = {
            'accident_type': 'LOCA',
            'original_id': scenario_id,
            'split': 'val',
            'timesteps': len(df)
        }

    for scenario_id in test_scenarios:
        source_file = os.path.join(loca_source_path, f'{scenario_id}.csv')
        df = pd.read_csv(source_file)
        output_file = os.path.join(output_root, f'{scenario_id}.csv')
        df.to_csv(output_file, index=False)

        scenario_info[scenario_id] = {
            'accident_type': 'LOCA',
            'original_id': scenario_id,
            'split': 'test',
            'timesteps': len(df)
        }

    print(f"Copied {len(train_scenarios)} LOCA train scenarios")
    print(f"Copied {len(val_scenarios)} LOCA val scenarios")
    print(f"Copied {len(test_scenarios)} LOCA test scenarios")
    print()

    # Create metadata
    print("Creating metadata...")
    print("-" * 80)

    metadata = {
        'experiment_name': 'LOCA_only_train_LOCA_test',
        'dataset_type': 'pure_LOCA_baseline',
        'accident_types': ['LOCA'],
        'total_scenarios': len(scenario_info),
        'n_train': len(train_scenarios),
        'n_val': len(val_scenarios),
        'n_test': len(test_scenarios),
        'train_scenarios': train_scenarios,
        'val_scenarios': val_scenarios,
        'test_scenarios': test_scenarios,
        'scenario_specs': {
            'target_feature': 'TPCT',
            'target_column_index': 61,
            'num_features': 94
        },
        'raw_data_location': '.',
        'sampling_method': 'Latin_Hypercube_Sampling',
        'random_seed': random_seed,
        'accident_type_details': {
            'LOCA': {
                'description': 'Loss of Coolant Accident (Hot Leg) - Pure Baseline',
                'num_scenarios': 100,
                'id_range': '1-100',
                'train': len(train_scenarios),
                'val': len(val_scenarios),
                'test': len(test_scenarios)
            }
        },
        'scientific_question': 'Can models achieve competitive performance with only LOCA data?',
        'rationale': [
            'Pure LOCA baseline for comparison with cross-accident transfer learning',
            'Training: 75 LOCA scenarios',
            'Validation: 10 LOCA scenarios',
            'Test: 15 LOCA scenarios',
            'All other experiments (exp1-4) will be compared against this baseline'
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
        f.write("Experiment 0 Dataset Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write("Scientific Question: Can models achieve competitive performance with only LOCA data?\n\n")

        f.write("Training Strategy:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Train: {len(train_scenarios)} LOCA scenarios\n")
        f.write(f"  Val:   {len(val_scenarios)} LOCA scenarios\n")
        f.write(f"  Test:  {len(test_scenarios)} LOCA scenarios\n\n")

        f.write("Sampling Method:\n")
        f.write(f"  - Latin Hypercube Sampling (LHS)\n")
        f.write(f"  - Random seed: {random_seed}\n")
        f.write(f"  - Ensures representative distribution across parameter space\n\n")

        f.write("Key Design Decisions:\n")
        f.write("  - Pure LOCA baseline (no cross-accident data)\n")
        f.write("  - LHS sampling for reproducible, representative splits\n")
        f.write("  - 75/10/15 split ratio for train/val/test\n")
        f.write("  - All 100 LOCA scenarios from raw data used\n\n")

        f.write("Expected Role:\n")
        f.write("  - Baseline for comparison with cross-accident transfer learning\n")
        f.write("  - Reference point for exp1 (LOCA+Normal), exp2 (LOCAC+Normal)\n")
        f.write("  - Demonstrates performance achievable with pure in-domain data\n\n")

    print(f"Saved summary: {summary_path}")
    print()

    # Print final summary
    print("=" * 80)
    print("EXPERIMENT 0 DATASET CREATION COMPLETE")
    print("=" * 80)
    print()
    print(f"Output directory: {output_root}")
    print(f"Total scenarios: {len(scenario_info)}")
    print()
    print("Split distribution:")
    print(f"  Train: {len(train_scenarios)} scenarios")
    print(f"  Val:   {len(val_scenarios)} scenarios")
    print(f"  Test:  {len(test_scenarios)} scenarios")
    print()
    print("Ready for training!")
    print()

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description='Create exp0 dataset: Pure LOCA Baseline'
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
        default=str(DATASET_ROOT / 'LOCA_only_train_LOCA_test'),
        help='Output directory for exp0 dataset'
    )
    parser.add_argument(
        '--random_seed',
        type=int,
        default=42,
        help='Random seed for reproducible LHS sampling'
    )

    args = parser.parse_args()

    create_exp0_dataset(
        raw_data_root=args.raw_data_root,
        output_root=args.output_root,
        random_seed=args.random_seed
    )


if __name__ == '__main__':
    main()
