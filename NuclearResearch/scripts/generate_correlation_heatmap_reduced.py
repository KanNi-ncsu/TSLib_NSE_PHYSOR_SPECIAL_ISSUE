#!/usr/bin/env python3
"""
Generate Reduced-Feature Correlation Heatmap for Nuclear Reactor Dataset

This script creates Pearson correlation heatmaps with ONLY the most important
features for publication-quality visualization.

Feature reduction: 74 → 34 features
- Removed symmetric pairs (keep Loop A, remove Loop B)
- Removed late/consequence indicators (radiation, dose, integrated values)
- Removed minor control parameters
- Kept only safety-critical and physics-important features

Selected 34 features grouped by category:
- Safety Critical (6): TPCT, LVCR, P, TAVG, VOL, LVPZ
- Thermal-Hydraulic (5): THA, TCA, WRCA, PSGA, LSGA
- Power/Energy (3): QMWT, PWNT, QMGA
- Safety Systems (5): WHPI, WECS, WCFT, WLPI, QRHR
- Accident Indicators (6): WLR, WUP, SCMA, QFCL, PRB, TRB
- Steam Generator (3): WFWA, WSTA, TKLV
- Reactivity (5): RHBR, RHMT, RHFL, RHRD, RH
- Miscellaneous (1): PPM
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXP4_DIR = PROJECT_ROOT / 'dataset' / 'normal_LOCAC_RW_LOF_SGATR_train_LOCA_test'


# Define the selected 34 features in logical order
SELECTED_FEATURES = [
    # Safety Critical (6)
    'TPCT',  # Peak Cladding Temperature (TARGET)
    'LVCR',  # Core water level
    'P',     # RCS Pressure
    'TAVG',  # RCS average temperature
    'VOL',   # RCS liquid volume
    'LVPZ',  # Pressurizer level

    # Thermal-Hydraulic (5)
    'THA',   # Hot leg temperature A
    'TCA',   # Cold leg temperature A
    'WRCA',  # Reactor coolant flow A
    'PSGA',  # Steam generator pressure A
    'LSGA',  # SG level A (wide range)

    # Power/Energy (3)
    'QMWT',  # Total thermal power
    'PWNT',  # Nuclear flux
    'QMGA',  # SG heat removal A

    # Safety Systems (5)
    'WHPI',  # High pressure injection
    'WECS',  # Total ECCS flow
    'WCFT',  # Accumulator flow
    'WLPI',  # Low pressure injection
    'QRHR',  # RHR heat removal

    # Accident Indicators (6)
    'WLR',   # RCS leak flow
    'WUP',   # PORV and safety valve flow
    'SCMA',  # Subcooling margin A
    'QFCL',  # Fan cooler heat removal
    'PRB',   # Reactor building pressure
    'TRB',   # Reactor building temperature

    # Steam Generator (3)
    'WFWA',  # SG feedwater flow A
    'WSTA',  # SG steam flow A
    'TKLV',  # RWST water volume

    # Reactivity (5)
    'RHBR',  # Reactivity from boron
    'RHMT',  # Reactivity from moderator temp
    'RHFL',  # Reactivity from fuel (Doppler)
    'RHRD',  # Reactivity from rods
    'RH',    # Total reactivity

    # Miscellaneous (1)
    'PPM',   # Boron concentration
]


def load_excluded_features(config_path):
    """Load list of features to exclude from correlation analysis."""
    excluded = []
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line and not line.startswith('#'):
                excluded.append(line)
    return excluded


def load_scenarios_reduced(data_root, scenario_ids, excluded_features, selected_features):
    """
    Load and concatenate multiple scenarios, keeping only selected features.

    Args:
        data_root: Root directory containing scenario CSV files
        scenario_ids: List of scenario IDs to load
        excluded_features: List of feature names to exclude first
        selected_features: List of specific features to keep for correlation

    Returns:
        DataFrame with only selected features from all scenarios
    """
    all_data = []

    for scenario_id in scenario_ids:
        csv_path = os.path.join(data_root, f'{scenario_id}.csv')
        if not os.path.exists(csv_path):
            print(f"WARNING: Scenario {scenario_id} not found at {csv_path}")
            continue

        df = pd.read_csv(csv_path)

        # Remove TIME column (not a feature)
        if 'TIME' in df.columns:
            df = df.drop(columns=['TIME'])

        # Remove excluded features
        cols_to_drop = [col for col in excluded_features if col in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        # Keep only selected features
        available_selected = [col for col in selected_features if col in df.columns]
        df = df[available_selected]

        all_data.append(df)

    # Concatenate all scenarios
    combined_df = pd.concat(all_data, axis=0, ignore_index=True)

    print(f"Loaded {len(scenario_ids)} scenarios")
    print(f"Total timesteps: {len(combined_df)}")
    print(f"Selected features: {len(combined_df.columns)}")
    print(f"Features: {list(combined_df.columns)}")

    return combined_df


def generate_correlation_heatmap_reduced(data, output_path, title, figsize=(16, 14)):
    """
    Generate and save a reduced correlation heatmap with enhanced readability.

    Args:
        data: DataFrame with features to correlate
        output_path: Path to save the figure
        title: Title for the plot
        figsize: Figure size (width, height)
    """
    # Compute Pearson correlation matrix
    corr_matrix = data.corr(method='pearson')

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Generate heatmap with annotations for better readability
    sns.heatmap(
        corr_matrix,
        cmap='RdBu_r',  # Red-Blue diverging, reversed
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={
            'shrink': 0.75,
            'label': 'Pearson Correlation Coefficient',
            'pad': 0.02
        },
        xticklabels=True,
        yticklabels=True,
        annot=False,  # Don't annotate (34x34 is still readable without values)
        ax=ax
    )

    # Customize plot
    plt.title(title, fontsize=18, pad=20, fontweight='bold')
    plt.xlabel('Features', fontsize=14, fontweight='bold')
    plt.ylabel('Features', fontsize=14, fontweight='bold')

    # Increase label font sizes for better readability
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)

    # Tight layout to prevent label cutoff
    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved reduced correlation heatmap: {output_path}")

    # Close figure to free memory
    plt.close()

    # Print correlation statistics
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    correlations = upper_triangle.values.flatten()
    correlations = correlations[~np.isnan(correlations)]

    print(f"\nCorrelation Statistics (34 features):")
    print(f"  Mean: {np.mean(correlations):.4f}")
    print(f"  Std:  {np.std(correlations):.4f}")
    print(f"  Min:  {np.min(correlations):.4f}")
    print(f"  Max:  {np.max(correlations):.4f}")

    # Find top correlations with TPCT (the target)
    if 'TPCT' in corr_matrix.columns:
        print(f"\nTop 10 Correlations with TPCT (Target):")
        tpct_corr = corr_matrix['TPCT'].drop('TPCT').abs().sort_values(ascending=False)
        for i, (feat, corr_val) in enumerate(tpct_corr.head(10).items()):
            actual_corr = corr_matrix.loc[feat, 'TPCT']
            print(f"  {i+1}. {feat}: {actual_corr:.4f}")

    # Find top 10 strongest absolute correlations overall
    print(f"\nTop 10 Strongest Absolute Correlations (Overall):")
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if not np.isnan(corr_val):
                corr_pairs.append((
                    abs(corr_val),
                    corr_val,
                    corr_matrix.columns[i],
                    corr_matrix.columns[j]
                ))

    corr_pairs.sort(reverse=True)

    for i, (abs_corr, corr, feat1, feat2) in enumerate(corr_pairs[:10]):
        print(f"  {i+1}. {feat1} <-> {feat2}: {corr:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate reduced correlation heatmap (34 features) for nuclear reactor dataset'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['multi_accident', 'loca_only', 'normal', 'all'],
        default='all',
        help='Dataset mode: multi_accident (exp4 train), loca_only, normal, or all'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default=str(EXP4_DIR),
        help='Root directory of exp4 dataset'
    )
    parser.add_argument(
        '--config_path',
        type=str,
        default=str(PROJECT_ROOT / 'NuclearResearch' / 'config' / 'excluded_features.txt'),
        help='Path to excluded features configuration'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=str(PROJECT_ROOT / 'NuclearResearch' / 'results_summary' / 'analysis_output' / 'figures' / 'data'),
        help='Output directory for figures'
    )
    parser.add_argument(
        '--metadata_path',
        type=str,
        default=str(EXP4_DIR / 'split_metadata_lhs.json'),
        help='Path to exp4 metadata JSON'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load excluded features
    print("=" * 80)
    print("REDUCED CORRELATION HEATMAP GENERATION (34 Selected Features)")
    print("=" * 80)
    print()
    excluded_features = load_excluded_features(args.config_path)
    print(f"Excluded features (from config): {len(excluded_features)}")
    print()

    # Load metadata to get scenario IDs
    with open(args.metadata_path, 'r') as f:
        metadata = json.load(f)

    train_scenarios = metadata['train_scenarios']

    # Extract LOCA-only training scenarios (IDs 1-100 in train set)
    loca_train_scenarios = [s for s in train_scenarios if s <= 100]

    # Extract normal scenarios (IDs 601-620)
    normal_scenarios = [s for s in train_scenarios if 601 <= s <= 620]

    print(f"Selected {len(SELECTED_FEATURES)} most important features:")
    print("=" * 80)
    for feat in SELECTED_FEATURES:
        print(f"  - {feat}")
    print()

    print(f"Total training scenarios: {len(train_scenarios)}")
    print(f"LOCA training scenarios: {len(loca_train_scenarios)}")
    print(f"Normal scenarios: {len(normal_scenarios)}")
    print()

    # Generate heatmaps based on mode
    modes_to_run = []
    if args.mode == 'all':
        modes_to_run = ['multi_accident', 'loca_only', 'normal']
    else:
        modes_to_run = [args.mode]

    for mode in modes_to_run:
        print("=" * 80)
        print(f"Generating {mode.upper()} correlation heatmap (34 features)")
        print("=" * 80)

        if mode == 'multi_accident':
            # Use ALL 395 training scenarios from exp4
            scenario_ids = train_scenarios
            data_root = args.data_root
            title = 'Pearson Correlation Heatmap (34 Selected Features, Multi-Accident Training)'
            output_filename = 'parameter_correlation_heatmap_34features_multi_accident.png'

        elif mode == 'loca_only':
            # Use 75 LOCA training scenarios
            scenario_ids = loca_train_scenarios
            data_root = args.data_root
            title = 'Pearson Correlation Heatmap (34 Selected Features, LOCA-Only Training)'
            output_filename = 'parameter_correlation_heatmap_34features_loca_only.png'

        elif mode == 'normal':
            # Use 20 normal operation scenarios
            scenario_ids = normal_scenarios
            data_root = args.data_root
            title = 'Pearson Correlation Heatmap (34 Selected Features, Normal Operation)'
            output_filename = 'parameter_correlation_heatmap_34features_normal.png'

        # Load scenarios with selected features only
        data = load_scenarios_reduced(
            data_root,
            scenario_ids,
            excluded_features,
            SELECTED_FEATURES
        )

        # Generate heatmap
        output_path = os.path.join(args.output_dir, output_filename)
        generate_correlation_heatmap_reduced(data, output_path, title)

        print()

    print("=" * 80)
    print("REDUCED CORRELATION HEATMAP GENERATION COMPLETE")
    print("=" * 80)
    print(f"Output directory: {args.output_dir}")
    print(f"Generated 34-feature correlation heatmaps for publication quality")
    print()


if __name__ == '__main__':
    main()
