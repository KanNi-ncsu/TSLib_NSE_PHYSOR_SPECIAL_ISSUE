#!/usr/bin/env python3
"""
Plot LOCA Parameter Evolution - Dual Panel Figure

Parameters plotted:
- TPCT: Peak Cladding Temperature (degrees C) - primary y-axis (left)
- WCFT: Accumulator Flow (t/hr) - secondary y-axis (right)
- WHPI: High Pressure Injection Flow (t/hr) - secondary y-axis (right)
- P:    RCS Pressure (bar) - tertiary y-axis (far right)

Two scenarios (break size in % of 100 cm2): 42 and 84 by default.

Usage:
    python NuclearResearch/scripts/plot_loca_parameters_evolution.py
"""

import os
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Publication-quality color scheme
COLORS = {
    'TPCT': '#D62728',  # Red - critical temperature
    'WCFT': '#2CA02C',  # Green - accumulator injection (passive)
    'WHPI': '#1F77B4',  # Blue - high pressure injection (active)
    'P': '#9467BD',     # Purple - RCS pressure
}


def load_scenario_data(data_root, scenario_id):
    """Load scenario data from CSV file."""
    csv_path = os.path.join(data_root, f'{scenario_id}.csv')
    df = pd.read_csv(csv_path)
    return df


def plot_loca_panel(ax, df, scenario_id, break_size_pct, panel_label):
    """
    Plot a single panel showing LOCA parameter evolution.

    Args:
        ax: Matplotlib axis for the panel
        df: DataFrame with scenario data
        scenario_id: Scenario ID number
        break_size_pct: Break size as percentage of 100 cm2
        panel_label: Label for the panel (e.g., 'a', 'b')
    """
    time = df['TIME'].values
    tpct = df['TPCT'].values  # Peak cladding temperature (C)
    wcft = df['WCFT'].values  # Accumulator flow (t/hr)
    whpi = df['WHPI'].values  # High pressure injection (t/hr)
    pres = df['P'].values     # RCS pressure (bar)

    # Primary y-axis: TPCT (in Celsius)
    line1 = ax.plot(time, tpct, color=COLORS['TPCT'], linewidth=2.5,
                    label='TPCT (Peak Cladding Temp)', zorder=5)
    ax.set_xlabel('Accident Time (s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Peak Cladding Temperature (°C)', fontsize=11,
                  fontweight='bold', color=COLORS['TPCT'])
    ax.tick_params(axis='y', labelcolor=COLORS['TPCT'], labelsize=10)
    ax.tick_params(axis='x', labelsize=10)
    ax.set_ylim([min(tpct) * 0.95, max(tpct) * 1.05])

    # Secondary y-axis: ECCS flow rates (WCFT, WHPI)
    ax2 = ax.twinx()
    line2 = ax2.plot(time, wcft, color=COLORS['WCFT'], linewidth=2.0,
                     label='WCFT (Accumulator)', zorder=4)
    line3 = ax2.plot(time, whpi, color=COLORS['WHPI'], linewidth=2.0,
                     linestyle='--', label='WHPI (HPI)', zorder=3)
    ax2.set_ylabel('ECCS Flow Rate (t/hr)', fontsize=11, fontweight='bold')
    ax2.tick_params(axis='y', labelsize=10)
    max_eccs = max(max(wcft), max(whpi))
    ax2.set_ylim([0, max_eccs * 1.15])

    # Tertiary y-axis: RCS pressure (P)
    ax3 = ax.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    line4 = ax3.plot(time, pres, color=COLORS['P'], linewidth=2.0,
                     label='P (RCS Pressure)', zorder=2)
    ax3.set_ylabel('RCS Pressure (bar)', fontsize=11,
                   fontweight='bold', color=COLORS['P'])
    ax3.tick_params(axis='y', labelcolor=COLORS['P'], labelsize=10)
    p_min, p_max = min(pres), max(pres)
    p_range = p_max - p_min
    ax3.set_ylim([p_min - 0.1 * p_range, p_max + 0.1 * p_range])

    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.set_title(f'({panel_label}) {break_size_pct}% Break Size',
                 fontsize=13, fontweight='bold', pad=10)

    # Combined legend
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper right', fontsize=9,
              framealpha=0.9, edgecolor='gray')

    for side in ['top', 'right', 'bottom', 'left']:
        ax.spines[side].set_visible(True)


def main():
    parser = argparse.ArgumentParser(
        description='Plot LOCA parameter evolution for two scenarios'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default=str(PROJECT_ROOT / 'dataset' / 'LOCA_only_train_LOCA_test'),
        help='Directory containing the LOCA scenario CSVs'
    )
    parser.add_argument('--mid_scenario', type=int, default=42,
                        help='Mid-severity break scenario ID')
    parser.add_argument('--large_scenario', type=int, default=84,
                        help='Large break scenario ID')
    parser.add_argument(
        '--output_dir',
        type=str,
        default=str(PROJECT_ROOT / 'NuclearResearch' / 'results_summary' / 'analysis_output' / 'figures' / 'data'),
        help='Output directory for figure'
    )
    parser.add_argument('--output_filename', type=str,
                        default='loca_parameters_evolution.png',
                        help='Output filename')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df_mid = load_scenario_data(args.data_root, args.mid_scenario)
    df_large = load_scenario_data(args.data_root, args.large_scenario)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))
    # LOCA scenario IDs equal the break size in % of 100 cm2
    plot_loca_panel(ax1, df_mid, args.mid_scenario, args.mid_scenario, 'a')
    plot_loca_panel(ax2, df_large, args.large_scenario, args.large_scenario, 'b')
    plt.tight_layout()

    output_path = os.path.join(args.output_dir, args.output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {output_path}")

    for pct, df in [(args.mid_scenario, df_mid), (args.large_scenario, df_large)]:
        print(f"{pct}% break: peak TPCT {df['TPCT'].max():.1f} C, "
              f"max WCFT {df['WCFT'].max():.1f} t/hr, max WHPI {df['WHPI'].max():.1f} t/hr, "
              f"min P {df['P'].min():.1f} bar")


if __name__ == '__main__':
    main()
