"""
Uncertainty Visualization Module

Creates plots for conformal prediction intervals, error distributions,
and coverage validation.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional
import sys
sys.path.append(str(Path(__file__).parent.parent))
import config


class UncertaintyPlots:
    """Generate uncertainty-related visualizations."""

    def __init__(self, uncertainty_dir: Path):
        """
        Initialize uncertainty plotter.

        Args:
            uncertainty_dir: Directory containing uncertainty CSV files
        """
        self.uncertainty_dir = Path(uncertainty_dir)
        self._setup_style()

        # Load data
        self.conformal_df = pd.read_csv(uncertainty_dir / 'conformal_intervals.csv')
        self.dist_df = pd.read_csv(uncertainty_dir / 'error_distributions.csv')
        self.coverage_df = pd.read_csv(uncertainty_dir / 'coverage_validation.csv')

    def _setup_style(self):
        """Configure matplotlib style."""
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['figure.dpi'] = 300

    def generate_all(self, output_dir: Path = None) -> Dict[str, Path]:
        """
        Generate all uncertainty visualizations.

        Args:
            output_dir: Output directory (default: config.FIGURES_DIR / 'uncertainty')

        Returns:
            Dictionary mapping plot names to file paths
        """
        if output_dir is None:
            output_dir = config.FIGURES_DIR / 'uncertainty'

        output_dir.mkdir(parents=True, exist_ok=True)

        print("Generating uncertainty visualizations...")

        plots = {}

        # 1. Conformal interval width by horizon
        for conf_level in [90, 95]:
            path = self._plot_interval_width_by_horizon(output_dir, conf_level)
            plots[f'interval_width_{conf_level}'] = path

        # 2. Coverage validation
        path = self._plot_coverage_validation(output_dir)
        plots['coverage_validation'] = path

        # 3. Error distribution violin plots
        for horizon in config.HORIZON_ORDER:
            path = self._plot_error_distribution(horizon, output_dir)
            if path:
                plots[f'error_dist_{horizon}'] = path

        # 4. Conformal interval comparison by family
        for horizon in config.HORIZON_ORDER:
            path = self._plot_interval_by_family(horizon, output_dir)
            if path:
                plots[f'interval_family_{horizon}'] = path

        print(f"OK Generated {len(plots)} uncertainty visualizations")
        return plots

    def _plot_interval_width_by_horizon(
        self,
        output_dir: Path,
        conf_level: int = 90
    ) -> Path:
        """Plot conformal interval width as function of horizon."""
        print(f"  - Creating interval width plot ({conf_level}% confidence)...")

        col_name = f'conf_{conf_level}_radius'

        # Create figure
        fig, axes = plt.subplots(1, len(config.EXPERIMENT_ORDER), figsize=(15, 5))

        for idx, exp in enumerate(config.EXPERIMENT_ORDER):
            ax = axes[idx]
            exp_data = self.conformal_df[self.conformal_df['experiment'] == exp]

            if len(exp_data) == 0:
                continue

            # Plot for each horizon
            horizons = []
            radii = []
            for h_name in config.HORIZON_ORDER:
                h_data = exp_data[exp_data['horizon'] == h_name]
                if len(h_data) > 0:
                    horizons.append(h_name)
                    radii.append(h_data[col_name].mean())

            if horizons:
                ax.plot(horizons, radii, marker='o', markersize=8, linewidth=2)
                ax.set_title(f'{exp}', fontweight='bold')
                ax.set_xlabel('Prediction Horizon', fontweight='bold')
                ax.set_ylabel(f'{conf_level}% Conf. Interval Radius', fontweight='bold')
                ax.grid(alpha=0.3)

        fig.suptitle(f'{conf_level}% Conformal Interval Width by Horizon',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_path = output_dir / f'line_interval_width_{conf_level}pct.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def _plot_coverage_validation(self, output_dir: Path) -> Path:
        """Plot coverage validation comparing predicted vs empirical."""
        print("  - Creating coverage validation plot...")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 90% coverage
        ax = axes[0]
        for exp in config.EXPERIMENT_ORDER:
            exp_data = self.coverage_df[self.coverage_df['experiment'] == exp]
            if len(exp_data) == 0:
                continue

            ax.scatter(
                [0.90] * len(exp_data),
                exp_data['coverage_90_empirical'],
                alpha=0.5,
                label=exp,
                s=50
            )

        ax.axhline(0.90, color='red', linestyle='--', linewidth=2, label='Target')
        ax.set_xlim(0.85, 0.95)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Target Coverage', fontweight='bold')
        ax.set_ylabel('Empirical Coverage', fontweight='bold')
        ax.set_title('90% Confidence Intervals', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        # 95% coverage
        ax = axes[1]
        for exp in config.EXPERIMENT_ORDER:
            exp_data = self.coverage_df[self.coverage_df['experiment'] == exp]
            if len(exp_data) == 0:
                continue

            ax.scatter(
                [0.95] * len(exp_data),
                exp_data['coverage_95_empirical'],
                alpha=0.5,
                label=exp,
                s=50
            )

        ax.axhline(0.95, color='red', linestyle='--', linewidth=2, label='Target')
        ax.set_xlim(0.90, 1.0)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Target Coverage', fontweight='bold')
        ax.set_ylabel('Empirical Coverage', fontweight='bold')
        ax.set_title('95% Confidence Intervals', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        fig.suptitle('Conformal Prediction Coverage Validation',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_path = output_dir / 'scatter_coverage_validation.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def _plot_error_distribution(self, horizon: str, output_dir: Path) -> Optional[Path]:
        """Plot error distribution violin plots by family."""
        print(f"  - Creating error distribution plot ({horizon})...")

        horizon_data = self.dist_df[self.dist_df['horizon'] == horizon]
        if len(horizon_data) == 0:
            return None

        # Add family information
        horizon_data = horizon_data.copy()
        horizon_data['family'] = horizon_data['model'].apply(config.get_family)

        # Create figure
        fig, axes = plt.subplots(1, len(config.EXPERIMENT_ORDER), figsize=(18, 5))

        for idx, exp in enumerate(config.EXPERIMENT_ORDER):
            ax = axes[idx]
            exp_data = horizon_data[horizon_data['experiment'] == exp]

            if len(exp_data) == 0:
                continue

            # Prepare data for violin plot
            plot_data = []
            labels = []
            present_families = []
            short = {
                'Transformers': 'Trans', 'CNNs': 'CNNs',
                'RNNs': 'RNNs', 'MLPs': 'MLPs',
                'Baselines': 'Base', 'Foundation': 'Found',
                'Foundation-FT': 'FT',
            }
            for family in config.MODEL_FAMILIES.keys():
                family_data = exp_data[exp_data['family'] == family]
                if len(family_data) > 0:
                    plot_data.append(family_data['q90_ae'].values)
                    labels.append(short.get(family, family[:4]))
                    present_families.append(family)

            if plot_data:
                parts = ax.violinplot(plot_data, positions=range(len(plot_data)),
                                     showmeans=True, showmedians=True)

                # Color by family (use only families that have data)
                for pc, family in zip(parts['bodies'], present_families):
                    pc.set_facecolor(config.FAMILY_COLORS[family])
                    pc.set_alpha(0.7)

                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=0)
                ax.set_title(f'{exp}', fontweight='bold')
                ax.set_ylabel('90th Percentile Absolute Error', fontweight='bold')
                ax.grid(axis='y', alpha=0.3)

        fig.suptitle(f'Error Distribution by Family ({horizon})',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_path = output_dir / f'violin_error_dist_{horizon}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def _plot_interval_by_family(self, horizon: str, output_dir: Path) -> Optional[Path]:
        """Plot conformal interval comparison by architecture family."""
        print(f"  - Creating interval by family plot ({horizon})...")

        horizon_data = self.conformal_df[self.conformal_df['horizon'] == horizon]
        if len(horizon_data) == 0:
            return None

        # Add family information
        horizon_data = horizon_data.copy()
        horizon_data['family'] = horizon_data['model'].apply(config.get_family)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Group by family and experiment
        x_pos = 0
        x_ticks = []
        x_labels = []

        for family in config.MODEL_FAMILIES.keys():
            family_data = horizon_data[horizon_data['family'] == family]

            if len(family_data) == 0:
                continue

            # Plot for each experiment
            for exp in config.EXPERIMENT_ORDER:
                exp_family_data = family_data[family_data['experiment'] == exp]

                if len(exp_family_data) > 0:
                    mean_radius = exp_family_data['conf_90_radius'].mean()
                    ax.bar(x_pos, mean_radius, color=config.FAMILY_COLORS[family],
                          alpha=0.7, edgecolor='black')

                    x_ticks.append(x_pos)
                    short = {
                        'Transformers': 'Trans', 'CNNs': 'CNNs',
                        'RNNs': 'RNNs', 'MLPs': 'MLPs',
                        'Baselines': 'Base', 'Foundation': 'Found',
                        'Foundation-FT': 'FT',
                    }
                    x_labels.append(f'{short.get(family, family[:4])}\n{exp}')
                    x_pos += 1

            x_pos += 0.5  # Gap between families

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=0, fontsize=7)
        ax.set_title(f'90% Conformal Interval by Family ({horizon})',
                     fontweight='bold', pad=15)
        ax.set_ylabel('Interval Radius', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        output_path = output_dir / f'bar_interval_family_{horizon}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path


if __name__ == '__main__':
    print("="*80)
    print("Uncertainty Plots Test")
    print("="*80)
    print()

    uncertainty_dir = config.OUTPUT_DIR / 'uncertainty'

    if not uncertainty_dir.exists():
        print(f"ERROR: Uncertainty directory not found: {uncertainty_dir}")
        print("Please run uncertainty_batch_processor.py first")
        sys.exit(1)

    # Generate plots
    plotter = UncertaintyPlots(uncertainty_dir)
    plots = plotter.generate_all()

    print("\n" + "="*80)
    print("Generated Uncertainty Plots")
    print("="*80)
    for name, path in plots.items():
        if path:
            print(f"OK {name}: {path.name}")

    print("\nOK Uncertainty plots test successful!")
