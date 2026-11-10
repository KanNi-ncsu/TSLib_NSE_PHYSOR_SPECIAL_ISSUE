"""
Comparison Plots for exp0-4 Analysis Suite

Creates bar charts, scatter plots, and comparison visualizations.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import sys
sys.path.append(str(Path(__file__).parent.parent))
import config


class ComparisonPlots:
    """Generate comparison plots for model performance analysis."""

    def __init__(self, df: pd.DataFrame):
        """Initialize comparison plots generator."""
        self.df = df.copy()
        self._setup_style()

    def _setup_style(self):
        """Configure matplotlib style."""
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['figure.dpi'] = 300
        sns.set_palette("husl")

    def generate_all(self, output_dir: Path = None) -> Dict[str, Path]:
        """Generate all comparison plots."""
        if output_dir is None:
            output_dir = config.COMPARISONS_DIR

        output_dir.mkdir(parents=True, exist_ok=True)

        print("Generating comparison plots...")

        plots = {}

        # 1. Top-N model comparison
        for horizon in config.HORIZON_ORDER:
            path = self._create_top_models_comparison(horizon, output_dir, n=10)
            plots[f'top10_{horizon}'] = path

        # 2. Family performance comparison
        for horizon in config.HORIZON_ORDER:
            path = self._create_family_comparison(horizon, output_dir)
            plots[f'family_{horizon}'] = path

        # 3. Improvement distribution
        for target_exp in ['exp1', 'exp2', 'exp3', 'exp4']:
            path = self._create_improvement_distribution(target_exp, output_dir)
            if path:
                plots[f'improvement_dist_{target_exp}'] = path

        print(f"OK Generated {len(plots)} comparison plots")
        return plots

    def _create_top_models_comparison(
        self,
        horizon: str,
        output_dir: Path,
        n: int = 10
    ) -> Path:
        """Create bar chart comparing top-N models across experiments."""
        print(f"  - Creating top-{n} models comparison ({horizon})...")

        horizon_data = self.df[self.df['horizon'] == horizon]

        # Get top-N models by exp0 performance
        exp0_data = horizon_data[horizon_data['experiment'] == 'exp0']
        top_models = exp0_data.nsmallest(n, 'mae')['model'].tolist()

        # Filter data
        plot_data = horizon_data[horizon_data['model'].isin(top_models)]

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Prepare data for grouped bar chart
        pivot = plot_data.pivot_table(
            values='mae',
            index='model',
            columns='experiment',
            aggfunc='mean'
        )
        # Only use available experiments
        available_exps = [exp for exp in config.EXPERIMENT_ORDER if exp in pivot.columns]
        pivot = pivot[available_exps]

        # Plot
        pivot.plot(kind='bar', ax=ax, width=0.8)

        ax.set_title(f'Top-{n} Models Performance Across Experiments ({horizon})',
                     fontweight='bold', pad=15)
        ax.set_xlabel('Model', fontweight='bold')
        ax.set_ylabel('MAE', fontweight='bold')
        ax.legend(title='Experiment', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        output_path = output_dir / f'bar_top{n}_{horizon}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def _create_family_comparison(self, horizon: str, output_dir: Path) -> Path:
        """Create family performance comparison with error bars."""
        print(f"  - Creating family comparison ({horizon})...")

        horizon_data = self.df[self.df['horizon'] == horizon]

        # Aggregate by family
        family_stats = horizon_data.groupby(['family', 'experiment'])['mae'].agg([
            'mean', 'std', 'min', 'max'
        ]).reset_index()

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot grouped bars with error bars
        x = np.arange(len(config.MODEL_FAMILIES))
        width = 0.15
        multiplier = 0

        for exp in config.EXPERIMENT_ORDER:
            exp_data = family_stats[family_stats['experiment'] == exp]
            exp_data = exp_data.set_index('family').reindex(config.MODEL_FAMILIES.keys())

            # Fill NaN std with 0 (families with single model)
            exp_data['std'] = exp_data['std'].fillna(0)

            offset = width * multiplier
            bars = ax.bar(
                x + offset,
                exp_data['mean'],
                width,
                yerr=exp_data['std'],
                label=exp,
                capsize=3
            )
            multiplier += 1

        ax.set_title(f'Architecture Family Performance Comparison ({horizon})',
                     fontweight='bold', pad=15)
        ax.set_xlabel('Architecture Family', fontweight='bold')
        ax.set_ylabel('MAE (mean ± std)', fontweight='bold')
        ax.set_xticks(x + width * (len(config.EXPERIMENT_ORDER) - 1) / 2)
        ax.set_xticklabels(config.MODEL_FAMILIES.keys())
        ax.legend(title='Experiment')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        output_path = output_dir / f'bar_family_{horizon}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def _create_improvement_distribution(
        self,
        target_exp: str,
        output_dir: Path
    ) -> Optional[Path]:
        """Create distribution of improvements vs exp0."""
        print(f"  - Creating improvement distribution ({target_exp})...")

        plot_data = self.df[
            (self.df['experiment'] == target_exp) &
            (self.df['improvement_vs_exp0_pct'].notna())
        ]

        if len(plot_data) == 0:
            return None

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for idx, horizon in enumerate(config.HORIZON_ORDER):
            ax = axes[idx]
            horizon_data = plot_data[plot_data['horizon'] == horizon]

            if len(horizon_data) == 0:
                continue

            # Histogram
            ax.hist(
                horizon_data['improvement_vs_exp0_pct'],
                bins=20,
                alpha=0.7,
                edgecolor='black'
            )

            # Add vertical line at 0
            ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No change')

            # Add mean line
            mean_improvement = horizon_data['improvement_vs_exp0_pct'].mean()
            ax.axvline(
                mean_improvement,
                color='green',
                linestyle='-',
                linewidth=2,
                label=f'Mean: {mean_improvement:.1f}%'
            )

            ax.set_title(f'{horizon}', fontweight='bold')
            ax.set_xlabel('Improvement vs exp0 (%)', fontweight='bold')
            ax.set_ylabel('Count', fontweight='bold')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)

        fig.suptitle(f'Distribution of Performance Improvement: {target_exp} vs exp0',
                     fontsize=14, fontweight='bold')

        plt.tight_layout()

        output_path = output_dir / f'hist_improvement_{target_exp}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path


if __name__ == '__main__':
    print("="*80)
    print("Comparison Plots Test")
    print("="*80)

    from data_loader import DataLoader
    from metrics_calculator import MetricsCalculator

    # Load data
    loader = DataLoader()
    df = loader.load()
    calculator = MetricsCalculator(df)
    extended_df = calculator.calculate_all()

    # Generate plots
    plotter = ComparisonPlots(extended_df)
    plots = plotter.generate_all()

    print("\n" + "="*80)
    print("Generated Comparison Plots")
    print("="*80)
    for name, path in plots.items():
        if path:
            print(f"OK {name}: {path.name}")

    print("\nOK Comparison plots test successful!")
