"""
Ranking Visualizer for exp0-4 Analysis Suite

Creates visualizations for model rankings and rank stability analysis.
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


class RankingVisualizer:
    """Generate ranking-related visualizations."""

    def __init__(self, df: pd.DataFrame):
        """Initialize ranking visualizer."""
        self.df = df.copy()
        self._setup_style()

    def _setup_style(self):
        """Configure matplotlib style."""
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['figure.dpi'] = 300

    def generate_all(self, output_dir: Path = None) -> Dict[str, Path]:
        """Generate all ranking visualizations."""
        if output_dir is None:
            output_dir = config.RANKINGS_DIR

        output_dir.mkdir(parents=True, exist_ok=True)

        print("Generating ranking visualizations...")

        plots = {}

        # 1. Rank stability heatmap (Kendall's tau)
        path = self._create_rank_stability_heatmap(output_dir)
        plots['rank_stability'] = path

        # 2. Top-K model overlap
        for horizon in config.HORIZON_ORDER:
            path = self._create_topk_overlap(horizon, output_dir, k=5)
            plots[f'top5_overlap_{horizon}'] = path

        # 3. Win-loss visualization
        for horizon in config.HORIZON_ORDER:
            path = self._create_win_loss_plot(horizon, output_dir)
            if path:
                plots[f'win_loss_{horizon}'] = path

        # 4. Percentile distribution
        for horizon in config.HORIZON_ORDER:
            path = self._create_percentile_plot(horizon, output_dir)
            plots[f'percentile_{horizon}'] = path

        print(f"OK Generated {len(plots)} ranking visualizations")
        return plots

    def _create_rank_stability_heatmap(self, output_dir: Path) -> Path:
        """Create heatmap showing Kendall's tau correlations."""
        print(f"  - Creating rank stability heatmap...")

        # Load rank correlations
        corr_path = config.STATISTICAL_DIR / 'rank_correlations.csv'
        if not corr_path.exists():
            print("    Warning: rank_correlations.csv not found")
            return None

        corr_df = pd.read_csv(corr_path)

        # Create figure with subplots for each horizon
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for idx, horizon in enumerate(config.HORIZON_ORDER):
            ax = axes[idx]
            horizon_data = corr_df[corr_df['horizon'] == horizon]

            # Create correlation matrix
            matrix = pd.DataFrame(
                index=config.EXPERIMENT_ORDER,
                columns=config.EXPERIMENT_ORDER,
                dtype=float
            )

            # Fill diagonal with 1.0
            for exp in config.EXPERIMENT_ORDER:
                matrix.loc[exp, exp] = 1.0

            # Fill from pairwise data
            for _, row in horizon_data.iterrows():
                matrix.loc[row['experiment_a'], row['experiment_b']] = row['kendall_tau']
                matrix.loc[row['experiment_b'], row['experiment_a']] = row['kendall_tau']

            # Plot heatmap
            sns.heatmap(
                matrix.astype(float),
                annot=True,
                fmt='.2f',
                cmap='RdYlGn',
                center=0.5,
                vmin=0,
                vmax=1,
                cbar_kws={'label': "Kendall's tau"},
                linewidths=0.5,
                ax=ax
            )

            ax.set_title(f'{horizon}', fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel('')

        fig.suptitle("Rank Stability Across Experiments (Kendall's tau)",
                     fontsize=14, fontweight='bold')

        plt.tight_layout()

        output_path = output_dir / 'heatmap_rank_stability.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def _create_topk_overlap(self, horizon: str, output_dir: Path, k: int = 5) -> Path:
        """Create bar chart showing top-K model overlap between experiments."""
        print(f"  - Creating top-{k} overlap plot ({horizon})...")

        horizon_data = self.df[self.df['horizon'] == horizon]

        # Get top-K models for each experiment
        topk_models = {}
        for exp in config.EXPERIMENT_ORDER:
            exp_data = horizon_data[horizon_data['experiment'] == exp]
            topk_models[exp] = set(exp_data.nsmallest(k, 'mae')['model'].tolist())

        # Calculate pairwise overlaps
        overlaps = []
        for i, exp1 in enumerate(config.EXPERIMENT_ORDER):
            for exp2 in config.EXPERIMENT_ORDER[i+1:]:
                overlap = len(topk_models[exp1] & topk_models[exp2])
                overlaps.append({
                    'pair': f'{exp1} ∩ {exp2}',
                    'overlap': overlap,
                    'percentage': (overlap / k) * 100
                })

        overlap_df = pd.DataFrame(overlaps)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(
            overlap_df['pair'],
            overlap_df['overlap'],
            color='steelblue',
            edgecolor='black'
        )

        # Add percentage labels
        for bar, pct in zip(bars, overlap_df['percentage']):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f'{pct:.0f}%',
                ha='center',
                va='bottom',
                fontsize=9
            )

        ax.set_title(f'Top-{k} Model Overlap Between Experiments ({horizon})',
                     fontweight='bold', pad=15)
        ax.set_xlabel('Experiment Pair', fontweight='bold')
        ax.set_ylabel(f'Number of Overlapping Models (out of {k})', fontweight='bold')
        ax.set_ylim(0, k + 0.5)
        ax.grid(axis='y', alpha=0.3)

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        output_path = output_dir / f'bar_top{k}_overlap_{horizon}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def _create_win_loss_plot(self, horizon: str, output_dir: Path) -> Optional[Path]:
        """Create win-loss matrix visualization."""
        print(f"  - Creating win-loss plot ({horizon})...")

        # Load win-loss matrix
        wl_path = config.METRICS_DIR / f'win_loss_matrix_{horizon}.csv'
        if not wl_path.exists():
            print(f"    Warning: win_loss_matrix_{horizon}.csv not found")
            return None

        wl_matrix = pd.read_csv(wl_path, index_col=0)

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 7))

        sns.heatmap(
            wl_matrix.astype(int),
            annot=True,
            fmt='d',
            cmap='RdYlGn',
            cbar_kws={'label': 'Number of Wins'},
            linewidths=0.5,
            ax=ax
        )

        ax.set_title(f'Win-Loss Matrix: Row vs Column ({horizon})',
                     fontweight='bold', pad=15)
        ax.set_xlabel('Column Experiment', fontweight='bold')
        ax.set_ylabel('Row Experiment (Wins)', fontweight='bold')

        plt.tight_layout()

        output_path = output_dir / f'heatmap_win_loss_{horizon}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def _create_percentile_plot(self, horizon: str, output_dir: Path) -> Path:
        """Create percentile distribution plot."""
        print(f"  - Creating percentile plot ({horizon})...")

        horizon_data = self.df[self.df['horizon'] == horizon]

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Box plot by family
        family_order = list(config.MODEL_FAMILIES.keys())
        plot_data = []
        labels = []

        for family in family_order:
            for exp in config.EXPERIMENT_ORDER:
                data = horizon_data[
                    (horizon_data['family'] == family) &
                    (horizon_data['experiment'] == exp)
                ]['percentile_rank']

                if len(data) > 0:
                    plot_data.append(data)
                    short_names = {
                        'Transformers': 'Trans',
                        'CNNs': 'CNNs',
                        'RNNs': 'RNNs',
                        'MLPs': 'MLPs',
                        'Baselines': 'Base',
                        'Foundation': 'Found',
                        'Foundation-FT': 'FT',
                    }
                    labels.append(f'{short_names.get(family, family[:4])}\n{exp}')

        bp = ax.boxplot(
            plot_data,
            labels=labels,
            patch_artist=True,
            widths=0.6
        )

        # Color by family
        colors = []
        for family in family_order:
            for _ in config.EXPERIMENT_ORDER:
                colors.append(config.FAMILY_COLORS[family])

        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(f'Percentile Rank Distribution by Family ({horizon})',
                     fontweight='bold', pad=15)
        ax.set_xlabel('Family - Experiment', fontweight='bold')
        ax.set_ylabel('Percentile Rank (0-100)', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 100)

        plt.xticks(rotation=90, fontsize=8)
        plt.tight_layout()

        output_path = output_dir / f'box_percentile_{horizon}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path


if __name__ == '__main__':
    print("="*80)
    print("Ranking Visualizer Test")
    print("="*80)

    from data_loader import DataLoader
    from metrics_calculator import MetricsCalculator

    # Load data
    loader = DataLoader()
    df = loader.load()
    calculator = MetricsCalculator(df)
    extended_df = calculator.calculate_all()

    # Generate visualizations
    visualizer = RankingVisualizer(extended_df)
    plots = visualizer.generate_all()

    print("\n" + "="*80)
    print("Generated Ranking Visualizations")
    print("="*80)
    for name, path in plots.items():
        if path:
            print(f"OK {name}: {path.name}")

    print("\nOK Ranking visualizer test successful!")
