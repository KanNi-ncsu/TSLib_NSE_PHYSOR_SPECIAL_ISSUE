"""
Heatmap Generator for exp0-4 Analysis Suite

Creates comprehensive heatmaps for performance visualization.
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


class HeatmapGenerator:
    """Generate publication-quality heatmaps for model performance."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize heatmap generator.

        Args:
            df: DataFrame with extended metrics
        """
        self.df = df.copy()
        self._setup_style()

    def _setup_style(self):
        """Configure matplotlib style for publication."""
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 9
        plt.rcParams['figure.dpi'] = 300

    def generate_all(self, output_dir: Path = None) -> Dict[str, Path]:
        """
        Generate all heatmaps.

        Returns:
            Dictionary mapping heatmap names to file paths
        """
        if output_dir is None:
            output_dir = config.HEATMAPS_DIR

        output_dir.mkdir(parents=True, exist_ok=True)

        print("Generating heatmaps...")

        heatmaps = {}

        # 1. Model x Experiment heatmaps (one per horizon)
        for horizon in config.HORIZON_ORDER:
            path = self._create_model_experiment_heatmap(horizon, output_dir)
            heatmaps[f'model_experiment_{horizon}'] = path

        # 2. Family x Experiment heatmaps
        for horizon in config.HORIZON_ORDER:
            path = self._create_family_experiment_heatmap(horizon, output_dir)
            heatmaps[f'family_experiment_{horizon}'] = path

        # 3. Improvement heatmap (vs exp0)
        for horizon in config.HORIZON_ORDER:
            path = self._create_improvement_heatmap(horizon, output_dir)
            heatmaps[f'improvement_{horizon}'] = path

        # 4. Rank stability heatmap
        for horizon in config.HORIZON_ORDER:
            path = self._create_rank_heatmap(horizon, output_dir)
            heatmaps[f'rank_{horizon}'] = path

        # 5. Cross-horizon comparison
        path = self._create_cross_horizon_heatmap(output_dir)
        heatmaps['cross_horizon'] = path

        print(f"OK Generated {len(heatmaps)} heatmaps")
        return heatmaps

    def _create_model_experiment_heatmap(
        self,
        horizon: str,
        output_dir: Path
    ) -> Path:
        """Create Model x Experiment MAE heatmap."""
        print(f"  - Creating model-experiment heatmap ({horizon})...")

        horizon_data = self.df[self.df['horizon'] == horizon]

        # Pivot: models as rows, experiments as columns
        pivot = horizon_data.pivot_table(
            values='mae',
            index='model',
            columns='experiment',
            aggfunc='mean'
        )

        # Reorder columns (only available ones)
        available_exps = [exp for exp in config.EXPERIMENT_ORDER if exp in pivot.columns]
        pivot = pivot[available_exps]

        # Sort by exp0 performance
        if 'exp0' in pivot.columns:
            pivot = pivot.sort_values('exp0')

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 10))

        # Create heatmap
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn_r',
            cbar_kws={'label': 'MAE'},
            linewidths=0.5,
            ax=ax,
            vmin=0,
            vmax=pivot.quantile(0.9).max()  # Cap at 90th percentile
        )

        ax.set_title(f'Model Performance Across Experiments ({horizon})',
                     fontweight='bold', pad=15)
        ax.set_xlabel('Experiment', fontweight='bold')
        ax.set_ylabel('Model', fontweight='bold')

        plt.tight_layout()

        output_path = output_dir / f'heatmap_model_experiment_{horizon}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def _create_family_experiment_heatmap(
        self,
        horizon: str,
        output_dir: Path
    ) -> Path:
        """Create Family x Experiment MAE heatmap."""
        print(f"  - Creating family-experiment heatmap ({horizon})...")

        horizon_data = self.df[self.df['horizon'] == horizon]

        # Aggregate by family
        pivot = horizon_data.pivot_table(
            values='mae',
            index='family',
            columns='experiment',
            aggfunc='mean'
        )

        # Reorder (only available)
        available_exps = [exp for exp in config.EXPERIMENT_ORDER if exp in pivot.columns]
        pivot = pivot[available_exps]
        pivot = pivot.reindex(config.MODEL_FAMILIES.keys())

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 5))

        sns.heatmap(
            pivot,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn_r',
            cbar_kws={'label': 'Mean MAE'},
            linewidths=0.5,
            ax=ax,
            vmin=0
        )

        ax.set_title(f'Architecture Family Performance ({horizon})',
                     fontweight='bold', pad=15)
        ax.set_xlabel('Experiment', fontweight='bold')
        ax.set_ylabel('Architecture Family', fontweight='bold')

        plt.tight_layout()

        output_path = output_dir / f'heatmap_family_experiment_{horizon}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def _create_improvement_heatmap(
        self,
        horizon: str,
        output_dir: Path
    ) -> Path:
        """Create improvement vs exp0 heatmap."""
        print(f"  - Creating improvement heatmap ({horizon})...")

        horizon_data = self.df[
            (self.df['horizon'] == horizon) &
            (self.df['improvement_vs_exp0_pct'].notna())
        ]

        if len(horizon_data) == 0:
            return None

        # Pivot
        pivot = horizon_data.pivot_table(
            values='improvement_vs_exp0_pct',
            index='model',
            columns='experiment',
            aggfunc='mean'
        )

        # Remove exp0 column (all zeros)
        pivot = pivot.drop(columns=['exp0'], errors='ignore')

        # Sort by average improvement
        pivot['avg_improvement'] = pivot.mean(axis=1)
        pivot = pivot.sort_values('avg_improvement', ascending=False)
        pivot = pivot.drop(columns=['avg_improvement'])

        # Create figure
        fig, ax = plt.subplots(figsize=(7, 10))

        sns.heatmap(
            pivot,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn',
            center=0,
            cbar_kws={'label': 'Improvement vs exp0 (%)'},
            linewidths=0.5,
            ax=ax
        )

        ax.set_title(f'Performance Improvement vs exp0 ({horizon})',
                     fontweight='bold', pad=15)
        ax.set_xlabel('Experiment', fontweight='bold')
        ax.set_ylabel('Model', fontweight='bold')

        plt.tight_layout()

        output_path = output_dir / f'heatmap_improvement_{horizon}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def _create_rank_heatmap(
        self,
        horizon: str,
        output_dir: Path
    ) -> Path:
        """Create rank heatmap showing model rankings across experiments."""
        print(f"  - Creating rank heatmap ({horizon})...")

        horizon_data = self.df[self.df['horizon'] == horizon]

        # Pivot
        pivot = horizon_data.pivot_table(
            values='rank_mae',
            index='model',
            columns='experiment',
            aggfunc='mean'
        )

        # Reorder (only available)
        available_exps = [exp for exp in config.EXPERIMENT_ORDER if exp in pivot.columns]
        pivot = pivot[available_exps]

        # Sort by exp0 rank
        if 'exp0' in pivot.columns:
            pivot = pivot.sort_values('exp0')

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 10))

        sns.heatmap(
            pivot,
            annot=True,
            fmt='.0f',
            cmap='RdYlGn',
            cbar_kws={'label': 'Rank (1=best)'},
            linewidths=0.5,
            ax=ax,
            vmin=1
        )

        ax.set_title(f'Model Rankings Across Experiments ({horizon})',
                     fontweight='bold', pad=15)
        ax.set_xlabel('Experiment', fontweight='bold')
        ax.set_ylabel('Model', fontweight='bold')

        plt.tight_layout()

        output_path = output_dir / f'heatmap_rank_{horizon}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def _create_cross_horizon_heatmap(self, output_dir: Path) -> Path:
        """Create Model x Horizon heatmap for exp1."""
        print(f"  - Creating cross-horizon heatmap...")

        # Focus on exp1 (best overall experiment)
        exp1_data = self.df[self.df['experiment'] == 'exp1']

        # Pivot
        pivot = exp1_data.pivot_table(
            values='mae',
            index='model',
            columns='horizon',
            aggfunc='mean'
        )

        # Reorder
        pivot = pivot[config.HORIZON_ORDER]

        # Sort by 300s performance
        pivot = pivot.sort_values('300s')

        # Create figure
        fig, ax = plt.subplots(figsize=(6, 10))

        sns.heatmap(
            pivot,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn_r',
            cbar_kws={'label': 'MAE'},
            linewidths=0.5,
            ax=ax,
            vmin=0
        )

        ax.set_title('Model Performance Across Horizons (exp1)',
                     fontweight='bold', pad=15)
        ax.set_xlabel('Horizon', fontweight='bold')
        ax.set_ylabel('Model', fontweight='bold')

        plt.tight_layout()

        output_path = output_dir / 'heatmap_cross_horizon.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path


if __name__ == '__main__':
    print("="*80)
    print("Heatmap Generator Test")
    print("="*80)

    from data_loader import DataLoader
    from metrics_calculator import MetricsCalculator

    # Load data
    loader = DataLoader()
    df = loader.load()
    calculator = MetricsCalculator(df)
    extended_df = calculator.calculate_all()

    # Generate heatmaps
    generator = HeatmapGenerator(extended_df)
    heatmaps = generator.generate_all()

    print("\n" + "="*80)
    print("Generated Heatmaps")
    print("="*80)
    for name, path in heatmaps.items():
        if path:
            print(f"OK {name}: {path.name}")

    print("\nOK Heatmap generator test successful!")
