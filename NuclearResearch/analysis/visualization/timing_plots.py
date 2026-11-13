"""
Timing visualization module for publication-ready figures.

Generates training time rankings, inference speed benchmarks,
and efficiency (time vs MAE) scatter plots with Pareto fronts.
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
import sys

sys.path.append(str(Path(__file__).parent.parent))
import config


# Models excluded from the training-time charts
TRAINING_EXCLUDE_MODELS = {'movingaverage', 'naive', 'arima'}


class TimingPlots:
    """Generate timing-related visualizations."""

    def __init__(self, timing_analyzer):
        """
        Args:
            timing_analyzer: TimingAnalyzer instance with loaded data.
        """
        self.analyzer = timing_analyzer
        self._setup_style()

    def _setup_style(self):
        """Configure matplotlib style for publication."""
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['figure.dpi'] = 300

    def generate_all(self, output_dir: Path = None) -> Dict[str, Path]:
        """Generate all timing visualizations."""
        if output_dir is None:
            output_dir = config.FIGURES_DIR / 'timing'
        output_dir.mkdir(parents=True, exist_ok=True)

        plots = {}
        plots['bar_training_by_model'] = self._plot_training_bars(output_dir)
        plots['bar_training_by_family'] = self._plot_family_training(output_dir)

        if len(self.analyzer.inference_df) > 0:
            plots['bar_inference_by_model'] = self._plot_inference_bars(output_dir)
            plots['bar_inference_by_model_specialist'] = self._plot_inference_bars_specialist(output_dir)

        results = self.analyzer.analyze()
        if 'efficiency_table' in results and len(results['efficiency_table']) > 0:
            for hz in ['300s', '600s']:
                key = f'scatter_efficiency_{hz}'
                plots[key] = self._plot_efficiency_scatter(
                    hz, results['efficiency_table'], output_dir)

        return plots

    def _plot_training_bars(self, output_dir: Path) -> Path:
        """Horizontal bar chart: models sorted by avg training time."""
        results = self.analyzer.analyze()
        df = results['training_summary'].copy()

        df = df[~df['model'].str.lower().isin(TRAINING_EXCLUDE_MODELS)]

        # Separate specialist and foundation-ft, sort each by avg time
        specialist = df[df['model_type'] == 'specialist'].sort_values(
            'avg_time_overall', ascending=True)
        foundation_ft = df[df['model_type'] == 'foundation-ft'].sort_values(
            'avg_time_overall', ascending=True)

        # Combine: specialist first, then foundation-ft
        plot_df = pd.concat([specialist, foundation_ft], ignore_index=True)

        fig, ax = plt.subplots(figsize=(12, max(8, len(plot_df) * 0.35)))

        y_pos = np.arange(len(plot_df))
        colors = [config.FAMILY_COLORS.get(f, '#999999') for f in plot_df['family']]

        bars = ax.barh(y_pos, plot_df['avg_time_overall'], color=colors,
                       edgecolor='white', linewidth=0.5)

        # Error bars if std is available
        if 'std_time_overall' in plot_df.columns:
            ax.errorbar(plot_df['avg_time_overall'], y_pos,
                        xerr=plot_df['std_time_overall'].fillna(0),
                        fmt='none', ecolor='gray', elinewidth=0.8, capsize=2)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_df['model'], fontsize=9)
        ax.set_xscale('log')
        ax.set_xlabel('Average Total Time (seconds, log scale)')
        ax.set_title('Training Time per Model (exp4)')

        # Separator line between specialist and foundation-ft
        if len(specialist) > 0 and len(foundation_ft) > 0:
            sep_y = len(specialist) - 0.5
            ax.axhline(y=sep_y, color='black', linewidth=1.5, linestyle='--',
                        alpha=0.5)
            ax.text(0.98, sep_y + 0.3, 'Foundation (fine-tuned)',
                    fontsize=8, fontstyle='italic', alpha=0.7,
                    ha='right', transform=ax.get_yaxis_transform())

        # Legend for families
        handles = []
        for family, color in config.FAMILY_COLORS.items():
            if family in plot_df['family'].values:
                handles.append(plt.Rectangle((0, 0), 1, 1, fc=color,
                                             label=family))
        ax.legend(handles=handles, loc='upper right', fontsize=9)

        ax.invert_yaxis()
        plt.tight_layout()

        path = output_dir / 'bar_training_by_model.png'
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f"OK Saved: {path.name}")
        return path

    def _plot_family_training(self, output_dir: Path) -> Path:
        """Grouped bar chart: family avg training time per horizon."""
        df = self.analyzer.training_df.copy()

        df = df[~df['model'].str.lower().isin(TRAINING_EXCLUDE_MODELS)]

        # Only standard horizons
        df = df[df['horizon'].isin(config.HORIZON_ORDER)]

        # Average per (family, horizon) across models and experiments
        family_hz = df.groupby(['family', 'horizon'])['total_seconds'].mean()
        family_hz = family_hz.reset_index()

        families = list(config.FAMILY_COLORS.keys())
        families = [f for f in families if f in family_hz['family'].values]

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(config.HORIZON_ORDER))
        width = 0.8 / len(families)

        for i, family in enumerate(families):
            fam_data = family_hz[family_hz['family'] == family]
            values = []
            for hz in config.HORIZON_ORDER:
                hz_val = fam_data[fam_data['horizon'] == hz]['total_seconds']
                values.append(hz_val.values[0] if len(hz_val) > 0 else 0)

            offset = (i - len(families) / 2 + 0.5) * width
            ax.bar(x + offset, values, width,
                   label=family, color=config.FAMILY_COLORS[family],
                   edgecolor='white', linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(config.HORIZON_ORDER)
        ax.set_xlabel('Prediction Horizon')
        ax.set_ylabel('Average Total Time (seconds)')
        ax.set_title('Training + Test Time by Architecture Family and Horizon')
        ax.legend(fontsize=8)

        plt.tight_layout()
        path = output_dir / 'bar_training_by_family.png'
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f"OK Saved: {path.name}")
        return path

    def _plot_inference_bars(self, output_dir: Path) -> Path:
        """Single-panel horizontal bar chart: inference time per sample."""
        results = self.analyzer.analyze()
        inf = results.get('inference_summary', pd.DataFrame())
        if len(inf) == 0:
            return output_dir / 'bar_inference_by_model.png'

        # Use 300s horizon as representative
        df = inf[inf['horizon'] == '300s'].copy()
        if len(df) == 0:
            df = inf.drop_duplicates('model', keep='first')
        df = df.sort_values('single_fwd_mean_ms', ascending=True)

        fig, ax = plt.subplots(figsize=(10, max(8, len(df) * 0.35)))

        y_pos = np.arange(len(df))
        colors = [config.FAMILY_COLORS.get(f, '#999999') for f in df['family']]

        ax.barh(y_pos, df['single_fwd_mean_ms'], color=colors,
                edgecolor='white', linewidth=0.5)
        if 'single_fwd_std_ms' in df.columns:
            ax.errorbar(df['single_fwd_mean_ms'].values, y_pos,
                        xerr=df['single_fwd_std_ms'].fillna(0).values,
                        fmt='none', ecolor='gray', elinewidth=0.8, capsize=2)
        ax.set_xscale('log')
        ax.set_xlabel('Inference Time per Sample (ms, log scale)')


        ax.set_yticks(y_pos)
        ax.set_yticklabels(df['model'], fontsize=9)
        ax.invert_yaxis()

        # Legend
        handles = []
        for family, color in config.FAMILY_COLORS.items():
            if family in df['family'].values:
                handles.append(plt.Rectangle((0, 0), 1, 1, fc=color,
                                             label=family))
        ax.legend(handles=handles, loc='upper right', fontsize=9)

        ax.set_title('Inference Speed Benchmark (NVIDIA L4 GPU, 300s horizon)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()

        path = output_dir / 'bar_inference_by_model.png'
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f"OK Saved: {path.name}")
        return path

    def _plot_inference_bars_specialist(self, output_dir: Path) -> Path:
        """Horizontal bar chart: inference time per sample, specialist models only."""
        results = self.analyzer.analyze()
        inf = results.get('inference_summary', pd.DataFrame())
        if len(inf) == 0:
            return output_dir / 'bar_inference_by_model_specialist.png'

        # Use 300s horizon as representative
        df = inf[inf['horizon'] == '300s'].copy()
        if len(df) == 0:
            df = inf.drop_duplicates('model', keep='first')

        # Exclude foundation models
        exclude = {'chronos_ft', 'chronos', 'timesfm', 'chronos2', 'chronos2_ft'}
        df = df[~df['model'].str.lower().isin(exclude)]
        df = df.sort_values('single_fwd_mean_ms', ascending=True)

        fig, ax = plt.subplots(figsize=(10, max(8, len(df) * 0.35)))

        y_pos = np.arange(len(df))
        colors = [config.FAMILY_COLORS.get(f, '#999999') for f in df['family']]

        ax.barh(y_pos, df['single_fwd_mean_ms'], color=colors,
                edgecolor='white', linewidth=0.5)
        if 'single_fwd_std_ms' in df.columns:
            ax.errorbar(df['single_fwd_mean_ms'].values, y_pos,
                        xerr=df['single_fwd_std_ms'].fillna(0).values,
                        fmt='none', ecolor='gray', elinewidth=0.8, capsize=2)
        ax.set_xscale('log')
        ax.set_xlabel('Inference Time per Sample (ms, log scale)')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(df['model'], fontsize=9)
        ax.invert_yaxis()

        # Legend
        handles = []
        for family, color in config.FAMILY_COLORS.items():
            if family in df['family'].values:
                handles.append(plt.Rectangle((0, 0), 1, 1, fc=color,
                                             label=family))
        ax.legend(handles=handles, loc='upper right', fontsize=9)

        ax.set_title('Inference Speed Benchmark',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()

        path = output_dir / 'bar_inference_by_model_specialist.png'
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f"OK Saved: {path.name}")
        return path

    def _plot_efficiency_scatter(self, horizon: str,
                                  efficiency_df: pd.DataFrame,
                                  output_dir: Path) -> Path:
        """Scatter plot: inference time vs MAE with Pareto front."""
        df = efficiency_df[efficiency_df['horizon'] == horizon].copy()
        if len(df) == 0:
            path = output_dir / f'scatter_efficiency_{horizon}.png'
            return path

        fig, ax = plt.subplots(figsize=(12, 8))

        # Marker shapes by family
        family_markers = {
            'Transformers': 'o', 'CNNs': 's', 'RNNs': '^',
            'MLPs': 'D', 'Baselines': 'P', 'Foundation': '*'
        }

        for family in df['family'].unique():
            fam_df = df[df['family'] == family]
            marker = family_markers.get(family, 'o')
            color = config.FAMILY_COLORS.get(family, '#999999')

            ax.scatter(fam_df['single_fwd_mean_ms'], fam_df['mae'],
                       marker=marker, c=color, s=100, label=family,
                       edgecolors='black', linewidth=0.5, zorder=3)

        # Draw Pareto front
        pareto_df = df[df['pareto_optimal']].sort_values('single_fwd_mean_ms')
        if len(pareto_df) > 1:
            ax.plot(pareto_df['single_fwd_mean_ms'], pareto_df['mae'],
                    'k--', linewidth=1, alpha=0.5, zorder=2,
                    label='Pareto front')

        # Annotate all models
        for _, row in df.iterrows():
            fontweight = 'bold' if row.get('pareto_optimal', False) else 'normal'
            ax.annotate(
                row['model'],
                (row['single_fwd_mean_ms'], row['mae']),
                textcoords='offset points',
                xytext=(6, 4),
                fontsize=7,
                fontweight=fontweight,
                alpha=0.85
            )

        ax.set_xscale('log')
        ax.set_xlabel('Inference Time per Sample (ms, log scale)')
        ax.set_ylabel('MAE in Experiment 4 (K)')
        ax.set_title(f'Inference Efficiency: Speed vs Accuracy ({horizon} horizon)')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = output_dir / f'scatter_efficiency_{horizon}.png'
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f"OK Saved: {path.name}")
        return path


if __name__ == '__main__':
    print("=" * 80)
    print("Timing Plots Test")
    print("=" * 80)

    from timing_analyzer import TimingAnalyzer

    analyzer = TimingAnalyzer()
    config.create_output_directories()

    plotter = TimingPlots(analyzer)
    plots = plotter.generate_all()

    print(f"\nGenerated {len(plots)} timing plots")
    for name, path in plots.items():
        print(f"  - {name}: {path}")

    print("\nOK Timing plots test successful!")
