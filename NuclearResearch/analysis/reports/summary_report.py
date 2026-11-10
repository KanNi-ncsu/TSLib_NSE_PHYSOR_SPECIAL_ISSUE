"""
Summary Report Generator for exp0-4 Analysis Suite

Creates comprehensive text-based summary reports.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).parent.parent))
import config


class SummaryReport:
    """Generate summary reports from analysis results."""

    def __init__(self, df: pd.DataFrame):
        """Initialize report generator."""
        self.df = df.copy()

    def generate_all(self, output_dir: Path = None) -> Dict[str, Path]:
        """Generate all summary reports."""
        if output_dir is None:
            output_dir = config.REPORTS_DIR

        output_dir.mkdir(parents=True, exist_ok=True)

        print("Generating summary reports...")

        reports = {}

        # 1. Executive summary
        path = self._create_executive_summary(output_dir)
        reports['executive_summary'] = path

        # 2. Per-experiment report
        for exp in config.EXPERIMENT_ORDER:
            path = self._create_experiment_report(exp, output_dir)
            reports[f'experiment_{exp}'] = path

        # 3. Model comparison report
        path = self._create_model_comparison_report(output_dir)
        reports['model_comparison'] = path

        # 4. Architecture family report
        path = self._create_family_report(output_dir)
        reports['family_report'] = path

        print(f"OK Generated {len(reports)} summary reports")
        return reports

    def _create_executive_summary(self, output_dir: Path) -> Path:
        """Create high-level executive summary."""
        print("  - Creating executive summary...")

        output_path = output_dir / 'executive_summary.txt'

        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("EXECUTIVE SUMMARY: exp0-4 Analysis Results\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Key findings
            f.write("KEY FINDINGS:\n")
            f.write("-" * 80 + "\n\n")

            # Best overall model
            best_model_300s = self.df[
                (self.df['horizon'] == '300s') &
                (self.df['experiment'] == 'exp1')
            ].nsmallest(1, 'mae').iloc[0]

            f.write(f"1. Best Performing Model (300s, exp1):\n")
            f.write(f"   Model: {best_model_300s['model']}\n")
            f.write(f"   MAE: {best_model_300s['mae']:.2f}\n")
            f.write(f"   RMSE: {best_model_300s['rmse']:.2f}\n")
            f.write(f"   Family: {best_model_300s['family']}\n\n")

            # Experiment comparison
            f.write("2. Experiment Performance Summary (300s horizon):\n")
            exp_summary = self.df[self.df['horizon'] == '300s'].groupby('experiment')['mae'].agg([
                'mean', 'std', 'min'
            ]).round(2)

            for exp in config.EXPERIMENT_ORDER:
                if exp in exp_summary.index:
                    row = exp_summary.loc[exp]
                    f.write(f"   {exp} ({config.EXPERIMENTS[exp]['name']}): ")
                    f.write(f"Best={row['min']:.2f}, Avg={row['mean']:.2f}±{row['std']:.2f}\n")

            f.write("\n")

            # Family performance
            f.write("3. Architecture Family Rankings (300s, exp1):\n")
            family_summary = self.df[
                (self.df['horizon'] == '300s') &
                (self.df['experiment'] == 'exp1')
            ].groupby('family')['mae'].mean().sort_values()

            for idx, (family, mae) in enumerate(family_summary.items(), 1):
                f.write(f"   {idx}. {family}: {mae:.2f} MAE\n")

            f.write("\n")

            # Consistency analysis
            f.write("4. Model Consistency Across Horizons (exp1):\n")
            f.write("   Top-3 models with consistent performance:\n")

            # Find models with best average rank across horizons
            exp1_data = self.df[self.df['experiment'] == 'exp1']
            avg_rank = exp1_data.groupby('model')['rank_mae'].mean().sort_values()

            for idx, (model, avg_rank_val) in enumerate(avg_rank.head(3).items(), 1):
                model_perf = exp1_data[exp1_data['model'] == model]
                f.write(f"   {idx}. {model} (Avg Rank: {avg_rank_val:.1f})\n")
                for _, row in model_perf.iterrows():
                    f.write(f"      {row['horizon']}: MAE={row['mae']:.2f}, Rank={row['rank_mae']:.0f}\n")

            f.write("\n")

            # Transfer learning benefit
            f.write("5. Transfer Learning Insights:\n")
            exp0_best = self.df[
                (self.df['experiment'] == 'exp0') &
                (self.df['horizon'] == '300s')
            ]['mae'].min()

            exp1_best = self.df[
                (self.df['experiment'] == 'exp1') &
                (self.df['horizon'] == '300s')
            ]['mae'].min()

            improvement = ((exp0_best - exp1_best) / exp0_best) * 100

            f.write(f"   Adding normal operation data (exp0→exp1):\n")
            f.write(f"   Best MAE: {exp0_best:.2f} → {exp1_best:.2f}\n")
            f.write(f"   Improvement: {improvement:.1f}%\n\n")

            # Recommendations
            f.write("RECOMMENDATIONS:\n")
            f.write("-" * 80 + "\n\n")

            top_model = best_model_300s['model']
            f.write(f"1. For production deployment, consider {top_model} ")
            f.write(f"({best_model_300s['family']}) as it achieves the best MAE of {best_model_300s['mae']:.2f}\n\n")

            best_exp_idx = exp_summary['min'].idxmin()
            f.write(f"2. Training Configuration: Use {best_exp_idx} ")
            f.write(f"({config.EXPERIMENTS[best_exp_idx]['name']}) dataset composition\n\n")

            top_family = family_summary.index[0]
            f.write(f"3. Architecture Selection: {top_family} family shows best overall performance\n\n")

            f.write("="*80 + "\n")

        return output_path

    def _create_experiment_report(self, experiment: str, output_dir: Path) -> Path:
        """Create detailed report for a specific experiment."""
        print(f"  - Creating {experiment} report...")

        output_path = output_dir / f'experiment_{experiment}_report.txt'

        exp_data = self.df[self.df['experiment'] == experiment]

        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"EXPERIMENT REPORT: {experiment}\n")
            f.write("="*80 + "\n\n")

            # Metadata
            exp_meta = config.EXPERIMENTS[experiment]
            f.write(f"Name: {exp_meta['name']}\n")
            f.write(f"Description: {exp_meta['description']}\n")
            f.write(f"Dataset Size: {exp_meta['n_scenarios']} scenarios\n")
            f.write(f"Composition: {exp_meta['composition']}\n\n")

            # Performance by horizon
            f.write("PERFORMANCE BY HORIZON:\n")
            f.write("-" * 80 + "\n\n")

            for horizon in config.HORIZON_ORDER:
                horizon_data = exp_data[exp_data['horizon'] == horizon]

                if len(horizon_data) == 0:
                    f.write(f"{horizon}: No data available\n\n")
                    continue

                f.write(f"{horizon} Ahead Forecasting:\n")
                f.write(f"  Models Evaluated: {len(horizon_data)}\n")

                # Top-5 models
                top5 = horizon_data.nsmallest(5, 'mae')
                f.write(f"  Top-5 Models:\n")
                for idx, row in top5.iterrows():
                    f.write(f"    {row['rank_mae']:.0f}. {row['model']:15s} ")
                    f.write(f"MAE={row['mae']:7.2f}  RMSE={row['rmse']:7.2f}  ")
                    f.write(f"({row['family']})\n")

                # Statistics
                f.write(f"  Statistics:\n")
                f.write(f"    Mean MAE: {horizon_data['mae'].mean():.2f}\n")
                f.write(f"    Std MAE:  {horizon_data['mae'].std():.2f}\n")
                f.write(f"    Best MAE: {horizon_data['mae'].min():.2f}\n")
                f.write(f"    Worst MAE: {horizon_data['mae'].max():.2f}\n\n")

            # Family analysis
            f.write("ARCHITECTURE FAMILY PERFORMANCE:\n")
            f.write("-" * 80 + "\n\n")

            family_perf = exp_data.groupby('family')['mae'].agg(['mean', 'std', 'min', 'count'])
            family_perf = family_perf.sort_values('mean')

            for family, row in family_perf.iterrows():
                f.write(f"{family:15s} ")
                f.write(f"Avg={row['mean']:7.2f}±{row['std']:6.2f}  ")
                f.write(f"Best={row['min']:7.2f}  ")
                f.write(f"Models={row['count']:.0f}\n")

            f.write("\n" + "="*80 + "\n")

        return output_path

    def _create_model_comparison_report(self, output_dir: Path) -> Path:
        """Create comprehensive model comparison report."""
        print("  - Creating model comparison report...")

        output_path = output_dir / 'model_comparison_report.txt'

        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MODEL COMPARISON REPORT\n")
            f.write("="*80 + "\n\n")

            # Focus on exp1, 300s
            main_data = self.df[
                (self.df['experiment'] == 'exp1') &
                (self.df['horizon'] == '300s')
            ].sort_values('mae')

            f.write("Primary Analysis: exp1 (LOCA+Normal), 300s Horizon\n")
            f.write("-" * 80 + "\n\n")

            # Full ranking table
            f.write("COMPLETE MODEL RANKINGS:\n\n")
            f.write(f"{'Rank':<6}{'Model':<18}{'Family':<15}{'MAE':<10}{'RMSE':<10}{'Ratio':<8}\n")
            f.write("-" * 80 + "\n")

            for idx, row in main_data.iterrows():
                f.write(f"{row['rank_mae']:<6.0f}{row['model']:<18}")
                f.write(f"{row['family']:<15}{row['mae']:<10.2f}")
                f.write(f"{row['rmse']:<10.2f}{row['mae_rmse_ratio']:<8.3f}\n")

            f.write("\n")

            # Cross-experiment consistency
            f.write("CROSS-EXPERIMENT CONSISTENCY (300s):\n")
            f.write("-" * 80 + "\n\n")

            f.write(f"{'Model':<18}")
            for exp in ['exp0', 'exp1', 'exp2', 'exp3']:
                f.write(f"{exp:<10}")
            f.write(f"{'Std':<8}\n")
            f.write("-" * 80 + "\n")

            # Get top-10 models from exp1
            top_models = main_data.head(10)['model'].tolist()

            for model in top_models:
                f.write(f"{model:<18}")
                model_data = self.df[
                    (self.df['model'] == model) &
                    (self.df['horizon'] == '300s')
                ]

                maes = []
                for exp in ['exp0', 'exp1', 'exp2', 'exp3']:
                    exp_row = model_data[model_data['experiment'] == exp]
                    if len(exp_row) > 0:
                        mae = exp_row.iloc[0]['mae']
                        f.write(f"{mae:<10.2f}")
                        maes.append(mae)
                    else:
                        f.write(f"{'N/A':<10}")

                if len(maes) > 1:
                    f.write(f"{np.std(maes):<8.2f}")

                f.write("\n")

            f.write("\n" + "="*80 + "\n")

        return output_path

    def _create_family_report(self, output_dir: Path) -> Path:
        """Create architecture family analysis report."""
        print("  - Creating architecture family report...")

        output_path = output_dir / 'architecture_family_report.txt'

        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ARCHITECTURE FAMILY ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")

            # Summary table across experiments
            f.write("FAMILY PERFORMANCE ACROSS EXPERIMENTS (300s):\n")
            f.write("-" * 80 + "\n\n")

            f.write(f"{'Family':<15}")
            for exp in config.EXPERIMENT_ORDER:
                f.write(f"{exp:<12}")
            f.write("\n")
            f.write("-" * 80 + "\n")

            for family in config.MODEL_FAMILIES.keys():
                f.write(f"{family:<15}")

                for exp in config.EXPERIMENT_ORDER:
                    family_exp_data = self.df[
                        (self.df['family'] == family) &
                        (self.df['experiment'] == exp) &
                        (self.df['horizon'] == '300s')
                    ]

                    if len(family_exp_data) > 0:
                        avg_mae = family_exp_data['mae'].mean()
                        f.write(f"{avg_mae:<12.2f}")
                    else:
                        f.write(f"{'N/A':<12}")

                f.write("\n")

            f.write("\n")

            # Best model per family
            f.write("BEST MODEL PER FAMILY (exp1, 300s):\n")
            f.write("-" * 80 + "\n\n")

            exp1_300s = self.df[
                (self.df['experiment'] == 'exp1') &
                (self.df['horizon'] == '300s')
            ]

            for family in config.MODEL_FAMILIES.keys():
                family_data = exp1_300s[exp1_300s['family'] == family]

                if len(family_data) > 0:
                    best = family_data.nsmallest(1, 'mae').iloc[0]
                    f.write(f"{family:15s} → {best['model']:18s} ")
                    f.write(f"MAE={best['mae']:7.2f}  Rank={best['rank_mae']:.0f}\n")

            f.write("\n")

            # Robustness analysis
            f.write("ROBUSTNESS ACROSS HORIZONS (exp1):\n")
            f.write("(Lower std = more consistent across horizons)\n")
            f.write("-" * 80 + "\n\n")

            exp1_data = self.df[self.df['experiment'] == 'exp1']
            robustness = exp1_data.groupby('family')['mae'].std().sort_values()

            for family, std_val in robustness.items():
                f.write(f"{family:15s} Std={std_val:7.2f}\n")

            f.write("\n" + "="*80 + "\n")

        return output_path


if __name__ == '__main__':
    print("="*80)
    print("Summary Report Test")
    print("="*80)

    from data_loader import DataLoader
    from metrics_calculator import MetricsCalculator

    # Load data
    loader = DataLoader()
    df = loader.load()
    calculator = MetricsCalculator(df)
    extended_df = calculator.calculate_all()

    # Generate reports
    reporter = SummaryReport(extended_df)
    reports = reporter.generate_all()

    print("\n" + "="*80)
    print("Generated Summary Reports")
    print("="*80)
    for name, path in reports.items():
        print(f"OK {name}: {path.name}")

    print("\nOK Summary report test successful!")
