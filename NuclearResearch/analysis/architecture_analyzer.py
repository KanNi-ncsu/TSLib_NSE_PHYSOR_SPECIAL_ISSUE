"""
Architecture Family Analyzer for exp0-4 Analysis Suite

Analyzes performance patterns across architecture families.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
import config


class ArchitectureAnalyzer:
    """Analyze architecture family patterns and performance."""

    def __init__(self, df: pd.DataFrame):
        """Initialize architecture analyzer."""
        self.df = df.copy()

    def analyze(self) -> Dict[str, pd.DataFrame]:
        """Run all architecture family analyses."""
        print("Running architecture family analysis...")

        results = {}
        results['family_summary'] = self._create_family_summary()
        results['family_robustness'] = self._analyze_robustness()
        results['family_specialization'] = self._analyze_specialization()
        results['best_in_family'] = self._find_best_in_family()

        print("OK Architecture family analysis complete")
        return results

    def _create_family_summary(self) -> pd.DataFrame:
        """Create comprehensive family performance summary."""
        print("  - Creating family performance summary...")

        summary = self.df.groupby(['family', 'experiment', 'horizon']).agg({
            'mae': ['mean', 'std', 'min', 'max', 'median'],
            'rmse': ['mean', 'std', 'min', 'max', 'median'],
            'model': 'count'
        }).reset_index()

        summary.columns = [
            'family', 'experiment', 'horizon',
            'mae_mean', 'mae_std', 'mae_min', 'mae_max', 'mae_median',
            'rmse_mean', 'rmse_std', 'rmse_min', 'rmse_max', 'rmse_median',
            'n_models'
        ]

        # Add coefficient of variation
        summary['mae_cv'] = summary['mae_std'] / summary['mae_mean']

        return summary

    def _analyze_robustness(self) -> pd.DataFrame:
        """Analyze family robustness across experiments."""
        print("  - Analyzing family robustness...")

        robustness_data = []

        for family in config.MODEL_FAMILIES.keys():
            family_data = self.df[self.df['family'] == family]

            for horizon in config.HORIZON_ORDER:
                horizon_data = family_data[family_data['horizon'] == horizon]

                if len(horizon_data) == 0:
                    continue

                # Calculate std dev across experiments
                exp_means = []
                for exp in config.EXPERIMENT_ORDER:
                    exp_mae = horizon_data[horizon_data['experiment'] == exp]['mae'].mean()
                    if not np.isnan(exp_mae):
                        exp_means.append(exp_mae)

                if exp_means:
                    robustness_data.append({
                        'family': family,
                        'horizon': horizon,
                        'mean_mae': np.mean(exp_means),
                        'std_across_exps': np.std(exp_means),
                        'cv_across_exps': np.std(exp_means) / np.mean(exp_means),
                        'min_mae': np.min(exp_means),
                        'max_mae': np.max(exp_means),
                        'range': np.max(exp_means) - np.min(exp_means)
                    })

        return pd.DataFrame(robustness_data)

    def _analyze_specialization(self) -> pd.DataFrame:
        """Analyze which families specialize for which horizons."""
        print("  - Analyzing family specialization...")

        specialization_data = []

        for family in config.MODEL_FAMILIES.keys():
            family_data = self.df[self.df['family'] == family]

            row = {'family': family}

            for horizon in config.HORIZON_ORDER:
                horizon_data = family_data[family_data['horizon'] == horizon]
                if len(horizon_data) > 0:
                    row[f'{horizon}_mean_mae'] = horizon_data['mae'].mean()
                    row[f'{horizon}_best_mae'] = horizon_data['mae'].min()

            # Calculate horizon preference (which horizon has lowest error?)
            mean_maes = {h: row.get(f'{h}_mean_mae', np.inf) for h in config.HORIZON_ORDER}
            row['preferred_horizon'] = min(mean_maes, key=mean_maes.get)

            specialization_data.append(row)

        return pd.DataFrame(specialization_data)

    def _find_best_in_family(self) -> pd.DataFrame:
        """Find champion model in each family."""
        print("  - Finding best model per family...")

        champions_data = []

        for family in config.MODEL_FAMILIES.keys():
            family_data = self.df[self.df['family'] == family]

            for (exp, horizon), group in family_data.groupby(['experiment', 'horizon']):
                best_idx = group['mae'].idxmin()
                best = group.loc[best_idx]

                champions_data.append({
                    'family': family,
                    'experiment': exp,
                    'horizon': horizon,
                    'champion_model': best['model'],
                    'mae': best['mae'],
                    'rmse': best['rmse'],
                    'rank': best.get('rank_mae', np.nan)
                })

        return pd.DataFrame(champions_data)

    def save_all(self, output_dir: Path = None):
        """Save all analysis results."""
        if output_dir is None:
            output_dir = config.METRICS_DIR

        output_dir.mkdir(parents=True, exist_ok=True)

        results = self.analyze()

        for name, df in results.items():
            output_path = output_dir / f'architecture_{name}.csv'
            df.to_csv(output_path, index=False)
            print(f"OK Saved: architecture_{name}.csv")


if __name__ == '__main__':
    print("="*80)
    print("Architecture Analyzer Test")
    print("="*80)

    from data_loader import DataLoader
    from metrics_calculator import MetricsCalculator

    loader = DataLoader()
    df = loader.load()
    calculator = MetricsCalculator(df)
    extended_df = calculator.calculate_all()

    analyzer = ArchitectureAnalyzer(extended_df)
    results = analyzer.analyze()

    print("\n" + "="*80)
    print("Family Robustness (300s)")
    print("="*80)
    robustness = results['family_robustness']
    sample = robustness[robustness['horizon'] == '300s']
    print(sample.to_string(index=False))

    analyzer.save_all()
    print("\nOK Architecture analyzer test successful!")
