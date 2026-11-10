"""
Extended Metrics Calculator for exp0-4 Analysis Suite

Calculates additional performance metrics beyond MAE and RMSE.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import config


class MetricsCalculator:
    """Calculate extended performance metrics."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize metrics calculator.

        Args:
            df: DataFrame with columns: experiment, model, horizon, mae, rmse, family
        """
        self.df = df.copy()

    def calculate_all(self) -> pd.DataFrame:
        """
        Calculate all extended metrics.

        Returns:
            DataFrame with additional metric columns
        """
        print("Calculating extended metrics...")

        # Calculate baseline metrics
        self._calculate_baseline_comparisons()

        # Calculate efficiency metrics
        self._calculate_efficiency_metrics()

        # Calculate improvement metrics
        self._calculate_improvement_metrics()

        # Calculate rank-based metrics
        self._calculate_rank_metrics()

        print(f"OK Extended metrics calculated")
        print(f"OK Total columns: {len(self.df.columns)}")

        return self.df

    def _calculate_baseline_comparisons(self):
        """Calculate metrics relative to baseline models."""
        print("  - Calculating baseline comparisons...")

        # For each (experiment, horizon), calculate improvement vs baselines
        for (exp, horizon), group in self.df.groupby(['experiment', 'horizon']):
            # Get baseline MAE/RMSE
            baselines = {
                'naive': group[group['model'] == 'Naive']['mae'].values,
                'moving_avg': group[group['model'] == 'MovingAverage']['mae'].values,
                'arima': group[group['model'] == 'ARIMA']['mae'].values
            }

            # Calculate improvement vs each baseline
            for baseline_name, baseline_mae in baselines.items():
                if len(baseline_mae) > 0:
                    baseline_val = baseline_mae[0]
                    mask = (self.df['experiment'] == exp) & (self.df['horizon'] == horizon)
                    self.df.loc[mask, f'improvement_vs_{baseline_name}'] = \
                        (baseline_val - self.df.loc[mask, 'mae']) / baseline_val * 100

    def _calculate_efficiency_metrics(self):
        """Calculate error efficiency metrics."""
        print("  - Calculating efficiency metrics...")

        # MAE/RMSE ratio (indicates error distribution shape)
        # Ratio close to 1 → errors normally distributed
        # Ratio << 1 → presence of outliers
        self.df['mae_rmse_ratio'] = self.df['mae'] / self.df['rmse']

        # Geometric mean of errors (robust to outliers)
        self.df['gmean_error'] = np.sqrt(self.df['mae'] * self.df['rmse'])

        # Error efficiency score (lower is better)
        # Penalizes both MAE and RMSE
        self.df['error_efficiency'] = 1 / np.sqrt(self.df['mae'] * self.df['rmse'])

        # Normalized MAE (MAE / typical PCT value)
        self.df['normalized_mae'] = self.df['mae'] / config.PCT_MEAN

        # Relative error (approximate percentage)
        self.df['relative_error_pct'] = (self.df['mae'] / config.PCT_MEAN) * 100

    def _calculate_improvement_metrics(self):
        """Calculate improvement metrics vs exp0 baseline."""
        print("  - Calculating improvement vs exp0...")

        # For each (model, horizon), calculate improvement from exp0 to other experiments
        for (model, horizon), group in self.df.groupby(['model', 'horizon']):
            # Get exp0 baseline
            exp0_mae = group[group['experiment'] == 'exp0']['mae'].values

            if len(exp0_mae) > 0:
                baseline = exp0_mae[0]

                # Calculate improvement for all experiments
                mask = (self.df['model'] == model) & (self.df['horizon'] == horizon)
                self.df.loc[mask, 'improvement_vs_exp0_pct'] = \
                    (baseline - self.df.loc[mask, 'mae']) / baseline * 100

                # Flag if experiment is better than exp0
                self.df.loc[mask, 'better_than_exp0'] = \
                    self.df.loc[mask, 'mae'] < baseline

    def _calculate_rank_metrics(self):
        """Calculate ranking-based metrics."""
        print("  - Calculating rankings...")

        # Rank models within each (experiment, horizon) by MAE
        self.df['rank_mae'] = self.df.groupby(['experiment', 'horizon'])['mae'].rank(method='min')

        # Rank models within each (experiment, horizon) by RMSE
        self.df['rank_rmse'] = self.df.groupby(['experiment', 'horizon'])['rmse'].rank(method='min')

        # Calculate percentile rank (0-100, higher is better)
        # IMPORTANT: This uses a SCORE-BASED interpretation where:
        #   - percentile_rank = 100 means "best performer" (lowest MAE)
        #   - percentile_rank = 95 means "better than 95% of models"
        #   - percentile_rank = 0 means "worst performer" (highest MAE)
        #
        # This is INVERTED from traditional percentile ranking where higher rank
        # typically means higher value. Here, we invert it so higher percentile_rank
        # always means better performance (lower MAE).
        #
        # Formula: (n - rank) / n * 100
        #   where rank=1 is best (lowest MAE), rank=n is worst (highest MAE)
        #
        # Example with 25 models:
        #   Best model (MAE rank 1):  (25 - 1) / 25 * 100 = 96% (top performer)
        #   Worst model (MAE rank 25): (25 - 25) / 25 * 100 = 0% (bottom performer)
        def percentile_rank(x):
            return (len(x) - x.rank(method='min')) / len(x) * 100

        self.df['percentile_rank'] = self.df.groupby(['experiment', 'horizon'])['mae'].transform(percentile_rank)

        # Flag top-k models
        for k in [1, 3, 5, 10]:
            self.df[f'top{k}'] = self.df['rank_mae'] <= k

    def get_summary_by_family(self) -> pd.DataFrame:
        """Get summary statistics by architecture family."""
        summary = self.df.groupby(['family', 'experiment', 'horizon']).agg({
            'mae': ['mean', 'std', 'min', 'max'],
            'rmse': ['mean', 'std', 'min', 'max'],
            'model': 'count'
        }).reset_index()

        summary.columns = [
            'family', 'experiment', 'horizon',
            'mae_mean', 'mae_std', 'mae_min', 'mae_max',
            'rmse_mean', 'rmse_std', 'rmse_min', 'rmse_max',
            'n_models'
        ]

        return summary

    def get_summary_by_experiment(self) -> pd.DataFrame:
        """Get summary statistics by experiment."""
        summary = self.df.groupby(['experiment', 'horizon']).agg({
            'mae': ['mean', 'std', 'min', 'max'],
            'rmse': ['mean', 'std', 'min', 'max'],
            'improvement_vs_exp0_pct': ['mean', 'median'],
            'model': 'count'
        }).reset_index()

        summary.columns = [
            'experiment', 'horizon',
            'mae_mean', 'mae_std', 'mae_min', 'mae_max',
            'rmse_mean', 'rmse_std', 'rmse_min', 'rmse_max',
            'avg_improvement_vs_exp0', 'median_improvement_vs_exp0',
            'n_models'
        ]

        return summary

    def get_top_models(self, n: int = 5, horizon: str = None, experiment: str = None) -> pd.DataFrame:
        """
        Get top-N models by MAE.

        Args:
            n: Number of top models
            horizon: Filter by horizon (optional)
            experiment: Filter by experiment (optional)

        Returns:
            DataFrame with top models
        """
        df_filtered = self.df.copy()

        if horizon:
            df_filtered = df_filtered[df_filtered['horizon'] == horizon]
        if experiment:
            df_filtered = df_filtered[df_filtered['experiment'] == experiment]

        return df_filtered.nsmallest(n, 'mae')[
            ['experiment', 'model', 'family', 'horizon', 'mae', 'rmse', 'rank_mae']
        ]

    def save_extended_metrics(self, output_path: Path = None):
        """Save extended metrics to CSV."""
        if output_path is None:
            output_path = config.METRICS_DIR / 'extended_metrics.csv'

        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(output_path, index=False)
        print(f"OK Saved extended metrics to: {output_path}")

    def save_family_summary(self, output_path: Path = None):
        """Save family summary to CSV."""
        if output_path is None:
            output_path = config.METRICS_DIR / 'architecture_family_summary.csv'

        summary = self.get_summary_by_family()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_path, index=False)
        print(f"OK Saved family summary to: {output_path}")

    def save_experiment_summary(self, output_path: Path = None):
        """Save experiment summary to CSV."""
        if output_path is None:
            output_path = config.METRICS_DIR / 'experiment_summary.csv'

        summary = self.get_summary_by_experiment()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_path, index=False)
        print(f"OK Saved experiment summary to: {output_path}")


if __name__ == '__main__':
    # Test metrics calculator
    print("="*80)
    print("Metrics Calculator Test")
    print("="*80)
    print()

    # Load data
    from data_loader import DataLoader
    loader = DataLoader()
    df = loader.load()

    # Calculate extended metrics
    calculator = MetricsCalculator(df)
    extended_df = calculator.calculate_all()

    print("\n" + "="*80)
    print("Extended Metrics Columns")
    print("="*80)
    for col in extended_df.columns:
        print(f"  - {col}")

    # Test summary functions
    print("\n" + "="*80)
    print("Family Summary (exp1, 300s)")
    print("="*80)
    family_summary = calculator.get_summary_by_family()
    subset = family_summary[
        (family_summary['experiment'] == 'exp1') &
        (family_summary['horizon'] == '300s')
    ]
    print(subset[['family', 'mae_mean', 'mae_std', 'n_models']].to_string(index=False))

    print("\n" + "="*80)
    print("Top 5 Models (exp1, 300s)")
    print("="*80)
    top5 = calculator.get_top_models(n=5, experiment='exp1', horizon='300s')
    print(top5.to_string(index=False))

    # Save outputs
    print("\n" + "="*80)
    print("Saving Outputs")
    print("="*80)
    config.create_output_directories()
    calculator.save_extended_metrics()
    calculator.save_family_summary()
    calculator.save_experiment_summary()

    print("\nOK Metrics calculator test successful!")
