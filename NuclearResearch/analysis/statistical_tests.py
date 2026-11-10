"""
Statistical Tests for exp0-4 Analysis Suite

Performs pairwise comparisons and significance testing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
from scipy import stats
import config


class StatisticalTests:
    """Perform statistical significance tests on model performance."""

    def __init__(self, df: pd.DataFrame):
        """Initialize statistical tests."""
        self.df = df.copy()

    def run_all(self) -> Dict[str, pd.DataFrame]:
        """Run all statistical tests."""
        print("Running statistical tests...")

        results = {}
        results['pairwise_comparisons'] = self._pairwise_model_comparisons()
        results['rank_correlations'] = self._calculate_rank_correlations()

        print("OK Statistical tests complete")
        return results

    def _pairwise_model_comparisons(self) -> Dict[str, pd.DataFrame]:
        """
        Pairwise Wilcoxon signed-rank tests for top models.

        Compares Model A vs Model B using MAE differences across experiments
        as paired samples. Tests null hypothesis: median(diff) = 0.

        p < 0.05: Models are significantly different
        p >= 0.05: No significant difference
        """
        print("  - Running pairwise model comparisons...")

        comparison_matrices = {}

        for horizon in config.HORIZON_ORDER:
            horizon_data = self.df[self.df['horizon'] == horizon]

            # Get top-10 models by average MAE across all experiments
            model_avg_mae = horizon_data.groupby('model')['mae'].mean().sort_values()
            top_models = model_avg_mae.head(10).index.tolist()

            # Create p-value matrix
            n_models = len(top_models)
            p_matrix = pd.DataFrame(
                np.zeros((n_models, n_models)),
                index=top_models,
                columns=top_models
            )

            # Perform pairwise comparisons
            for i, model_a in enumerate(top_models):
                for j, model_b in enumerate(top_models):
                    if i == j:
                        p_matrix.iloc[i, j] = 1.0  # Same model
                    elif i < j:  # Only compute upper triangle
                        # Get MAE values across experiments for both models
                        mae_a = []
                        mae_b = []

                        for exp in config.EXPERIMENT_ORDER:
                            mae_a_exp = horizon_data[
                                (horizon_data['model'] == model_a) &
                                (horizon_data['experiment'] == exp)
                            ]['mae'].values

                            mae_b_exp = horizon_data[
                                (horizon_data['model'] == model_b) &
                                (horizon_data['experiment'] == exp)
                            ]['mae'].values

                            if len(mae_a_exp) > 0 and len(mae_b_exp) > 0:
                                mae_a.append(mae_a_exp[0])
                                mae_b.append(mae_b_exp[0])

                        if len(mae_a) >= 3:  # Need at least 3 pairs for Wilcoxon
                            try:
                                statistic, p_value = stats.wilcoxon(mae_a, mae_b)
                                p_matrix.iloc[i, j] = p_value
                                p_matrix.iloc[j, i] = p_value  # Symmetric
                            except:
                                p_matrix.iloc[i, j] = np.nan
                                p_matrix.iloc[j, i] = np.nan
                        else:
                            p_matrix.iloc[i, j] = np.nan
                            p_matrix.iloc[j, i] = np.nan

            comparison_matrices[horizon] = p_matrix

        return comparison_matrices

    def _calculate_rank_correlations(self) -> pd.DataFrame:
        """
        Calculate rank correlations between experiments.

        Uses Kendall's tau to measure rank stability:
        - τ close to 1: Very stable rankings
        - τ close to 0: No correlation (rankings change dramatically)
        """
        print("  - Calculating rank correlations...")

        correlation_data = []

        for horizon in config.HORIZON_ORDER:
            horizon_data = self.df[self.df['horizon'] == horizon]

            # For each pair of experiments
            for i, exp_a in enumerate(config.EXPERIMENT_ORDER):
                for exp_b in config.EXPERIMENT_ORDER[i+1:]:
                    # Get common models
                    models_a = set(horizon_data[horizon_data['experiment'] == exp_a]['model'])
                    models_b = set(horizon_data[horizon_data['experiment'] == exp_b]['model'])
                    common_models = list(models_a & models_b)

                    if len(common_models) >= 3:
                        # Get ranks for common models
                        ranks_a = []
                        ranks_b = []

                        for model in common_models:
                            rank_a = horizon_data[
                                (horizon_data['experiment'] == exp_a) &
                                (horizon_data['model'] == model)
                            ]['rank_mae'].values

                            rank_b = horizon_data[
                                (horizon_data['experiment'] == exp_b) &
                                (horizon_data['model'] == model)
                            ]['rank_mae'].values

                            if len(rank_a) > 0 and len(rank_b) > 0:
                                ranks_a.append(rank_a[0])
                                ranks_b.append(rank_b[0])

                        if len(ranks_a) >= 3:
                            tau, p_value = stats.kendalltau(ranks_a, ranks_b)

                            correlation_data.append({
                                'horizon': horizon,
                                'experiment_a': exp_a,
                                'experiment_b': exp_b,
                                'kendall_tau': tau,
                                'p_value': p_value,
                                'n_models': len(ranks_a),
                                'significant': p_value < config.ALPHA
                            })

        return pd.DataFrame(correlation_data)

    def get_significance_summary(self, horizon: str = '300s') -> pd.DataFrame:
        """
        Get summary of significant differences between top models.

        Args:
            horizon: Horizon to analyze

        Returns:
            DataFrame with significant comparisons
        """
        results = self.run_all()
        p_matrix = results['pairwise_comparisons'][horizon]

        # Find significant differences (p < 0.05)
        significant_pairs = []

        for i, model_a in enumerate(p_matrix.index):
            for j, model_b in enumerate(p_matrix.columns):
                if i < j:  # Upper triangle only
                    p_value = p_matrix.iloc[i, j]
                    if not np.isnan(p_value) and p_value < config.ALPHA:
                        significant_pairs.append({
                            'model_a': model_a,
                            'model_b': model_b,
                            'p_value': p_value,
                            'significant': True
                        })

        return pd.DataFrame(significant_pairs)

    def save_all(self, output_dir: Path = None):
        """Save all statistical test results."""
        if output_dir is None:
            output_dir = config.STATISTICAL_DIR

        output_dir.mkdir(parents=True, exist_ok=True)

        results = self.run_all()

        # Save pairwise comparison matrices
        for horizon, matrix in results['pairwise_comparisons'].items():
            output_path = output_dir / f'pairwise_comparisons_{horizon}.csv'
            matrix.to_csv(output_path)
            print(f"OK Saved: pairwise_comparisons_{horizon}.csv")

        # Save rank correlations
        results['rank_correlations'].to_csv(
            output_dir / 'rank_correlations.csv', index=False
        )
        print(f"OK Saved: rank_correlations.csv")

        # Save significance summaries
        for horizon in config.HORIZON_ORDER:
            summary = self.get_significance_summary(horizon)
            summary.to_csv(
                output_dir / f'significant_differences_{horizon}.csv', index=False
            )
            print(f"OK Saved: significant_differences_{horizon}.csv")


if __name__ == '__main__':
    print("="*80)
    print("Statistical Tests")
    print("="*80)

    from data_loader import DataLoader
    from metrics_calculator import MetricsCalculator

    loader = DataLoader()
    df = loader.load()
    calculator = MetricsCalculator(df)
    extended_df = calculator.calculate_all()

    tester = StatisticalTests(extended_df)
    results = tester.run_all()

    print("\n" + "="*80)
    print("Rank Correlations (300s)")
    print("="*80)
    correlations = results['rank_correlations']
    sample = correlations[correlations['horizon'] == '300s']
    print(sample.to_string(index=False))

    print("\n" + "="*80)
    print("Significant Differences (300s, p < 0.05)")
    print("="*80)
    sig_diff = tester.get_significance_summary('300s')
    if len(sig_diff) > 0:
        print(sig_diff.head(10).to_string(index=False))
    else:
        print("No significant differences found")

    tester.save_all()
    print("\nOK Statistical tests successful!")
