"""
Cross-Experiment Analyzer for exp0-4 Analysis Suite

Analyzes transfer learning effectiveness and data composition impact.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import config


class CrossExperimentAnalyzer:
    """Analyze cross-experiment patterns and transfer learning benefits."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize cross-experiment analyzer.

        Args:
            df: DataFrame with extended metrics
        """
        self.df = df.copy()

    def analyze(self) -> Dict[str, pd.DataFrame]:
        """
        Run all cross-experiment analyses.

        Returns:
            Dictionary containing analysis results
        """
        print("Running cross-experiment analysis...")

        results = {}

        # 1. Improvement matrix (vs exp0)
        results['improvement_matrix'] = self._create_improvement_matrix()

        # 2. Experiment rankings
        results['experiment_rankings'] = self._rank_experiments()

        # 3. Data efficiency analysis
        results['data_efficiency'] = self._analyze_data_efficiency()

        # 4. Best experiment per model
        results['best_experiment_per_model'] = self._find_best_experiment_per_model()

        # 5. Win/loss matrix
        results['win_loss_matrix'] = self._create_win_loss_matrix()

        print("OK Cross-experiment analysis complete")

        return results

    def _create_improvement_matrix(self) -> pd.DataFrame:
        """
        Create improvement matrix showing % change vs exp0.

        Returns:
            DataFrame with models as rows, experiments as columns
        """
        print("  - Creating improvement matrix vs exp0...")

        improvement_data = []

        for (model, horizon), group in self.df.groupby(['model', 'horizon']):
            row = {'model': model, 'horizon': horizon}

            # Get exp0 baseline
            exp0_mae = group[group['experiment'] == 'exp0']['mae'].values

            if len(exp0_mae) > 0:
                baseline = exp0_mae[0]

                # Calculate improvement for each experiment
                for exp in config.EXPERIMENT_ORDER:
                    exp_mae = group[group['experiment'] == exp]['mae'].values
                    if len(exp_mae) > 0:
                        improvement = (baseline - exp_mae[0]) / baseline * 100
                        row[exp] = improvement
                    else:
                        row[exp] = np.nan

            improvement_data.append(row)

        return pd.DataFrame(improvement_data)

    def _rank_experiments(self) -> pd.DataFrame:
        """
        Rank experiments by MAE for each (model, horizon).

        Returns:
            DataFrame with experiment ranks
        """
        print("  - Ranking experiments for each model...")

        ranking_data = []

        for (model, horizon), group in self.df.groupby(['model', 'horizon']):
            # Sort by MAE
            sorted_group = group.sort_values('mae')

            row = {
                'model': model,
                'horizon': horizon,
                'family': group['family'].iloc[0]
            }

            # Add ranks
            for rank, (_, exp_row) in enumerate(sorted_group.iterrows(), 1):
                exp = exp_row['experiment']
                row[f'{exp}_rank'] = rank
                row[f'{exp}_mae'] = exp_row['mae']

            # Add best experiment
            row['best_experiment'] = sorted_group.iloc[0]['experiment']
            row['best_mae'] = sorted_group.iloc[0]['mae']

            ranking_data.append(row)

        return pd.DataFrame(ranking_data)

    def _analyze_data_efficiency(self) -> pd.DataFrame:
        """
        Analyze performance vs dataset size.

        Returns:
            DataFrame with data efficiency metrics
        """
        print("  - Analyzing data efficiency...")

        efficiency_data = []

        for horizon in config.HORIZON_ORDER:
            horizon_data = self.df[self.df['horizon'] == horizon]

            for exp in config.EXPERIMENT_ORDER:
                exp_data = horizon_data[horizon_data['experiment'] == exp]

                if len(exp_data) > 0:
                    efficiency_data.append({
                        'experiment': exp,
                        'horizon': horizon,
                        'n_scenarios': config.EXPERIMENTS[exp]['n_scenarios'],
                        'best_mae': exp_data['mae'].min(),
                        'avg_mae': exp_data['mae'].mean(),
                        'best_model': exp_data.loc[exp_data['mae'].idxmin(), 'model']
                    })

        df_efficiency = pd.DataFrame(efficiency_data)

        # Calculate marginal benefit (improvement per additional scenario vs exp0)
        for horizon in config.HORIZON_ORDER:
            horizon_mask = df_efficiency['horizon'] == horizon
            exp0_scenarios = df_efficiency[
                (horizon_mask) & (df_efficiency['experiment'] == 'exp0')
            ]['n_scenarios'].values[0]
            exp0_mae = df_efficiency[
                (horizon_mask) & (df_efficiency['experiment'] == 'exp0')
            ]['best_mae'].values[0]

            for idx, row in df_efficiency[horizon_mask].iterrows():
                if row['experiment'] != 'exp0':
                    additional_scenarios = row['n_scenarios'] - exp0_scenarios
                    mae_improvement = exp0_mae - row['best_mae']
                    df_efficiency.loc[idx, 'marginal_benefit'] = \
                        mae_improvement / additional_scenarios if additional_scenarios > 0 else 0

        return df_efficiency

    def _find_best_experiment_per_model(self) -> pd.DataFrame:
        """
        Find which experiment is best for each model.

        Returns:
            DataFrame with best experiment recommendations
        """
        print("  - Finding best experiment per model...")

        best_exp_data = []

        for (model, horizon), group in self.df.groupby(['model', 'horizon']):
            best_idx = group['mae'].idxmin()
            best_row = group.loc[best_idx]

            best_exp_data.append({
                'model': model,
                'family': best_row['family'],
                'horizon': horizon,
                'best_experiment': best_row['experiment'],
                'best_mae': best_row['mae'],
                'best_rmse': best_row['rmse'],
                'improvement_vs_exp0': best_row.get('improvement_vs_exp0_pct', np.nan)
            })

        return pd.DataFrame(best_exp_data)

    def _create_win_loss_matrix(self) -> Dict[str, pd.DataFrame]:
        """
        Create win/loss matrices comparing experiments pairwise.

        Returns:
            Dictionary with win/loss matrices for each horizon
        """
        print("  - Creating win/loss matrices...")

        matrices = {}

        for horizon in config.HORIZON_ORDER:
            horizon_data = self.df[self.df['horizon'] == horizon]

            # Create matrix: rows=exp1, cols=exp2, value=# wins for exp1
            win_matrix = pd.DataFrame(
                index=config.EXPERIMENT_ORDER,
                columns=config.EXPERIMENT_ORDER,
                dtype=int
            )

            for exp1 in config.EXPERIMENT_ORDER:
                for exp2 in config.EXPERIMENT_ORDER:
                    if exp1 == exp2:
                        win_matrix.loc[exp1, exp2] = 0
                    else:
                        # Count how many models perform better in exp1 vs exp2
                        wins = 0
                        for model in horizon_data['model'].unique():
                            mae1 = horizon_data[
                                (horizon_data['model'] == model) &
                                (horizon_data['experiment'] == exp1)
                            ]['mae'].values

                            mae2 = horizon_data[
                                (horizon_data['model'] == model) &
                                (horizon_data['experiment'] == exp2)
                            ]['mae'].values

                            if len(mae1) > 0 and len(mae2) > 0:
                                if mae1[0] < mae2[0]:
                                    wins += 1

                        win_matrix.loc[exp1, exp2] = wins

            matrices[horizon] = win_matrix

        return matrices

    def get_transfer_learning_summary(self) -> pd.DataFrame:
        """Get summary of transfer learning benefits by family."""
        summary_data = []

        for family in config.MODEL_FAMILIES.keys():
            family_data = self.df[self.df['family'] == family]

            for horizon in config.HORIZON_ORDER:
                horizon_data = family_data[family_data['horizon'] == horizon]

                if len(horizon_data) == 0:
                    continue

                # Average improvement for each experiment vs exp0
                for exp in ['exp1', 'exp2', 'exp3', 'exp4']:
                    exp_data = horizon_data[horizon_data['experiment'] == exp]
                    if len(exp_data) > 0 and 'improvement_vs_exp0_pct' in exp_data.columns:
                        avg_improvement = exp_data['improvement_vs_exp0_pct'].mean()

                        summary_data.append({
                            'family': family,
                            'horizon': horizon,
                            'target_experiment': exp,
                            'avg_improvement_vs_exp0': avg_improvement,
                            'n_models': len(exp_data)
                        })

        return pd.DataFrame(summary_data)

    def save_all(self, output_dir: Path = None):
        """Save all analysis results."""
        if output_dir is None:
            output_dir = config.METRICS_DIR

        output_dir.mkdir(parents=True, exist_ok=True)

        results = self.analyze()

        # Save improvement matrix
        results['improvement_matrix'].to_csv(
            output_dir / 'cross_experiment_improvements.csv', index=False
        )
        print(f"OK Saved: cross_experiment_improvements.csv")

        # Save experiment rankings
        results['experiment_rankings'].to_csv(
            output_dir / 'experiment_rankings.csv', index=False
        )
        print(f"OK Saved: experiment_rankings.csv")

        # Save data efficiency
        results['data_efficiency'].to_csv(
            output_dir / 'data_efficiency_analysis.csv', index=False
        )
        print(f"OK Saved: data_efficiency_analysis.csv")

        # Save best experiment per model
        results['best_experiment_per_model'].to_csv(
            output_dir / 'best_experiment_per_model.csv', index=False
        )
        print(f"OK Saved: best_experiment_per_model.csv")

        # Save win/loss matrices
        for horizon, matrix in results['win_loss_matrix'].items():
            matrix.to_csv(output_dir / f'win_loss_matrix_{horizon}.csv')
            print(f"OK Saved: win_loss_matrix_{horizon}.csv")

        # Save transfer learning summary
        transfer_summary = self.get_transfer_learning_summary()
        transfer_summary.to_csv(
            output_dir / 'transfer_learning_summary.csv', index=False
        )
        print(f"OK Saved: transfer_learning_summary.csv")


if __name__ == '__main__':
    # Test cross-experiment analyzer
    print("="*80)
    print("Cross-Experiment Analyzer Test")
    print("="*80)
    print()

    # Load data with extended metrics
    from data_loader import DataLoader
    from metrics_calculator import MetricsCalculator

    loader = DataLoader()
    df = loader.load()
    calculator = MetricsCalculator(df)
    extended_df = calculator.calculate_all()

    # Run analysis
    analyzer = CrossExperimentAnalyzer(extended_df)
    results = analyzer.analyze()

    # Display sample results
    print("\n" + "="*80)
    print("Sample: Improvement Matrix (first 5 models, 300s)")
    print("="*80)
    improvement = results['improvement_matrix']
    sample = improvement[improvement['horizon'] == '300s'].head(5)
    print(sample.to_string(index=False))

    print("\n" + "="*80)
    print("Sample: Data Efficiency (300s)")
    print("="*80)
    efficiency = results['data_efficiency']
    sample = efficiency[efficiency['horizon'] == '300s']
    print(sample.to_string(index=False))

    # Save all results
    print("\n" + "="*80)
    print("Saving Results")
    print("="*80)
    analyzer.save_all()

    print("\nOK Cross-experiment analyzer test successful!")
