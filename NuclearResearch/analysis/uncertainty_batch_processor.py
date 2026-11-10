"""
Batch Processor for Uncertainty Analysis

Processes all result directories and generates comprehensive uncertainty metrics CSVs.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
import config
from uncertainty_analyzer import (
    ResultDirectoryMapper,
    ConformalPredictor,
    UncertaintyQuantifier,
    analyze_single_result
)


class UncertaintyBatchProcessor:
    """Batch process uncertainty analysis across all experiments and models."""

    def __init__(self, results_root: Path = None):
        """
        Initialize batch processor.

        Args:
            results_root: Root directory containing result folders
        """
        if results_root is None:
            results_root = config.RESULTS_DIR
        self.results_root = results_root
        self.mapper = ResultDirectoryMapper(results_root)

        # Mapping from pred_len to horizon name
        self.pred_len_to_horizon = {
            3: '30s',
            30: '300s',
            60: '600s'
        }

    def process_all(self, output_dir: Path = None) -> Dict[str, Path]:
        """
        Process all available results and generate uncertainty metrics.

        Args:
            output_dir: Output directory for CSVs (default: config.OUTPUT_DIR / 'uncertainty')

        Returns:
            Dictionary mapping metric names to output file paths
        """
        if output_dir is None:
            output_dir = config.OUTPUT_DIR / 'uncertainty'

        output_dir.mkdir(parents=True, exist_ok=True)

        print("="*80)
        print("UNCERTAINTY BATCH PROCESSING")
        print("="*80)
        print()

        # Get all available combinations
        combinations = self.mapper.get_available_combinations()
        print(f"Found {len(combinations)} result directories to process")
        print()
        if not combinations:
            print(f"No result directories under {self.results_root}; nothing written.")
            return {}

        # Process each combination
        results_list = []
        errors = []

        for idx, (experiment, model, pred_len) in enumerate(combinations, 1):
            # Convert pred_len to horizon name
            horizon_name = self.pred_len_to_horizon.get(pred_len)
            if horizon_name is None:
                continue  # Skip non-standard prediction lengths

            result_dir = self.mapper.get_result_dir(experiment, model, pred_len)
            if result_dir is None:
                continue

            try:
                # Progress indicator every 20 items
                if idx % 20 == 0 or idx == 1:
                    print(f"  Processing {idx}/{len(combinations)}: {experiment} {model} {horizon_name}")

                # Analyze
                metrics = analyze_single_result(result_dir, experiment, model, horizon_name)
                results_list.append(metrics)

            except Exception as e:
                errors.append({
                    'experiment': experiment,
                    'model': model,
                    'horizon': horizon_name,
                    'error': str(e)
                })
                print(f"    WARNING: Failed to process {experiment}/{model}/{horizon_name}: {e}")

        print()
        print(f"Successfully processed: {len(results_list)}")
        print(f"Errors: {len(errors)}")
        print()

        # Convert to DataFrame
        df = pd.DataFrame(results_list)

        # Save outputs
        output_files = {}

        # 1. Main uncertainty metrics
        main_output = output_dir / 'uncertainty_metrics.csv'
        df.to_csv(main_output, index=False)
        output_files['uncertainty_metrics'] = main_output
        print(f"Saved: {main_output.name}")

        # 2. Conformal intervals summary
        conf_cols = ['experiment', 'model', 'horizon',
                     'conf_90_radius', 'conf_95_radius', 'conf_99_radius',
                     'coverage_90_empirical', 'coverage_95_empirical']
        conformal_df = df[conf_cols].copy()
        conformal_output = output_dir / 'conformal_intervals.csv'
        conformal_df.to_csv(conformal_output, index=False)
        output_files['conformal_intervals'] = conformal_output
        print(f"Saved: {conformal_output.name}")

        # 3. Error distribution summary
        dist_cols = ['experiment', 'model', 'horizon',
                     'mae', 'median_ae', 'q90_ae', 'q95_ae', 'q99_ae',
                     'mean_residual', 'std_residual', 'skewness', 'kurtosis']
        dist_df = df[dist_cols].copy()
        dist_output = output_dir / 'error_distributions.csv'
        dist_df.to_csv(dist_output, index=False)
        output_files['error_distributions'] = dist_output
        print(f"Saved: {dist_output.name}")

        # 4. Coverage validation summary
        coverage_cols = ['experiment', 'model', 'horizon',
                         'coverage_90_predicted', 'coverage_90_empirical',
                         'coverage_95_predicted', 'coverage_95_empirical']
        coverage_df = df[coverage_cols].copy()
        coverage_df['coverage_90_gap'] = coverage_df['coverage_90_empirical'] - coverage_df['coverage_90_predicted']
        coverage_df['coverage_95_gap'] = coverage_df['coverage_95_empirical'] - coverage_df['coverage_95_predicted']
        coverage_output = output_dir / 'coverage_validation.csv'
        coverage_df.to_csv(coverage_output, index=False)
        output_files['coverage_validation'] = coverage_output
        print(f"Saved: {coverage_output.name}")

        # 5. Heteroscedasticity summary
        hetero_cols = ['experiment', 'model', 'horizon',
                       'hetero_spearman_corr', 'hetero_pvalue', 'hetero_var_ratio']
        hetero_df = df[hetero_cols].copy()
        hetero_output = output_dir / 'heteroscedasticity.csv'
        hetero_df.to_csv(hetero_output, index=False)
        output_files['heteroscedasticity'] = hetero_output
        print(f"Saved: {hetero_output.name}")

        # 6. Bootstrap CI summary
        bootstrap_cols = ['experiment', 'model', 'horizon',
                          'mae', 'mae_ci_lower', 'mae_ci_upper', 'mae_ci_width']
        bootstrap_df = df[bootstrap_cols].copy()
        bootstrap_output = output_dir / 'bootstrap_confidence_intervals.csv'
        bootstrap_df.to_csv(bootstrap_output, index=False)
        output_files['bootstrap_cis'] = bootstrap_output
        print(f"Saved: {bootstrap_output.name}")

        # 7. Error log (if any errors occurred)
        if errors:
            error_df = pd.DataFrame(errors)
            error_output = output_dir / 'processing_errors.csv'
            error_df.to_csv(error_output, index=False)
            output_files['errors'] = error_output
            print(f"Saved: {error_output.name} ({len(errors)} errors)")

        print()
        print("="*80)
        print("UNCERTAINTY BATCH PROCESSING COMPLETE")
        print("="*80)

        return output_files

    def generate_summary_report(self, df: pd.DataFrame, output_path: Path):
        """
        Generate human-readable summary report.

        Args:
            df: DataFrame with uncertainty metrics
            output_path: Path to save report
        """
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("UNCERTAINTY QUANTIFICATION SUMMARY REPORT\n")
            f.write("="*80 + "\n\n")

            # Overall statistics
            f.write("OVERALL STATISTICS\n")
            f.write("-"*80 + "\n\n")

            f.write(f"Total models analyzed: {len(df)}\n")
            f.write(f"Experiments: {', '.join(sorted(df['experiment'].unique()))}\n")
            f.write(f"Horizons: {', '.join(sorted(df['horizon'].unique()))}\n")
            f.write(f"Models: {len(df['model'].unique())}\n\n")

            # Conformal interval statistics by horizon
            f.write("CONFORMAL INTERVALS (90% Confidence)\n")
            f.write("-"*80 + "\n\n")

            for horizon in config.HORIZON_ORDER:
                horizon_data = df[df['horizon'] == horizon]
                if len(horizon_data) == 0:
                    continue

                f.write(f"{horizon}:\n")
                f.write(f"  Mean radius: {horizon_data['conf_90_radius'].mean():.3f}\n")
                f.write(f"  Std radius: {horizon_data['conf_90_radius'].std():.3f}\n")
                f.write(f"  Min/Max radius: {horizon_data['conf_90_radius'].min():.3f} / "
                       f"{horizon_data['conf_90_radius'].max():.3f}\n\n")

            # Coverage validation
            f.write("COVERAGE VALIDATION\n")
            f.write("-"*80 + "\n\n")

            f.write("90% Confidence Intervals:\n")
            mean_coverage_90 = df['coverage_90_empirical'].mean()
            f.write(f"  Target coverage: 90.0%\n")
            f.write(f"  Empirical coverage: {mean_coverage_90*100:.1f}%\n")
            f.write(f"  Gap: {(mean_coverage_90 - 0.90)*100:.1f}%\n\n")

            f.write("95% Confidence Intervals:\n")
            mean_coverage_95 = df['coverage_95_empirical'].mean()
            f.write(f"  Target coverage: 95.0%\n")
            f.write(f"  Empirical coverage: {mean_coverage_95*100:.1f}%\n")
            f.write(f"  Gap: {(mean_coverage_95 - 0.95)*100:.1f}%\n\n")

            # Best models by uncertainty
            f.write("MODELS WITH TIGHTEST UNCERTAINTY (90% conf, 300s horizon)\n")
            f.write("-"*80 + "\n\n")

            df_300s = df[df['horizon'] == '300s'].copy()
            if len(df_300s) > 0:
                top_models = df_300s.nsmallest(10, 'conf_90_radius')
                for idx, row in top_models.iterrows():
                    f.write(f"  {row['model']} ({row['experiment']}): ± {row['conf_90_radius']:.3f}\n")
                f.write("\n")

            # Error distribution characteristics
            f.write("ERROR DISTRIBUTION CHARACTERISTICS\n")
            f.write("-"*80 + "\n\n")

            f.write(f"Mean bias (all models): {df['mean_residual'].mean():.4f}\n")
            f.write(f"Mean skewness: {df['skewness'].mean():.4f}\n")
            f.write(f"Mean kurtosis: {df['kurtosis'].mean():.4f}\n\n")

            # Heteroscedasticity
            f.write("HETEROSCEDASTICITY ANALYSIS\n")
            f.write("-"*80 + "\n\n")

            significant_hetero = df[df['hetero_pvalue'] < 0.05]
            f.write(f"Models with significant heteroscedasticity (p < 0.05): "
                   f"{len(significant_hetero)} / {len(df)} ({len(significant_hetero)/len(df)*100:.1f}%)\n\n")

            f.write("="*80 + "\n")

        print(f"Saved summary report: {output_path.name}")


if __name__ == '__main__':
    print("Running uncertainty batch processor...\n")

    processor = UncertaintyBatchProcessor()
    output_files = processor.process_all()
    if not output_files:
        raise SystemExit(1)

    # Generate summary report
    df = pd.read_csv(output_files['uncertainty_metrics'])
    report_path = config.OUTPUT_DIR / 'uncertainty' / 'uncertainty_summary_report.txt'
    processor.generate_summary_report(df, report_path)

    print("\nDone!")
