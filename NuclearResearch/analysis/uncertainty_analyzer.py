"""
Uncertainty Quantification and Conformal Prediction for exp0-4 Analysis Suite

Extracts prediction intervals and error distributions from raw .npy prediction files.
Implements split conformal prediction for distribution-free uncertainty quantification.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
from scipy import stats
import config


class ResultDirectoryMapper:
    """Maps CSV entries to result directories containing pred.npy/true.npy files."""

    def __init__(self, results_root: Path = None):
        """
        Initialize mapper.

        Args:
            results_root: Root directory containing all result folders
        """
        if results_root is None:
            results_root = config.RESULTS_DIR
        self.results_root = results_root
        self._cache = {}
        self._build_cache()

    def _build_cache(self):
        """Build cache of available result directories.

        Seeded rerun folders (suffix '_seed<N>') are skipped; standalone baseline
        folders (suffix '_standalone') take precedence.
        """
        print("Scanning for result directories...")

        for pred_file in sorted(self.results_root.glob('**/pred.npy')):
            result_dir = pred_file.parent
            dir_name = result_dir.name
            if re.search(r'_seed\d+$', dir_name):
                continue

            # Parse directory name to extract metadata
            # Format: long_term_forecast_LOCA_TPCT_exp*_*_MODEL_pl*_...
            metadata = self._parse_directory_name(dir_name)
            if metadata:
                key = (metadata['experiment'], metadata['model'], metadata['pred_len'])
                current = self._cache.get(key)
                if current is not None and current.name.endswith('_standalone'):
                    continue
                self._cache[key] = result_dir

        print(f"  Found {len(self._cache)} result directories")

    def _parse_directory_name(self, dir_name: str) -> Optional[Dict]:
        """Parse result directory name to extract metadata."""
        # Match experiment pattern: exp0, exp1, exp2, exp3, exp4
        exp_match = re.search(r'exp(\d)', dir_name)
        if not exp_match:
            return None

        experiment = f'exp{exp_match.group(1)}'

        # Try finetuned model pattern first (handles underscored names like Chronos_FT)
        ft_match = re.search(r'_finetuned_([A-Za-z0-9_]+?)_pl(\d+)_', dir_name)
        if ft_match:
            model = ft_match.group(1)
            pred_len = int(ft_match.group(2))
        else:
            # Extract pred_len (pl value) - use FIRST occurrence
            pl_match = re.search(r'_pl(\d+)_', dir_name)
            if not pl_match:
                return None

            pred_len = int(pl_match.group(1))

            # Extract model name (appears before _pl)
            # Pattern: ..._MODEL_pl*_MODEL_...
            # Example: exp1_loca_normal_SCINet_pl30_SCINet_nuclear...
            model_match = re.search(r'_(?!exp)([A-Za-z0-9]+)_pl\d+', dir_name)
            if not model_match:
                return None

            model = model_match.group(1)

        return {
            'experiment': experiment,
            'model': model,
            'pred_len': pred_len
        }

    def get_result_dir(self, experiment: str, model: str, pred_len: int) -> Optional[Path]:
        """
        Get result directory for given experiment/model/pred_len.

        Args:
            experiment: Experiment name (exp0, exp1, etc.)
            model: Model name
            pred_len: Prediction length

        Returns:
            Path to result directory or None if not found
        """
        return self._cache.get((experiment, model, pred_len))

    def get_available_combinations(self) -> List[Tuple[str, str, int]]:
        """Get list of all available (experiment, model, pred_len) combinations."""
        return list(self._cache.keys())


class ConformalPredictor:
    """
    Split Conformal Prediction for time series forecasting.

    Provides distribution-free prediction intervals with finite-sample validity.
    """

    def __init__(
        self,
        pred: np.ndarray,
        true: np.ndarray,
        calibration_ratio: float = 0.8,
        random_seed: int = 42
    ):
        """
        Initialize conformal predictor.

        Args:
            pred: Predictions array (n_samples, n_horizons, n_features)
            true: Ground truth array (same shape as pred)
            calibration_ratio: Fraction of data to use for calibration (rest for test)
            random_seed: Random seed for reproducible random split
        """
        self.pred = pred
        self.true = true
        self.n_samples, self.n_horizons, self.n_features = pred.shape

        # Compute nonconformity scores (absolute residuals)
        self.residuals = pred - true
        self.abs_residuals = np.abs(self.residuals)

        # Random split using Latin Hypercube Sampling approach
        # Generate random permutation of indices
        np.random.seed(random_seed)
        indices = np.random.permutation(self.n_samples)

        # Split indices
        self.n_cal = int(calibration_ratio * self.n_samples)
        self.n_test = self.n_samples - self.n_cal

        cal_indices = indices[:self.n_cal]
        test_indices = indices[self.n_cal:]

        # Split scores using random indices
        self.cal_scores = self.abs_residuals[cal_indices]
        self.test_scores = self.abs_residuals[test_indices]

    def get_conformal_interval(
        self,
        alpha: float = 0.1,
        horizon: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute conformal prediction interval radius.

        Args:
            alpha: Miscoverage level (0.1 for 90% coverage)
            horizon: Specific horizon index (None for global across all horizons)

        Returns:
            Interval radius (scalar if horizon specified, array of length n_horizons otherwise)
        """
        if horizon is not None:
            # Per-horizon interval
            scores = self.cal_scores[:, horizon, 0]
        else:
            # Global interval across all horizons
            scores = self.cal_scores[:, :, 0].flatten()

        # Conformal quantile with finite-sample correction
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        q_level = min(q_level, 1.0)  # Cap at 1.0

        radius = np.quantile(scores, q_level)
        return radius

    def get_all_intervals(
        self,
        alpha_levels: List[float] = [0.01, 0.05, 0.10, 0.20]
    ) -> pd.DataFrame:
        """
        Compute conformal intervals for multiple confidence levels and all horizons.

        Args:
            alpha_levels: List of miscoverage levels

        Returns:
            DataFrame with columns: horizon, alpha, confidence_level, radius
        """
        results = []

        for alpha in alpha_levels:
            confidence = (1 - alpha) * 100

            # Global interval
            global_radius = self.get_conformal_interval(alpha=alpha, horizon=None)
            results.append({
                'horizon': 'all',
                'horizon_index': -1,
                'alpha': alpha,
                'confidence_level': confidence,
                'radius': global_radius
            })

            # Per-horizon intervals
            for h in range(self.n_horizons):
                radius = self.get_conformal_interval(alpha=alpha, horizon=h)
                results.append({
                    'horizon': f'h{h+1}',
                    'horizon_index': h,
                    'alpha': alpha,
                    'confidence_level': confidence,
                    'radius': radius
                })

        return pd.DataFrame(results)

    def validate_coverage(
        self,
        alpha: float = 0.1,
        horizon: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Validate empirical coverage on test set.

        Args:
            alpha: Miscoverage level
            horizon: Specific horizon (None for global)

        Returns:
            Dictionary with predicted_coverage, empirical_coverage, n_test
        """
        radius = self.get_conformal_interval(alpha=alpha, horizon=horizon)

        if horizon is not None:
            test_scores = self.test_scores[:, horizon, 0]
        else:
            test_scores = self.test_scores[:, :, 0].flatten()

        # Check coverage
        covered = test_scores <= radius
        empirical_coverage = covered.mean()
        predicted_coverage = 1 - alpha

        return {
            'predicted_coverage': predicted_coverage,
            'empirical_coverage': empirical_coverage,
            'n_test': len(test_scores),
            'coverage_gap': empirical_coverage - predicted_coverage
        }


class UncertaintyQuantifier:
    """
    Comprehensive uncertainty quantification from prediction residuals.

    Computes error distributions, quantiles, bootstrap CIs, and diagnostics.
    """

    def __init__(self, pred: np.ndarray, true: np.ndarray):
        """
        Initialize quantifier.

        Args:
            pred: Predictions array (n_samples, n_horizons, n_features)
            true: Ground truth array (same shape)
        """
        self.pred = pred
        self.true = true
        self.n_samples, self.n_horizons, self.n_features = pred.shape

        # Compute errors
        self.residuals = pred - true
        self.abs_errors = np.abs(self.residuals)
        self.squared_errors = self.residuals ** 2
        self.percentage_errors = np.abs(self.residuals / (true + 1e-8)) * 100

    def get_error_distribution_stats(
        self,
        horizon: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Compute error distribution statistics.

        Args:
            horizon: Specific horizon (None for all horizons combined)

        Returns:
            Dictionary with distribution statistics
        """
        if horizon is not None:
            residuals = self.residuals[:, horizon, 0]
            abs_errors = self.abs_errors[:, horizon, 0]
        else:
            residuals = self.residuals[:, :, 0].flatten()
            abs_errors = self.abs_errors[:, :, 0].flatten()

        return {
            'mean_residual': residuals.mean(),
            'std_residual': residuals.std(),
            'skewness': stats.skew(residuals),
            'kurtosis': stats.kurtosis(residuals),
            'mae': abs_errors.mean(),
            'median_ae': np.median(abs_errors),
            'q25_ae': np.quantile(abs_errors, 0.25),
            'q75_ae': np.quantile(abs_errors, 0.75),
            'iqr_ae': np.quantile(abs_errors, 0.75) - np.quantile(abs_errors, 0.25),
            'q90_ae': np.quantile(abs_errors, 0.90),
            'q95_ae': np.quantile(abs_errors, 0.95),
            'q99_ae': np.quantile(abs_errors, 0.99),
            'max_ae': abs_errors.max()
        }

    def get_quantile_intervals(
        self,
        quantiles: List[float] = [0.50, 0.90, 0.95, 0.99]
    ) -> pd.DataFrame:
        """
        Compute quantile-based prediction intervals for all horizons.

        Args:
            quantiles: List of quantile levels

        Returns:
            DataFrame with quantile intervals per horizon
        """
        results = []

        for h in range(self.n_horizons):
            horizon_errors = self.abs_errors[:, h, 0]

            row = {
                'horizon': f'h{h+1}',
                'horizon_index': h
            }

            for q in quantiles:
                row[f'q{int(q*100)}'] = np.quantile(horizon_errors, q)

            results.append(row)

        return pd.DataFrame(results)

    def bootstrap_ci_for_metric(
        self,
        metric: str = 'mae',
        horizon: Optional[int] = None,
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000,
        random_seed: int = 42
    ) -> Dict[str, float]:
        """
        Compute bootstrap confidence interval for an error metric.

        Args:
            metric: Metric to compute ('mae', 'rmse', 'median_ae')
            horizon: Specific horizon (None for all)
            confidence_level: Confidence level for CI
            n_bootstrap: Number of bootstrap samples
            random_seed: Random seed for reproducibility

        Returns:
            Dictionary with point_estimate, ci_lower, ci_upper
        """
        np.random.seed(random_seed)

        if horizon is not None:
            errors = self.abs_errors[:, horizon, 0]
        else:
            errors = self.abs_errors[:, :, 0].flatten()

        # Compute point estimate
        if metric == 'mae':
            point_estimate = errors.mean()
        elif metric == 'rmse':
            point_estimate = np.sqrt((errors ** 2).mean())
        elif metric == 'median_ae':
            point_estimate = np.median(errors)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        # Bootstrap
        bootstrap_estimates = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(errors, size=len(errors), replace=True)
            if metric == 'mae':
                bootstrap_estimates.append(sample.mean())
            elif metric == 'rmse':
                bootstrap_estimates.append(np.sqrt((sample ** 2).mean()))
            elif metric == 'median_ae':
                bootstrap_estimates.append(np.median(sample))

        bootstrap_estimates = np.array(bootstrap_estimates)

        # Compute CI
        alpha = 1 - confidence_level
        ci_lower = np.quantile(bootstrap_estimates, alpha / 2)
        ci_upper = np.quantile(bootstrap_estimates, 1 - alpha / 2)

        return {
            'point_estimate': point_estimate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_width': ci_upper - ci_lower
        }

    def detect_heteroscedasticity(self, horizon: int = 0) -> Dict[str, float]:
        """
        Test for heteroscedasticity (non-constant error variance).

        Args:
            horizon: Horizon to test

        Returns:
            Dictionary with test statistics
        """
        true_vals = self.true[:, horizon, 0]
        abs_errors = self.abs_errors[:, horizon, 0]

        # Spearman correlation between true values and absolute errors
        corr, p_value = stats.spearmanr(true_vals, abs_errors)

        # Split into quantiles and compare variances
        q33, q67 = np.quantile(true_vals, [0.33, 0.67])
        low_mask = true_vals <= q33
        mid_mask = (true_vals > q33) & (true_vals <= q67)
        high_mask = true_vals > q67

        var_low = abs_errors[low_mask].var()
        var_mid = abs_errors[mid_mask].var()
        var_high = abs_errors[high_mask].var()

        return {
            'spearman_corr': corr,
            'spearman_pvalue': p_value,
            'var_low_tercile': var_low,
            'var_mid_tercile': var_mid,
            'var_high_tercile': var_high,
            'var_ratio_high_low': var_high / (var_low + 1e-8)
        }


def analyze_single_result(
    result_dir: Path,
    experiment: str,
    model: str,
    horizon_name: str
) -> Dict:
    """
    Perform complete uncertainty analysis for a single result directory.

    Args:
        result_dir: Path to result directory containing pred.npy and true.npy
        experiment: Experiment name
        model: Model name
        horizon_name: Horizon name (30s, 300s, 600s)

    Returns:
        Dictionary with all uncertainty metrics
    """
    # Load predictions
    pred = np.load(result_dir / 'pred.npy')
    true = np.load(result_dir / 'true.npy')

    # Initialize analyzers
    conformal = ConformalPredictor(pred, true)
    quantifier = UncertaintyQuantifier(pred, true)

    # Get distribution stats
    dist_stats = quantifier.get_error_distribution_stats()

    # Get conformal intervals for different confidence levels
    conf_90 = conformal.get_conformal_interval(alpha=0.10)
    conf_95 = conformal.get_conformal_interval(alpha=0.05)
    conf_99 = conformal.get_conformal_interval(alpha=0.01)

    # Validate coverage
    coverage_90 = conformal.validate_coverage(alpha=0.10)
    coverage_95 = conformal.validate_coverage(alpha=0.05)

    # Bootstrap CI for MAE
    bootstrap_ci = quantifier.bootstrap_ci_for_metric(metric='mae')

    # Heteroscedasticity test
    hetero = quantifier.detect_heteroscedasticity(horizon=0)

    # Compile results
    results = {
        'experiment': experiment,
        'model': model,
        'horizon': horizon_name,

        # Basic error statistics
        'mae': dist_stats['mae'],
        'median_ae': dist_stats['median_ae'],
        'q90_ae': dist_stats['q90_ae'],
        'q95_ae': dist_stats['q95_ae'],
        'q99_ae': dist_stats['q99_ae'],
        'max_ae': dist_stats['max_ae'],

        # Error distribution properties
        'mean_residual': dist_stats['mean_residual'],
        'std_residual': dist_stats['std_residual'],
        'skewness': dist_stats['skewness'],
        'kurtosis': dist_stats['kurtosis'],
        'iqr_ae': dist_stats['iqr_ae'],

        # Conformal intervals
        'conf_90_radius': conf_90,
        'conf_95_radius': conf_95,
        'conf_99_radius': conf_99,

        # Coverage validation
        'coverage_90_predicted': coverage_90['predicted_coverage'],
        'coverage_90_empirical': coverage_90['empirical_coverage'],
        'coverage_95_predicted': coverage_95['predicted_coverage'],
        'coverage_95_empirical': coverage_95['empirical_coverage'],

        # Bootstrap CI
        'mae_ci_lower': bootstrap_ci['ci_lower'],
        'mae_ci_upper': bootstrap_ci['ci_upper'],
        'mae_ci_width': bootstrap_ci['ci_width'],

        # Heteroscedasticity
        'hetero_spearman_corr': hetero['spearman_corr'],
        'hetero_pvalue': hetero['spearman_pvalue'],
        'hetero_var_ratio': hetero['var_ratio_high_low'],

        # Metadata
        'n_test_samples': pred.shape[0],
        'n_horizons': pred.shape[1]
    }

    return results


if __name__ == '__main__':
    print("="*80)
    print("Uncertainty Analyzer Test")
    print("="*80)
    print()

    # Test mapper
    mapper = ResultDirectoryMapper()
    print(f"Found {len(mapper.get_available_combinations())} result directories")
    print()

    # Test on a single result
    print("Testing on SCINet exp1 30-step prediction...")
    result_dir = mapper.get_result_dir('exp1', 'SCINet', 30)

    if result_dir:
        print(f"Result directory: {result_dir.name}")

        # Load and analyze
        pred = np.load(result_dir / 'pred.npy')
        true = np.load(result_dir / 'true.npy')

        print(f"Shape: {pred.shape}")

        # Conformal prediction
        conformal = ConformalPredictor(pred, true)
        intervals_df = conformal.get_all_intervals()
        print("\nConformal Intervals (first 5 horizons, 90% confidence):")
        print(intervals_df[intervals_df['alpha'] == 0.10].head().to_string(index=False))

        # Uncertainty quantification
        quantifier = UncertaintyQuantifier(pred, true)
        dist_stats = quantifier.get_error_distribution_stats(horizon=0)
        print("\nError Distribution (Horizon 1):")
        for key, value in list(dist_stats.items())[:8]:
            print(f"  {key}: {value:.4f}")

        print("\nOK Uncertainty analyzer test successful!")
    else:
        print("WARNING: Result directory not found")
