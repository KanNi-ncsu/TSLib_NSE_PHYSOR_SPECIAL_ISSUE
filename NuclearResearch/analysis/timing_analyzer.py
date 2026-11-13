"""
Training and inference timing analysis for nuclear LOCA forecasting models.

Loads two data sources:
- Training timing: timing_exp*_summary.csv (total wall-clock for train+test)
- Inference timing: inference_timing_exp*_summary.csv (per-sample forward pass benchmarks)

Optionally cross-references with MAE for efficiency (Pareto) analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import sys

sys.path.append(str(Path(__file__).parent))
import config


# Model name normalization: inference CSV names -> config names
_MODEL_NAME_MAP = {
    'TemporalFusionTransformer': 'TFT',
    'Nonstationary_Transformer': 'Nonstationary',
}


class TimingAnalyzer:
    """Analyze training and inference timing across models and experiments."""

    def __init__(self, metrics_df: pd.DataFrame = None):
        """
        Args:
            metrics_df: Extended metrics DataFrame from MetricsCalculator.
                        Enables efficiency analysis (time vs MAE) if provided.
        """
        self.metrics_df = metrics_df
        self.training_df = self._load_training_timing()
        self.inference_df = self._load_inference_timing()

    def _normalize_model_name(self, name: str) -> str:
        """Normalize model name to match config.MODEL_TO_FAMILY keys."""
        return _MODEL_NAME_MAP.get(name, name)

    def _load_training_timing(self) -> pd.DataFrame:
        """Load exp4 specialist timing + finetune timing for foundation models."""
        timing_dir = config.TIMING_DIR
        frames = []

        # Load specialist training times from exp4 only
        exp4_path = timing_dir / 'timing_exp4_summary.csv'
        if not exp4_path.exists():
            raise FileNotFoundError(f"timing_exp4_summary.csv not found in {timing_dir}")

        df = pd.read_csv(exp4_path)
        df = df[df['status'].str.startswith('OK')]
        df['experiment'] = 'exp4'
        df['model_type'] = 'specialist'
        df['model'] = df['model'].apply(self._normalize_model_name)
        df['family'] = df['model'].map(config.MODEL_TO_FAMILY).fillna('Unknown')
        frames.append(df[['experiment', 'model', 'family', 'model_type',
                          'horizon', 'pred_len', 'total_seconds']])

        # Load finetune times for Chronos/Chronos2
        ft_path = timing_dir / 'timing_exp4_finetune_summary.csv'
        if ft_path.exists():
            ft_df = pd.read_csv(ft_path)
            ft_df = ft_df[ft_df['status'].str.startswith('OK')]
            ft_df['model'] = ft_df['model'].map(config.FINETUNE_MODEL_MAP)
            ft_df = ft_df.dropna(subset=['model'])
            ft_df['horizon'] = ft_df['pred_len'].map(config.PRED_LEN_TO_HORIZON)
            ft_df['experiment'] = 'exp4'
            ft_df['model_type'] = 'foundation-ft'
            ft_df['family'] = ft_df['model'].map(config.MODEL_TO_FAMILY).fillna('Unknown')
            frames.append(ft_df[['experiment', 'model', 'family', 'model_type',
                                 'horizon', 'pred_len', 'total_seconds']])
            print(f"OK Loaded {len(ft_df)} finetune timing entries")

        result = pd.concat(frames, ignore_index=True)
        print(f"OK Loaded {len(result)} training timing entries (exp4 only)")
        return result

    def _load_inference_timing(self) -> pd.DataFrame:
        """Load inference_timing_exp*_summary.csv files if they exist."""
        timing_dir = config.TIMING_DIR
        frames = []

        for csv_path in sorted(timing_dir.glob('inference_timing_exp*_summary.csv')):
            filename = csv_path.name

            exp_match = re.search(r'exp(\d)', filename)
            if not exp_match:
                continue
            experiment = f'exp{exp_match.group(1)}'

            df = pd.read_csv(csv_path)
            df = df[df['status'] == 'OK']
            df['experiment'] = experiment
            df['model'] = df['model'].apply(self._normalize_model_name)
            df['family'] = df['model'].map(config.MODEL_TO_FAMILY).fillna('Unknown')

            frames.append(df)

        if not frames:
            print("  No inference_timing_exp*_summary.csv files found (optional)")
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        print(f"OK Loaded {len(result)} inference timing entries "
              f"from {len(frames)} files")
        return result

    def analyze(self) -> dict:
        """Run all timing analyses. Returns dict of name -> DataFrame."""
        results = {}
        results['training_summary'] = self._compute_training_summary()
        results['training_by_family'] = self._compute_family_training()

        if len(self.inference_df) > 0:
            results['inference_summary'] = self._compute_inference_summary()

        if self.metrics_df is not None and len(self.inference_df) > 0:
            results['efficiency_table'] = self._compute_efficiency()

        return results

    def _compute_training_summary(self) -> pd.DataFrame:
        """Per-model training time summary, averaged across experiments."""
        df = self.training_df

        # Pivot to get per-horizon columns
        rows = []
        for (model, family, model_type), group in df.groupby(
                ['model', 'family', 'model_type']):
            row = {
                'model': model,
                'family': family,
                'model_type': model_type,
            }

            for hz in config.HORIZON_ORDER:
                hz_data = group[group['horizon'] == hz]['total_seconds']
                if len(hz_data) > 0:
                    row[f'avg_time_{hz}'] = hz_data.mean()
                    row[f'std_time_{hz}'] = hz_data.std()
                else:
                    row[f'avg_time_{hz}'] = np.nan
                    row[f'std_time_{hz}'] = np.nan

            # Handle non-standard horizons (MultiPatchFormer)
            non_standard = group[~group['horizon'].isin(config.HORIZON_ORDER)]
            if len(non_standard) > 0:
                for _, ns_row in non_standard.iterrows():
                    hz = ns_row['horizon']
                    row[f'avg_time_{hz}'] = ns_row['total_seconds']

            all_times = group['total_seconds']
            row['avg_time_overall'] = all_times.mean()
            row['std_time_overall'] = all_times.std()
            row['min_time'] = all_times.min()
            row['max_time'] = all_times.max()
            row['n_experiments'] = group['experiment'].nunique()

            rows.append(row)

        result = pd.DataFrame(rows)
        result = result.sort_values('avg_time_overall', ascending=True)
        return result.reset_index(drop=True)

    def _compute_family_training(self) -> pd.DataFrame:
        """Per-family training time summary."""
        df = self.training_df

        rows = []
        for (family, model_type), group in df.groupby(['family', 'model_type']):
            # Average time per model first, then summarize
            model_avgs = group.groupby('model')['total_seconds'].mean()

            rows.append({
                'family': family,
                'model_type': model_type,
                'avg_time': model_avgs.mean(),
                'std_time': model_avgs.std(),
                'fastest_model': model_avgs.idxmin(),
                'fastest_time': model_avgs.min(),
                'slowest_model': model_avgs.idxmax(),
                'slowest_time': model_avgs.max(),
                'n_models': len(model_avgs),
            })

        result = pd.DataFrame(rows)
        return result.sort_values('avg_time', ascending=True).reset_index(drop=True)

    def _compute_inference_summary(self) -> pd.DataFrame:
        """Per-model inference time summary from benchmark data."""
        df = self.inference_df
        if len(df) == 0:
            return pd.DataFrame()

        cols = ['model', 'family', 'horizon', 'pred_len', 'batch_size',
                'single_fwd_mean_ms', 'single_fwd_std_ms',
                'n_params', 'speedup_vs_realtime', 'gpu_name']

        available_cols = [c for c in cols if c in df.columns]
        result = df[available_cols].copy()
        result = result.sort_values(['horizon', 'single_fwd_mean_ms'],
                                    ascending=[True, True])
        return result.reset_index(drop=True)

    def _compute_efficiency(self) -> pd.DataFrame:
        """Cross-reference inference time with Experiment 4 MAE for Pareto analysis."""
        if self.metrics_df is None or len(self.inference_df) == 0:
            return pd.DataFrame()

        inf_df = self.inference_df.copy()

        # Only standard horizons for MAE cross-reference
        standard_horizons = set(config.HORIZON_ORDER)
        inf_df = inf_df[inf_df['horizon'].isin(standard_horizons)]

        # Experiment 4 MAE per (model, horizon)
        exp4 = self.metrics_df[self.metrics_df['experiment'] == 'exp4']
        mae_exp4 = exp4.groupby(['model', 'horizon']).agg(
            mae=('mae', 'mean'),
            rmse=('rmse', 'mean'),
        ).reset_index()

        # Merge
        merged = inf_df.merge(mae_exp4, on=['model', 'horizon'], how='inner')

        if len(merged) == 0:
            print("  WARNING: No matching models between inference and metrics data")
            return pd.DataFrame()

        # Compute Pareto front per horizon
        merged['pareto_optimal'] = False
        for hz in merged['horizon'].unique():
            mask = merged['horizon'] == hz
            hz_df = merged[mask]
            pareto = self._compute_pareto_front(
                hz_df, 'single_fwd_mean_ms', 'mae')
            merged.loc[mask, 'pareto_optimal'] = pareto.values

        result_cols = ['model', 'family', 'horizon', 'n_params',
                       'single_fwd_mean_ms', 'single_fwd_std_ms',
                       'speedup_vs_realtime', 'mae', 'rmse',
                       'pareto_optimal']
        available = [c for c in result_cols if c in merged.columns]
        result = merged[available].sort_values(
            ['horizon', 'mae'], ascending=[True, True])
        return result.reset_index(drop=True)

    @staticmethod
    def _compute_pareto_front(df: pd.DataFrame, x_col: str,
                              y_col: str) -> pd.Series:
        """Return boolean Series marking Pareto-optimal rows (minimize both)."""
        is_pareto = pd.Series(True, index=df.index)
        x = df[x_col].values
        y = df[y_col].values
        idx = df.index.values

        for i in range(len(idx)):
            if not is_pareto.iloc[i]:
                continue
            for j in range(len(idx)):
                if i == j:
                    continue
                if (x[j] <= x[i] and y[j] <= y[i] and
                        (x[j] < x[i] or y[j] < y[i])):
                    is_pareto.iloc[i] = False
                    break

        return is_pareto

    def save_all(self, output_dir: Path = None):
        """Save all timing analysis CSVs."""
        if output_dir is None:
            output_dir = config.METRICS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        results = self.analyze()
        for name, df in results.items():
            path = output_dir / f'{name}.csv'
            df.to_csv(path, index=False)
            print(f"OK Saved: {name}.csv ({len(df)} rows)")


if __name__ == '__main__':
    print("=" * 80)
    print("Timing Analyzer Test")
    print("=" * 80)

    analyzer = TimingAnalyzer()
    results = analyzer.analyze()

    print("\n" + "=" * 80)
    print("Training Summary (top 10 fastest)")
    print("=" * 80)
    ts = results['training_summary']
    print(ts[['model', 'family', 'model_type', 'avg_time_overall']].head(10)
          .to_string(index=False))

    if 'inference_summary' in results:
        print("\n" + "=" * 80)
        print("Inference Summary (300s horizon, sorted by speed)")
        print("=" * 80)
        inf = results['inference_summary']
        inf_300 = inf[inf['horizon'] == '300s']
        print(inf_300[['model', 'family', 'single_fwd_mean_ms',
                        'speedup_vs_realtime', 'n_params']].to_string(index=False))

    config.create_output_directories()
    analyzer.save_all()
    print("\nOK Timing analyzer test successful!")
