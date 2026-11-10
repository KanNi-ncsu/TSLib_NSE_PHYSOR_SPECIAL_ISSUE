"""
Data Loader for exp0-4 Analysis Suite

Loads and validates the CSV file containing experimental results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import config


class DataLoader:
    """Load and validate experimental results CSV."""

    def __init__(self, csv_path: Path = None):
        """
        Initialize data loader.

        Args:
            csv_path: Path to CSV file (default: from config)
        """
        self.csv_path = csv_path or config.INPUT_CSV
        self.df = None

    def load(self) -> pd.DataFrame:
        """
        Load CSV file and perform validation.

        Returns:
            DataFrame with columns: experiment, model, horizon, mae, rmse
        """
        print(f"Loading data from: {self.csv_path}")

        # Load CSV
        self.df = pd.read_csv(self.csv_path)

        # Validate structure
        self._validate_columns()
        self._validate_data_types()
        self._validate_values()

        # Add derived columns
        self._add_derived_columns()

        print(f"OK Loaded {len(self.df)} rows")
        print(f"OK Experiments: {sorted(self.df['experiment'].unique())}")
        print(f"OK Models: {sorted(self.df['model'].unique())}")
        print(f"OK Horizons: {sorted(self.df['horizon'].unique())}")

        return self.df

    def _validate_columns(self):
        """Validate required columns exist."""
        required = ['experiment', 'model', 'horizon', 'mae', 'rmse']
        missing = [col for col in required if col not in self.df.columns]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        print(f"OK All required columns present: {required}")

    def _validate_data_types(self):
        """Validate and convert data types."""
        # Convert to appropriate types
        self.df['experiment'] = self.df['experiment'].astype(str)
        self.df['model'] = self.df['model'].astype(str)
        self.df['horizon'] = self.df['horizon'].astype(str)
        self.df['mae'] = pd.to_numeric(self.df['mae'], errors='coerce')
        self.df['rmse'] = pd.to_numeric(self.df['rmse'], errors='coerce')

        # Check for NaNs after conversion
        if self.df[['mae', 'rmse']].isna().any().any():
            n_missing = self.df[['mae', 'rmse']].isna().sum().sum()
            print(f"WARNING: {n_missing} missing metric values found")

        print("OK Data types validated")

    def _validate_values(self):
        """Validate data values are reasonable."""
        # Check for negative errors
        if (self.df['mae'] < 0).any() or (self.df['rmse'] < 0).any():
            raise ValueError("Negative error values found!")

        # Check MAE <= RMSE (should always be true)
        invalid = self.df['mae'] > self.df['rmse']
        if invalid.any():
            n_invalid = invalid.sum()
            print(f"WARNING: {n_invalid} rows have MAE > RMSE (unusual but possible)")

        # Check for unreasonably large errors (>1000K for PCT)
        large_mae = self.df['mae'] > 1000
        if large_mae.any():
            n_large = large_mae.sum()
            models = self.df[large_mae]['model'].unique()
            print(f"WARNING: {n_large} rows with MAE > 1000K")
            print(f"   Models: {list(models)}")

        print("OK Data values validated")

    def _add_derived_columns(self):
        """Add derived columns for analysis."""
        # Architecture family
        self.df['family'] = self.df['model'].apply(config.get_family)

        # MAE/RMSE ratio (error distribution indicator)
        self.df['mae_rmse_ratio'] = self.df['mae'] / self.df['rmse']

        # Normalized MAE (approximate, using typical PCT range)
        self.df['normalized_mae'] = self.df['mae'] / config.PCT_MEAN

        print("OK Added derived columns: family, mae_rmse_ratio, normalized_mae")

    def get_experiment_data(self, experiment: str) -> pd.DataFrame:
        """Get data for specific experiment."""
        return self.df[self.df['experiment'] == experiment].copy()

    def get_horizon_data(self, horizon: str) -> pd.DataFrame:
        """Get data for specific horizon."""
        return self.df[self.df['horizon'] == horizon].copy()

    def get_model_data(self, model: str) -> pd.DataFrame:
        """Get data for specific model."""
        return self.df[self.df['model'] == model].copy()

    def get_family_data(self, family: str) -> pd.DataFrame:
        """Get data for specific architecture family."""
        return self.df[self.df['family'] == family].copy()

    def get_summary_stats(self) -> Dict:
        """Get summary statistics of the dataset."""
        return {
            'n_total': len(self.df),
            'n_experiments': self.df['experiment'].nunique(),
            'n_models': self.df['model'].nunique(),
            'n_horizons': self.df['horizon'].nunique(),
            'experiments': sorted(self.df['experiment'].unique()),
            'models': sorted(self.df['model'].unique()),
            'horizons': sorted(self.df['horizon'].unique()),
            'mae_range': (self.df['mae'].min(), self.df['mae'].max()),
            'rmse_range': (self.df['rmse'].min(), self.df['rmse'].max()),
            'best_overall': {
                'model': self.df.loc[self.df['mae'].idxmin(), 'model'],
                'mae': self.df['mae'].min(),
                'experiment': self.df.loc[self.df['mae'].idxmin(), 'experiment'],
                'horizon': self.df.loc[self.df['mae'].idxmin(), 'horizon']
            }
        }

    def print_summary(self):
        """Print dataset summary."""
        stats = self.get_summary_stats()

        print("\n" + "="*80)
        print("DATASET SUMMARY")
        print("="*80)
        print(f"Total rows: {stats['n_total']}")
        print(f"Experiments: {stats['n_experiments']} - {stats['experiments']}")
        print(f"Models: {stats['n_models']}")
        print(f"Horizons: {stats['n_horizons']} - {stats['horizons']}")
        print(f"\nMAE range: [{stats['mae_range'][0]:.2f}, {stats['mae_range'][1]:.2f}] K")
        print(f"RMSE range: [{stats['rmse_range'][0]:.2f}, {stats['rmse_range'][1]:.2f}] K")
        print(f"\nBest overall performance:")
        print(f"  Model: {stats['best_overall']['model']}")
        print(f"  MAE: {stats['best_overall']['mae']:.2f} K")
        print(f"  Experiment: {stats['best_overall']['experiment']}")
        print(f"  Horizon: {stats['best_overall']['horizon']}")
        print("="*80 + "\n")


if __name__ == '__main__':
    # Test data loader
    print("="*80)
    print("Data Loader Test")
    print("="*80)
    print()

    loader = DataLoader()
    df = loader.load()
    loader.print_summary()

    # Test filtering functions
    print("Testing filter functions...")
    exp1_data = loader.get_experiment_data('exp1')
    print(f"OK exp1 data: {len(exp1_data)} rows")

    h300_data = loader.get_horizon_data('300s')
    print(f"OK 300s horizon data: {len(h300_data)} rows")

    transformer_data = loader.get_model_data('Transformer')
    print(f"OK Transformer data: {len(transformer_data)} rows")

    trans_family_data = loader.get_family_data('Transformers')
    print(f"OK Transformers family data: {len(trans_family_data)} rows")

    print("\nOK Data loader test successful!")
