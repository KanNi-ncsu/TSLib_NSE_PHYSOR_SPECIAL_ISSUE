"""
Data Loader for Standalone ARIMA Baseline

Loads nuclear accident data matching TSLib preprocessing:
- StandardScaler normalization (fit on train, transform test)
- Feature exclusion
- Target column positioning (MS mode: TPCT last)
- Per-scenario CSV loading

Author: Nuclear Research Team
Date: 2025
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Derive project root from this file's location
# This file is at: <PROJECT_ROOT>/NuclearResearch/src/baselines/arima_data_loader.py
_BASELINES_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_BASELINES_DIR, '..', '..', '..'))
_DEFAULT_ROOT_PATH = os.path.join(_PROJECT_ROOT, 'dataset')
_DEFAULT_EXCLUDED_PATH = os.path.join(_PROJECT_ROOT, 'NuclearResearch', 'config', 'excluded_features.txt')


class NuclearAccidentDataLoader:
    """
    Loads nuclear accident data for ARIMA forecasting

    Matches TSLib preprocessing exactly:
    - Loads data from individual scenario CSV files
    - Applies StandardScaler (fit on train, transform on test)
    - Handles feature exclusion
    - Supports MS mode (multivariate → univariate)

    Parameters:
    -----------
    root_path : str or None
        Path to dataset directory. Defaults to <PROJECT_ROOT>/dataset/

    data_path : str
        Dataset subdirectory name (e.g., 'normal_LOCAC_train_LOCA_test')

    target : str
        Target feature name (e.g., 'TPCT')

    scale : bool, default=True
        Whether to apply StandardScaler

    excluded_features_path : str or None
        Path to feature exclusion list. Defaults to <PROJECT_ROOT>/NuclearResearch/config/excluded_features.txt
    """

    def __init__(self, root_path=None,
                 data_path='normal_LOCAC_train_LOCA_test',
                 target='TPCT',
                 scale=True,
                 excluded_features_path=None):

        self.root_path = root_path or _DEFAULT_ROOT_PATH
        self.data_path = data_path
        self.target = target
        self.scale = scale
        self.excluded_features_path = excluded_features_path or _DEFAULT_EXCLUDED_PATH

        # Paths
        self.full_data_path = os.path.join(self.root_path, data_path)
        self.metadata_path = os.path.join(self.full_data_path, 'split_metadata_lhs.json')

        # Scaler
        self.scaler = StandardScaler() if scale else None

        # Load metadata
        self._load_metadata()

        # Load excluded features
        self._load_excluded_features()

    def _load_metadata(self):
        """Load split metadata containing train/val/test scenario IDs"""
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)

        self.train_scenarios = self.metadata['train_scenarios']
        self.val_scenarios = self.metadata['val_scenarios']
        self.test_scenarios = self.metadata['test_scenarios']

        print(f"Dataset: {self.metadata['experiment_name']}")
        print(f"  Train: {len(self.train_scenarios)} scenarios")
        print(f"  Val: {len(self.val_scenarios)} scenarios")
        print(f"  Test: {len(self.test_scenarios)} scenarios")

    def _load_excluded_features(self):
        """Load feature exclusion list"""
        self.excluded_features = []

        if os.path.exists(self.excluded_features_path):
            with open(self.excluded_features_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        self.excluded_features.append(line)

            print(f"Loaded {len(self.excluded_features)} excluded features")
        else:
            print(f"No exclusion file found at {self.excluded_features_path}")

    def _load_scenario(self, scenario_id):
        """Load single scenario CSV"""
        csv_path = os.path.join(self.full_data_path, f'{scenario_id}.csv')

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Scenario file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        return df

    def _prepare_column_order(self, sample_df):
        """
        Prepare column order matching TSLib preprocessing

        MS mode: Put target LAST (required by TSLib)
        Remove TIME column and excluded features
        """
        cols = sample_df.columns.tolist()

        if self.target not in cols:
            raise ValueError(f"Target '{self.target}' not found. Available: {cols}")

        # Remove TIME column
        feature_cols = [c for c in cols if c != 'TIME']

        # Remove excluded features
        feature_cols = [c for c in feature_cols if c not in self.excluded_features]

        # MS mode: Put target last
        other_cols = [c for c in feature_cols if c != self.target]
        column_order = other_cols + [self.target]

        return column_order

    def fit_scaler(self):
        """
        Fit scaler on ALL training data

        This matches TSLib preprocessing where scaler is fit on
        all training scenarios combined
        """
        if not self.scale:
            print("Scaling disabled, skipping scaler fitting")
            return

        print(f"Fitting scaler on {len(self.train_scenarios)} training scenarios...")

        # Load first scenario to determine column order
        first_df = self._load_scenario(self.train_scenarios[0])
        self.column_order = self._prepare_column_order(first_df)

        # Collect all training data
        all_train_data = []
        for sid in self.train_scenarios:
            df = self._load_scenario(sid)
            data = df[self.column_order].values
            all_train_data.append(data)

        all_train_data = np.vstack(all_train_data)

        # Fit scaler
        self.scaler.fit(all_train_data)

        print(f"Scaler fitted on {all_train_data.shape[0]} samples, {all_train_data.shape[1]} features")
        print(f"Using {len(self.column_order)} features (after exclusions)")
        print(f"Features: {self.column_order[:5]}...{self.column_order[-1]}")

    def load_test_data(self, seq_len=30, pred_len=30):
        """
        Load test data for ARIMA forecasting

        Returns test data as list of scenarios, where each scenario is:
        - Full time series (for sliding window extraction)
        - Shape: [timesteps, features]

        Parameters:
        -----------
        seq_len : int
            Sequence length (history window)

        pred_len : int
            Prediction length (forecast horizon)

        Returns:
        --------
        test_data : list of dict
            Each entry contains:
            - 'scenario_id': Scenario ID
            - 'data': Scaled data [timesteps, features]
            - 'target_idx': Index of target column (TPCT)
            - 'raw_data': Unscaled data for inverse transform
        """
        if self.scaler is None and self.scale:
            raise RuntimeError("Scaler not fitted. Call fit_scaler() first.")

        if not hasattr(self, 'column_order'):
            # Load first scenario to determine column order
            first_df = self._load_scenario(self.train_scenarios[0])
            self.column_order = self._prepare_column_order(first_df)

        print(f"\nLoading {len(self.test_scenarios)} test scenarios...")

        test_data = []

        for sid in self.test_scenarios:
            # Load scenario
            df = self._load_scenario(sid)

            # Extract features in correct order
            raw_data = df[self.column_order].values

            # Apply scaling
            if self.scale:
                scaled_data = self.scaler.transform(raw_data)
            else:
                scaled_data = raw_data.copy()

            # Target is LAST column (MS mode)
            target_idx = len(self.column_order) - 1

            test_data.append({
                'scenario_id': sid,
                'data': scaled_data,
                'target_idx': target_idx,
                'raw_data': raw_data,
                'length': len(scaled_data)
            })

        print(f"Loaded {len(test_data)} test scenarios")
        print(f"  Each scenario shape: {test_data[0]['data'].shape}")
        print(f"  Target column '{self.target}' at index {target_idx}")

        return test_data

    def inverse_transform(self, scaled_data):
        """
        Inverse transform scaled predictions back to original scale

        Parameters:
        -----------
        scaled_data : np.ndarray
            Scaled predictions

        Returns:
        --------
        original_data : np.ndarray
            Unscaled predictions
        """
        if self.scale and self.scaler is not None:
            return self.scaler.inverse_transform(scaled_data)
        return scaled_data


# Quick usage example
if __name__ == "__main__":
    print("="*80)
    print("Nuclear Accident Data Loader - Quick Test")
    print("="*80)
    print()

    # Test exp1
    print("Testing Experiment 1 (normal_LOCAC_train_LOCA_test)")
    print("-" * 80)

    loader = NuclearAccidentDataLoader(
        data_path='normal_LOCAC_train_LOCA_test',
        target='TPCT',
        scale=True
    )

    # Fit scaler on training data
    loader.fit_scaler()

    # Load test data
    test_data = loader.load_test_data(seq_len=30, pred_len=30)

    print()
    print(f"Test data loaded: {len(test_data)} scenarios")
    print(f"First scenario ID: {test_data[0]['scenario_id']}")
    print(f"First scenario shape: {test_data[0]['data'].shape}")
    print(f"Target index: {test_data[0]['target_idx']}")
    print()

    # Test inverse transform
    sample_scaled = test_data[0]['data'][:10]  # First 10 timesteps
    sample_original = loader.inverse_transform(sample_scaled)

    print(f"Scaled data range: [{sample_scaled.min():.3f}, {sample_scaled.max():.3f}]")
    print(f"Original data range: [{sample_original.min():.2f}, {sample_original.max():.2f}]")

    print()
    print("="*80)
    print("Data Loader Test Complete")
    print("="*80)
