#!/usr/bin/env python3
"""
Naive (persistence) and MovingAverage baselines for LOCA TPCT forecasting.

For each test window, Naive repeats the last observed TPCT value and
MovingAverage repeats the mean TPCT of the input window (seq_len steps).

Usage:
    python NuclearResearch/src/baselines/run_naive_baselines.py
    python NuclearResearch/src/baselines/run_naive_baselines.py --exp exp0 --pred_len 30

Output: results/baseline_LOCA_TPCT_<exp_tag>_<Model>_pl<H>_standalone/{pred,true}.npy
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

EXPERIMENTS = {
    'exp0': ('exp0_loca_only', 'LOCA_only_train_LOCA_test'),
    'exp1': ('exp1_loca_normal', 'LOCA_normal_train_LOCA_test'),
    'exp2': ('exp2_locac_normal', 'normal_LOCAC_train_LOCA_test'),
    'exp3': ('exp3_flb_normal', 'LOCA_FLB_normal_train_LOCA_test'),
    'exp4': ('exp4_multi_accident', 'normal_LOCAC_RW_LOF_SGATR_train_LOCA_test'),
}
PRED_LENS = [3, 30, 60]


def load_test_series(dataset_dir, target):
    """Load the target series of the test scenarios, in split-metadata order."""
    with open(os.path.join(dataset_dir, 'split_metadata_lhs.json')) as f:
        metadata = json.load(f)
    csv_dir = metadata.get('raw_data_location', '.')
    if not os.path.isabs(csv_dir):
        csv_dir = os.path.join(dataset_dir, csv_dir)
    if not os.path.isdir(csv_dir):
        csv_dir = dataset_dir
    return [pd.read_csv(os.path.join(csv_dir, f'{sid}.csv'))[target].values.astype(np.float64)
            for sid in metadata['test_scenarios']]


def make_forecasts(series_list, seq_len, pred_len):
    """Return Naive and MovingAverage forecasts and targets, each (n_windows, pred_len, 1)."""
    naive, moving_average, true = [], [], []
    for y in series_list:
        for start in range(len(y) - seq_len - pred_len + 1):
            history = y[start:start + seq_len]
            naive.append(np.full(pred_len, history[-1]))
            moving_average.append(np.full(pred_len, history.mean()))
            true.append(y[start + seq_len:start + seq_len + pred_len])
    as_array = lambda a: np.asarray(a, dtype=np.float64)[:, :, None]
    return {'Naive': as_array(naive), 'MovingAverage': as_array(moving_average)}, as_array(true)


def main():
    parser = argparse.ArgumentParser(description='Naive and MovingAverage baselines')
    parser.add_argument('--exp', nargs='+', default=list(EXPERIMENTS), choices=list(EXPERIMENTS))
    parser.add_argument('--pred_len', nargs='+', type=int, default=PRED_LENS)
    parser.add_argument('--seq_len', type=int, default=30)
    parser.add_argument('--target', type=str, default='TPCT')
    parser.add_argument('--root_path', type=str, default=str(PROJECT_ROOT / 'dataset'))
    parser.add_argument('--results_dir', type=str, default=str(PROJECT_ROOT / 'results'))
    args = parser.parse_args()

    for exp in args.exp:
        exp_tag, data_path = EXPERIMENTS[exp]
        series = load_test_series(os.path.join(args.root_path, data_path), args.target)
        for pred_len in args.pred_len:
            forecasts, true = make_forecasts(series, args.seq_len, pred_len)
            for model, pred in forecasts.items():
                out_dir = Path(args.results_dir) / f'baseline_LOCA_TPCT_{exp_tag}_{model}_pl{pred_len}_standalone'
                out_dir.mkdir(parents=True, exist_ok=True)
                np.save(out_dir / 'pred.npy', pred)
                np.save(out_dir / 'true.npy', true)
                mae = np.mean(np.abs(pred - true))
                print(f'{exp} {model:13s} pl{pred_len:<3d} windows={pred.shape[0]} MAE(all steps)={mae:.2f} -> {out_dir}')


if __name__ == '__main__':
    main()
