#!/usr/bin/env python3
"""
Unified Visualization Script for All Experiments (exp0-exp4)

Runs visualizations for all models across all 3 horizons (30s, 300s, 600s ahead).
Generates plots and metrics_summary.txt files for each result folder.

Usage:
    python visualize_experiment.py --exp exp0        # Run exp0 only
    python visualize_experiment.py --exp all         # Run all experiments
    python visualize_experiment.py --exp exp1 exp2   # Run specific experiments
"""

import argparse
import re
import subprocess
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).parent
RESULT_PROCESS_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(RESULT_PROCESS_DIR))

# Derive project root from script location
# Script is at: <PROJECT_ROOT>/NuclearResearch/src/result_process/visualization/
PROJECT_ROOT = Path(__file__).resolve().parents[4]

# ============================================================================
# CONFIGURATION
# ============================================================================

EXPERIMENTS = {
    'exp0': {
        'name': 'LOCA-only',
        'pattern': 'exp0_loca_only',
        'description': '75 LOCA training'
    },
    'exp1': {
        'name': 'LOCA + Normal',
        'pattern': 'exp1_loca_normal',
        'description': '75 LOCA + 20 Normal training'
    },
    'exp2': {
        'name': 'LOCAC + Normal',
        'pattern': 'exp2_locac_normal',
        'description': '75 LOCAC + 20 Normal training'
    },
    'exp3': {
        'name': 'FLB + Normal',
        'pattern': 'exp3_flb_normal',
        'description': '75 FLB + 20 Normal training'
    },
    'exp4': {
        'name': 'Multi-Accident',
        'pattern': 'exp4_multi_accident',
        'description': '75x5 accidents + 20 Normal'
    }
}

# Specialist (trained) deep learning models
MODELS = [
    'ARIMA', 'Autoformer', 'Crossformer', 'DLinear', 'ETSformer', 'FEDformer',
    'FreTS', 'GRU', 'Informer', 'LSTM', 'MICN',
    'MovingAverage', 'Naive', 'Nonstationary', 'PatchTST', 'Pyraformer',
    'Reformer', 'SCINet', 'SegRNN', 'TFT', 'TSMixer',
    'TimeMixer', 'TimesNet', 'Transformer', 'iTransformer'
]

# Foundation models (zero-shot, no training)
FOUNDATION_MODELS = ['Chronos', 'Chronos2', 'TimesFM']

# Fine-tuned foundation models
FINETUNED_MODELS = ['Chronos2_FT', 'Chronos_FT']

HORIZONS = [
    {'name': '30s', 'pl': 3, 'script': 'visualize_30s_ahead_flexible.py'},
    {'name': '300s', 'pl': 30, 'script': 'visualize_300s_ahead_flexible.py'},
    {'name': '600s', 'pl': 60, 'script': 'visualize_600s_ahead_flexible.py'}
]

RESULTS_ROOT = PROJECT_ROOT / 'results'
CENTRALIZED_OUTPUT_ROOT = PROJECT_ROOT / 'NuclearResearch' / 'results' / 'Visualization'
SEQ_LEN = 30  # Standard sequence length

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def find_result_folder(exp_id: str, exp_pattern: str, model: str, pred_len: int) -> Path:
    """
    Find result folder for given experiment, model, and prediction length.

    Tries specialist pattern first, then foundation model pattern.
    Seeded rerun folders (suffix '_seed<N>') are skipped; standalone baseline
    folders (suffix '_standalone') take precedence.

    Args:
        exp_id: Experiment ID (e.g., 'exp0')
        exp_pattern: Experiment pattern (e.g., 'exp0_loca_only')
        model: Model name (e.g., 'Transformer' or 'Chronos')
        pred_len: Prediction length (3, 30, or 60)

    Returns:
        Path to result folder, or None if not found
    """
    patterns = [
        f"*{exp_pattern}_{model}_pl{pred_len}_*",        # specialist (trained) models
        f"*{exp_id}_foundation_{model}_pl{pred_len}_*",   # zero-shot foundation models
        f"*{exp_id}_finetuned_{model}_pl{pred_len}_*",    # fine-tuned foundation models
    ]
    for pattern in patterns:
        matches = sorted(m for m in RESULTS_ROOT.glob(pattern)
                         if not re.search(r'_seed\d+$', m.name))
        if matches:
            standalone = [m for m in matches if m.name.endswith('_standalone')]
            return (standalone or matches)[0]

    return None


def run_visualization(result_folder: Path, viz_script: str, seq_len: int, pred_len: int,
                     output_dir: Path = None, exp_name: str = "", model_name: str = "") -> Tuple[bool, str]:
    """
    Run visualization script for a single result folder.

    Args:
        result_folder: Path to result folder
        viz_script: Name of visualization script
        seq_len: Sequence length
        pred_len: Prediction length
        output_dir: Custom output directory (centralized location)
        exp_name: Experiment name for plot titles
        model_name: Model name for plot titles and file naming

    Returns:
        (success, message) tuple
    """
    script_path = RESULT_PROCESS_DIR / viz_script

    if not script_path.exists():
        return False, f"Script not found: {script_path}"

    # Set environment variable for matplotlib backend
    env = os.environ.copy()
    env['MPLBACKEND'] = 'Agg'

    cmd = [
        'python', str(script_path),
        '--result_folder', str(result_folder),
        '--seq_len', str(seq_len),
        '--pred_len', str(pred_len)
    ]

    # Add custom output directory if specified
    if output_dir:
        cmd.extend(['--output_dir', str(output_dir)])

    # Add experiment and model names for proper labeling
    if exp_name:
        cmd.extend(['--exp_name', exp_name])
    if model_name:
        cmd.extend(['--model_name', model_name])
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            # Extract Average MAE from output
            for line in result.stdout.split('\n'):
                if 'Average MAE' in line:
                    return True, line.strip()
            return True, "Visualization completed"
        else:
            return False, f"Error: {result.stderr[:200]}"
            
    except subprocess.TimeoutExpired:
        return False, "Timeout (>5 minutes)"
    except Exception as e:
        return False, f"Exception: {str(e)[:200]}"


# ============================================================================
# MAIN LOGIC
# ============================================================================

def visualize_single_experiment(exp_id: str, verbose: bool = True, models: List[str] = None) -> Dict:
    """
    Run visualizations for all models and horizons in a single experiment.

    Args:
        exp_id: Experiment ID (e.g., 'exp0')
        verbose: Print detailed progress
        models: Optional list of model names to restrict to (default: all)

    Returns:
        Dictionary with results summary
    """
    exp_config = EXPERIMENTS[exp_id]

    # Optionally restrict to a subset of models (preserves list order)
    def _filter(lst):
        return [m for m in lst if m in models] if models else lst
    specialist = _filter(MODELS)
    foundation = _filter(FOUNDATION_MODELS)
    finetuned = _filter(FINETUNED_MODELS)
    all_models = specialist + foundation + finetuned

    print("=" * 80)
    print(f"EXPERIMENT: {exp_id.upper()} - {exp_config['name']}")
    print(f"Description: {exp_config['description']}")
    print(f"Models: {len(specialist)} specialist + {len(foundation)} foundation + {len(finetuned)} fine-tuned")
    if models:
        print(f"Restricted to: {', '.join(all_models) if all_models else '(none matched)'}")
    print("=" * 80)
    print()

    total_runs = len(all_models) * len(HORIZONS)
    current_run = 0
    results = {'success': 0, 'skipped': 0, 'failed': 0}

    for section_name, model_list in [('Specialist Models', specialist), ('Foundation Models', foundation), ('Fine-Tuned Foundation', finetuned)]:
        if verbose:
            print(f"\n{'=' * 80}")
            print(f"  {section_name}")
            print(f"{'=' * 80}")

        for model in model_list:
            if verbose:
                print(f"\n{'_' * 80}")
                print(f"MODEL: {model}")
                print(f"{'_' * 80}")

            for horizon in HORIZONS:
                current_run += 1

                # Find result folder
                result_folder = find_result_folder(exp_id, exp_config['pattern'], model, horizon['pl'])

                if result_folder is None:
                    if verbose:
                        print(f"  [{current_run:3d}/{total_runs}] {horizon['name']:5s} - SKIP: Folder not found")
                    results['skipped'] += 1
                    continue

                # Construct centralized output directory
                # Structure: {CENTRALIZED_OUTPUT_ROOT}/{exp_id}/{horizon_name}/
                centralized_output = CENTRALIZED_OUTPUT_ROOT / exp_id / f"{horizon['name']}_ahead"

                # Run visualization
                if verbose:
                    print(f"  [{current_run:3d}/{total_runs}] {horizon['name']:5s} - Running...", end='', flush=True)

                success, message = run_visualization(
                    result_folder,
                    horizon['script'],
                    SEQ_LEN,
                    horizon['pl'],
                    output_dir=centralized_output,
                    exp_name=f"{exp_id.upper()}: {exp_config['name']}",
                    model_name=model
                )

                if success:
                    if verbose:
                        print(f" OK - {message}")
                    results['success'] += 1
                else:
                    if verbose:
                        print(f" FAIL - {message}")
                    results['failed'] += 1

    # Summary
    print()
    print("=" * 80)
    print(f"{exp_id.upper()} COMPLETE")
    print("=" * 80)
    print(f"  Success: {results['success']:3d}/{total_runs}")
    print(f"  Skipped: {results['skipped']:3d}/{total_runs}")
    print(f"  Failed:  {results['failed']:3d}/{total_runs}")
    print("=" * 80)
    print()

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Run visualizations for LOCA forecasting experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_experiment.py --exp exp0           # Run exp0 only
  python visualize_experiment.py --exp all            # Run all experiments
  python visualize_experiment.py --exp exp1 exp2 exp3 # Run multiple experiments
  python visualize_experiment.py --exp exp0 --quiet   # Minimal output
        """
    )
    
    parser.add_argument(
        '--exp',
        nargs='+',
        required=True,
        choices=list(EXPERIMENTS.keys()) + ['all'],
        help='Experiment(s) to visualize'
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        default=None,
        help='Restrict to specific model name(s), e.g. --models SCINet Crossformer (default: all)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Minimal output (only show summaries)'
    )
    
    args = parser.parse_args()
    
    # Determine which experiments to run
    if 'all' in args.exp:
        experiments = list(EXPERIMENTS.keys())
    else:
        experiments = args.exp
    
    # Run visualizations
    verbose = not args.quiet
    all_results = {}
    
    for exp_id in experiments:
        results = visualize_single_experiment(exp_id, verbose=verbose, models=args.models)
        all_results[exp_id] = results
    
    # Final summary
    if len(experiments) > 1:
        print("\n" + "=" * 80)
        print("FINAL SUMMARY - ALL EXPERIMENTS")
        print("=" * 80)
        for exp_id, results in all_results.items():
            total = sum(results.values())
            print(f"{exp_id:5s}: {results['success']:2d} success, {results['skipped']:2d} skipped, {results['failed']:2d} failed (total: {total})")
        print("=" * 80)


if __name__ == '__main__':
    main()
