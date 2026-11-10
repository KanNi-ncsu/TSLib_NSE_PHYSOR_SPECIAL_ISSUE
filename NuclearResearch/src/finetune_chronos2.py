"""
Chronos-2 Multivariate Fine-Tuning on Nuclear Reactor Data (Exp4)

Fine-tunes amazon/chronos-2 on 395 training scenarios using the built-in
Chronos2Pipeline.fit() API with multivariate covariates (73 covariates + TPCT target).

Uses RAW (unscaled) data -- foundation models have internal normalization.
Reads scenario CSVs directly via split_metadata_lhs.json, filtering excluded features.

Usage:
    python NuclearResearch/src/finetune_chronos2.py --pred_len 30

Author: Nuclear Research Team
Date: 2026-03-25
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


def load_excluded_features():
    """Load the list of excluded features from config."""
    exclusion_path = os.path.join(
        project_root, "NuclearResearch", "config", "excluded_features.txt"
    )
    if not os.path.exists(exclusion_path):
        raise FileNotFoundError(f"Feature exclusion list not found: {exclusion_path}")
    excluded = []
    with open(exclusion_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                excluded.append(line)
    print(f"Loaded {len(excluded)} excluded features")
    return excluded


def load_scenarios(metadata, scenario_ids, raw_data_root, excluded_features, target="TPCT"):
    """
    Load scenarios as list of dicts for Chronos2Pipeline.fit().

    Each dict has:
        - "target": 1D numpy array of TPCT values (shape: (timesteps,))
        - "past_covariates": dict mapping covariate name -> 1D numpy array

    Uses RAW (unscaled) data -- Chronos-2 handles normalization internally.

    Parameters
    ----------
    metadata : dict
        Split metadata from JSON
    scenario_ids : list
        List of scenario IDs to load
    raw_data_root : str
        Path to directory containing scenario CSV files
    excluded_features : list
        Features to exclude
    target : str
        Target column name

    Returns
    -------
    list of dicts in Chronos-2 covariate format
    """
    inputs = []
    skipped = 0

    for sid in scenario_ids:
        csv_path = os.path.join(raw_data_root, f"{sid}.csv")
        if not os.path.exists(csv_path):
            print(f"WARNING: Scenario file not found: {csv_path}, skipping")
            skipped += 1
            continue

        df = pd.read_csv(csv_path)

        # Get feature columns (exclude TIME and excluded features)
        all_cols = [c for c in df.columns if c != "TIME"]
        feature_cols = [c for c in all_cols if c not in excluded_features]

        if target not in feature_cols:
            print(f"WARNING: Target '{target}' not found in scenario {sid}, skipping")
            skipped += 1
            continue

        # Extract target as 1D array
        target_values = df[target].values.astype(np.float32)

        # Extract covariates (all features except target)
        covariate_cols = [c for c in feature_cols if c != target]
        past_covariates = {}
        for col in covariate_cols:
            past_covariates[col] = df[col].values.astype(np.float32)

        inputs.append({
            "target": target_values,
            "past_covariates": past_covariates,
        })

    if skipped > 0:
        print(f"WARNING: Skipped {skipped} scenarios")

    print(f"Loaded {len(inputs)} scenarios")
    print(f"  Target: {target}")
    print(f"  Covariates per scenario: {len(inputs[0]['past_covariates'])}")
    print(f"  Timesteps per scenario: {len(inputs[0]['target'])}")

    return inputs


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Chronos-2 on Nuclear Exp4 Data")

    # Data configuration
    parser.add_argument("--dataset_dir", type=str,
                        default=str(project_root / "dataset" / "normal_LOCAC_RW_LOF_SGATR_train_LOCA_test"),
                        help="Path to exp4 dataset directory")
    parser.add_argument("--target", type=str, default="TPCT",
                        help="Target feature name")

    # Model configuration
    parser.add_argument("--model_name", type=str, default="amazon/chronos-2",
                        help="Pretrained model name or path")
    parser.add_argument("--pred_len", type=int, default=30,
                        help="Prediction length (3, 30, or 60)")
    parser.add_argument("--context_length", type=int, default=30,
                        help="Context length for fine-tuning (matches seq_len)")

    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--num_steps", type=int, default=8000,
                        help="Number of fine-tuning steps (~40 epochs with 395 scenarios, batch_size=32)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size (default 32 for 395 scenarios)")

    # Output
    parser.add_argument("--output_base", type=str,
                        default=str(project_root / "checkpoints_foundation" / "finetuned"),
                        help="Base directory for saving fine-tuned checkpoints")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU device ID")

    args = parser.parse_args()

    print("=" * 80)
    print("Chronos-2 Multivariate Fine-Tuning - Nuclear Exp4")
    print("=" * 80)
    print(f"  Model: {args.model_name}")
    print(f"  pred_len: {args.pred_len}")
    print(f"  context_length: {args.context_length}")
    print(f"  learning_rate: {args.learning_rate}")
    print(f"  num_steps: {args.num_steps}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  target: {args.target}")
    print(f"  GPU: {args.gpu}")
    print()

    # Set GPU
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Load metadata
    metadata_path = os.path.join(args.dataset_dir, "split_metadata_lhs.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Scenario CSV location (defaults to the dataset folder)
    raw_data_root = metadata.get("raw_data_location", ".")
    if not os.path.isabs(raw_data_root):
        raw_data_root = os.path.join(args.dataset_dir, raw_data_root)
    if not os.path.isdir(raw_data_root):
        raw_data_root = args.dataset_dir

    print(f"Dataset: {args.dataset_dir}")
    print(f"Raw data: {raw_data_root}")
    print(f"Train scenarios: {len(metadata['train_scenarios'])}")
    print(f"Val scenarios: {len(metadata['val_scenarios'])}")
    print()

    # Load excluded features
    excluded_features = load_excluded_features()

    # Load training data
    print("Loading training scenarios...")
    train_inputs = load_scenarios(
        metadata, metadata["train_scenarios"], raw_data_root,
        excluded_features, target=args.target
    )

    # Load validation data
    print("\nLoading validation scenarios...")
    val_inputs = load_scenarios(
        metadata, metadata["val_scenarios"], raw_data_root,
        excluded_features, target=args.target
    )

    # Output directory
    output_dir = os.path.join(
        args.output_base,
        f"chronos2_multivar_exp4_pl{args.pred_len}"
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Load pretrained model
    print(f"\nLoading pretrained model: {args.model_name}")
    from chronos import Chronos2Pipeline

    pipeline = Chronos2Pipeline.from_pretrained(
        args.model_name,
        device_map="cuda",
    )
    print("Model loaded.")

    # Fine-tune
    print(f"\nStarting fine-tuning for {args.num_steps} steps...")
    start_time = time.time()

    # load_best_model_at_end=True is set automatically by Chronos fit() when
    # validation_inputs is provided, so the best checkpoint is preserved even
    # if later steps overfit. No early stopping callback needed.
    finetuned_pipeline = pipeline.fit(
        inputs=train_inputs,
        prediction_length=args.pred_len,
        validation_inputs=val_inputs,
        context_length=args.context_length,
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        output_dir=output_dir,
        save_total_limit=3,
    )

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print(f"\nFine-tuning complete in {minutes}m {seconds}s")
    print(f"Fine-tuned checkpoint saved to: {output_dir}/finetuned-ckpt/")

    # Quick validation: run inference on first val scenario
    print("\nRunning quick validation inference...")
    val_sample = val_inputs[0]
    target_len = len(val_sample["target"])

    # Use last context_length steps as input, check output shape
    sample_input = {
        "target": val_sample["target"][-args.context_length:],
        "past_covariates": {
            k: v[-args.context_length:]
            for k, v in val_sample["past_covariates"].items()
        },
    }

    predictions = finetuned_pipeline.predict(
        [sample_input],
        prediction_length=args.pred_len,
    )

    print(f"  Input target shape: ({args.context_length},)")
    print(f"  Output prediction shape: {predictions[0].shape}")
    print(f"  Prediction sample (first 5): {predictions[0][0, :5].numpy()}")

    print("\n" + "=" * 80)
    print("Fine-tuning complete!")
    print(f"  Checkpoint: {output_dir}/finetuned-ckpt/")
    print(f"  Elapsed: {minutes}m {seconds}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
