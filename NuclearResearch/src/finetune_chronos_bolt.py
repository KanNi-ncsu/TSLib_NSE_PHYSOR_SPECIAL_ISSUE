"""
Chronos (Bolt) Univariate Fine-Tuning on Nuclear Reactor Data (Exp4)

Fine-tunes amazon/chronos-bolt-base on 395 training scenarios using a custom
PyTorch training loop. Only TPCT (univariate) is used -- Chronos Bolt does not
support multivariate fine-tuning.

Uses RAW (unscaled) data -- foundation models have internal normalization.
Reads scenario CSVs directly via split_metadata_lhs.json.

ChronosBoltModelForForecasting has a FIXED prediction_length=64.
Shorter pred_lens are handled by only using the first pred_len steps of output.
We still fine-tune with target padded to 64 steps (zero-padded).

Usage:
    python NuclearResearch/src/finetune_chronos_bolt.py --pred_len 30

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
import torch
from torch.utils.data import Dataset, DataLoader

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# ChronosBolt fixed prediction length
CHRONOS_BOLT_PRED_LEN = 64


class TPCTSlidingWindowDataset(Dataset):
    """
    Sliding window dataset over TPCT values from multiple scenarios.

    Each sample is a (context, target) pair:
        - context: (seq_len,) tensor of TPCT values
        - target: (CHRONOS_BOLT_PRED_LEN,) tensor, zero-padded if pred_len < 64

    Never crosses scenario boundaries.
    """

    def __init__(self, scenario_series, seq_len, pred_len):
        """
        Parameters
        ----------
        scenario_series : list of 1D numpy arrays
            TPCT values per scenario
        seq_len : int
            Input context length
        pred_len : int
            Actual prediction length (will be padded to 64 for model)
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.windows = []

        for series in scenario_series:
            n = len(series)
            # Need at least seq_len + pred_len timesteps
            if n < seq_len + pred_len:
                continue

            max_start = n - seq_len - pred_len
            for start in range(max_start + 1):
                context = series[start:start + seq_len]
                target = series[start + seq_len:start + seq_len + pred_len]
                self.windows.append((
                    torch.tensor(context, dtype=torch.float32),
                    torch.tensor(target, dtype=torch.float32),
                ))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        context, target = self.windows[idx]

        # Pad target to CHRONOS_BOLT_PRED_LEN (64) if needed
        if self.pred_len < CHRONOS_BOLT_PRED_LEN:
            padded = torch.zeros(CHRONOS_BOLT_PRED_LEN, dtype=torch.float32)
            padded[:self.pred_len] = target
            target_mask = torch.zeros(CHRONOS_BOLT_PRED_LEN, dtype=torch.bool)
            target_mask[:self.pred_len] = True
            return context, padded, target_mask
        else:
            target_mask = torch.ones(CHRONOS_BOLT_PRED_LEN, dtype=torch.bool)
            return context, target[:CHRONOS_BOLT_PRED_LEN], target_mask


def load_tpct_series(metadata, scenario_ids, raw_data_root, target="TPCT"):
    """
    Load TPCT values from scenario CSVs as list of 1D arrays.

    Parameters
    ----------
    metadata : dict
        Split metadata
    scenario_ids : list
        Scenario IDs to load
    raw_data_root : str
        Directory containing CSV files
    target : str
        Target column name

    Returns
    -------
    list of 1D numpy arrays
    """
    series_list = []
    for sid in scenario_ids:
        csv_path = os.path.join(raw_data_root, f"{sid}.csv")
        if not os.path.exists(csv_path):
            print(f"WARNING: Missing {csv_path}, skipping")
            continue
        df = pd.read_csv(csv_path)
        if target not in df.columns:
            print(f"WARNING: '{target}' not in scenario {sid}, skipping")
            continue
        series_list.append(df[target].values.astype(np.float32))

    print(f"Loaded {len(series_list)} scenarios, "
          f"timesteps per scenario: {len(series_list[0])}")
    return series_list


def train_one_epoch(model, dataloader, optimizer, device):
    """Run one training epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for context, target, target_mask in dataloader:
        context = context.to(device)
        target = target.to(device)
        target_mask = target_mask.to(device)

        output = model(context=context, target=target, target_mask=target_mask)
        loss = output.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def validate(model, dataloader, device):
    """Run validation. Returns average loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for context, target, target_mask in dataloader:
            context = context.to(device)
            target = target.to(device)
            target_mask = target_mask.to(device)

            output = model(context=context, target=target, target_mask=target_mask)
            total_loss += output.loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Chronos Bolt on Nuclear Exp4 TPCT Data"
    )

    # Data
    parser.add_argument("--dataset_dir", type=str,
                        default=str(project_root / "dataset" / "normal_LOCAC_RW_LOF_SGATR_train_LOCA_test"),
                        help="Path to exp4 dataset directory")
    parser.add_argument("--target", type=str, default="TPCT",
                        help="Target feature name")

    # Model
    parser.add_argument("--model_name", type=str, default="amazon/chronos-bolt-base",
                        help="Pretrained model name or path")
    parser.add_argument("--pred_len", type=int, default=30,
                        help="Prediction length (3, 30, or 60)")
    parser.add_argument("--seq_len", type=int, default=30,
                        help="Input context length")

    # Training
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=40,
                        help="Maximum training epochs (~40 epochs to match Chronos-2 8000 steps)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience")
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="Freeze encoder layers for stability")

    # Output
    parser.add_argument("--output_base", type=str,
                        default=str(project_root / "checkpoints_foundation" / "finetuned"),
                        help="Base directory for checkpoints")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU device ID")

    args = parser.parse_args()

    print("=" * 80)
    print("Chronos Bolt Univariate Fine-Tuning - Nuclear Exp4 TPCT")
    print("=" * 80)
    print(f"  Model: {args.model_name}")
    print(f"  pred_len: {args.pred_len} (model fixed at {CHRONOS_BOLT_PRED_LEN})")
    print(f"  seq_len: {args.seq_len}")
    print(f"  learning_rate: {args.learning_rate}")
    print(f"  num_epochs: {args.num_epochs}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  patience: {args.patience}")
    print(f"  freeze_encoder: {args.freeze_encoder}")
    print(f"  target: {args.target}")
    print(f"  GPU: {args.gpu}")
    print()

    # Set GPU
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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

    # Load TPCT series
    print("\nLoading training TPCT series...")
    train_series = load_tpct_series(
        metadata, metadata["train_scenarios"], raw_data_root, target=args.target
    )

    print("Loading validation TPCT series...")
    val_series = load_tpct_series(
        metadata, metadata["val_scenarios"], raw_data_root, target=args.target
    )

    # Create datasets
    print("\nBuilding sliding window datasets...")
    train_dataset = TPCTSlidingWindowDataset(train_series, args.seq_len, args.pred_len)
    val_dataset = TPCTSlidingWindowDataset(val_series, args.seq_len, args.pred_len)
    print(f"  Train windows: {len(train_dataset)}")
    print(f"  Val windows: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # Load model
    print(f"\nLoading pretrained model: {args.model_name}")
    from chronos.chronos_bolt import ChronosBoltModelForForecasting

    model = ChronosBoltModelForForecasting.from_pretrained(args.model_name)
    model = model.to(device)
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optionally freeze encoder
    if args.freeze_encoder:
        frozen_count = 0
        for name, param in model.named_parameters():
            if "output_patch_embedding" not in name and "head" not in name:
                param.requires_grad = False
                frozen_count += 1
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Froze {frozen_count} parameter groups. Trainable params: {trainable:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    # Output directory
    output_dir = os.path.join(
        args.output_base,
        f"chronos_bolt_exp4_pl{args.pred_len}"
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Training loop with early stopping
    print(f"\nStarting training for up to {args.num_epochs} epochs...")
    best_val_loss = float("inf")
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, args.num_epochs + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)

        epoch_time = time.time() - epoch_start

        print(f"  Epoch {epoch:3d}/{args.num_epochs} | "
              f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
              f"time={epoch_time:.1f}s")

        # Check improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save best model
            model.save_pretrained(output_dir)
            print(f"    -> Saved best model (val_loss={val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs)")
                break

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print(f"\nTraining complete in {minutes}m {seconds}s")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Checkpoint saved to: {output_dir}/")

    # Quick validation inference
    print("\nRunning quick validation inference...")
    model.eval()
    with torch.no_grad():
        sample_context = train_dataset[0][0].unsqueeze(0).to(device)
        output = model(context=sample_context)
        preds = output.quantile_preds  # (1, num_quantiles, 64)
        median_idx = preds.shape[1] // 2
        median_pred = preds[0, median_idx, :args.pred_len]
        print(f"  Input shape: {sample_context.shape}")
        print(f"  Output quantile shape: {preds.shape}")
        print(f"  Median prediction (first 5): {median_pred[:5].cpu().numpy()}")

    print("\n" + "=" * 80)
    print("Fine-tuning complete!")
    print(f"  Checkpoint: {output_dir}/")
    print(f"  Best val_loss: {best_val_loss:.6f}")
    print(f"  Elapsed: {minutes}m {seconds}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
