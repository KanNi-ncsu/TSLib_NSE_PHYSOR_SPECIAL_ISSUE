# Multi-horizon LOCA peak cladding temperature forecasting

Code for the paper *A multi-horizon comparison of Transformer and foundation time-series models for post-LOCA peak cladding temperature forecasting* (Nuclear Science and Engineering).

The repository is built on the [Time Series Library (TSLib)](https://github.com/thuml/Time-Series-Library). It adds:

- a loader for the NPPAD nuclear accident simulations (`Dataset_Nuclear_Accident` in `data_provider/`),
- model extensions: GRU, LSTM, ARIMA, Moving Average and Naive, plus fine-tuned Chronos and Chronos-2 wrappers (`models/`; the statistical baselines reported in the paper are computed by `NuclearResearch/src/baselines/`),
- the experiment, evaluation and analysis pipeline for the paper (`NuclearResearch/`).

The TSLib benchmark scripts are unchanged. A few TSLib model files carry small compatibility edits (TimeMixer, TemporalFusionTransformer and the ETSformer layer); the TimesFM wrapper uses the `google/timesfm-2.0-500m-pytorch` checkpoint.

## Repository layout

```
data_provider/, exp/, layers/, models/, utils/, scripts/, run.py   TSLib (with the additions above)
NuclearResearch/
  config/excluded_features.txt         input features removed from the 96 NPPAD signals (74 remain)
  src/data_preprocessing/              build the five experiment datasets from NPPAD
  src/run_loca_forecast.py             train/test one TSLib model on a LOCA dataset
  src/run_loca_zeroshot.py             zero-shot and fine-tuned foundation-model evaluation
  src/finetune_chronos2.py, finetune_chronos_bolt.py
  src/baselines/                       ARIMA, Naive and Moving Average baselines
  src/run_inference_benchmark.py       inference latency benchmark
  src/result_process/                  rolling-window evaluation, metric extraction, prediction plots
  scripts/multi_horizon_final/         shell scripts that run the experiments
  scripts/                             data figures (parameter evolution, correlation heatmaps)
  analysis/                            cross-experiment analysis, conformal prediction, timing figures
  results_summary/                     metric and timing summaries reported in the paper
```

## Installation

Python 3.11 and a CUDA GPU are required. The reported runs used PyTorch 2.5.1 (CUDA 12.4) on NVIDIA L4 GPUs.

```bash
conda create -n loca-tslib python=3.11 -y
conda activate loca-tslib
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Chronos (`amazon/chronos-bolt-base`), Chronos-2 (`amazon/chronos-2`) and TimesFM (`google/timesfm-2.0-500m-pytorch`) weights are downloaded from the Hugging Face Hub on first use.

All commands below are run from the repository root.

## Data

The simulations come from the open NPPAD dataset (Qi et al., *Scientific Data* 9, 766, 2022; https://github.com/thu-inet/NuclearPowerPlantAccidentData). Copy the operation data of the six accident types used here:

```bash
git clone --depth 1 https://github.com/thu-inet/NuclearPowerPlantAccidentData.git /tmp/nppad
mkdir -p NuclearResearch/Raw_data/DONOTMODIFY/Operation_csv_data
for t in LOCA LOCAC FLB RW SGATR Normal; do
    cp -r /tmp/nppad/Operation_csv_data/$t NuclearResearch/Raw_data/DONOTMODIFY/Operation_csv_data/
done
```

Build the experiment datasets under `dataset/`:

```bash
python NuclearResearch/src/data_preprocessing/create_exp0_dataset.py
for e in 1 2 3 4; do python NuclearResearch/src/data_preprocessing/create_exp${e}_dataset.py; done
```

| Experiment | Dataset folder | Training scenarios |
|---|---|---|
| exp0 | `LOCA_only_train_LOCA_test` | 75 LOCA |
| exp1 | `LOCA_normal_train_LOCA_test` | 75 LOCA + 20 Normal |
| exp2 | `normal_LOCAC_train_LOCA_test` | 75 LOCAC + 20 Normal |
| exp3 | `LOCA_FLB_normal_train_LOCA_test` | 75 FLB + 20 Normal |
| exp4 | `normal_LOCAC_RW_LOF_SGATR_train_LOCA_test` | 75 each of LOCA, LOCAC, FLB, RW, SGATR + 20 Normal |

All experiments share the same 15 LOCA test scenarios (Latin hypercube split, seed 42): 1, 3, 5, 6, 14, 28, 33, 49, 53, 57, 68, 70, 73, 84, 100.

## Reproducing the experiments

Horizons are 3, 30 and 60 steps (30 s, 300 s and 600 s) with a 30-step (300 s) input window. Run outputs go to `checkpoints/`, `results/<setting>/{pred,true}.npy` and `NuclearResearch/logs/`.

**1. Specialist models** (one script per experiment, all horizons; about 9 to 52 GPU hours each):

```bash
bash NuclearResearch/scripts/multi_horizon_final/train_exp0_all_horizons.sh   # likewise exp1 ... exp4
```

**2. Statistical baselines:**

```bash
bash NuclearResearch/scripts/multi_horizon_final/train_arima_all_experiments.sh
python NuclearResearch/src/baselines/run_naive_baselines.py
```

**3. Foundation models** (zero-shot on all experiments; fine-tuning and evaluation on exp4):

```bash
bash NuclearResearch/scripts/multi_horizon_final/run_foundation_models_all_experiments.sh
bash NuclearResearch/scripts/multi_horizon_final/finetune_foundation_exp4.sh
bash NuclearResearch/scripts/multi_horizon_final/eval_finetuned_foundation_exp4.sh
```

**4. Seed sensitivity** (6 models x 5 seeds x 3 horizons on exp0 and exp4):

```bash
bash NuclearResearch/scripts/multi_horizon_final/train_exp0_seeded_rerun.sh
bash NuclearResearch/scripts/multi_horizon_final/train_exp4_seeded_rerun.sh
```

**5. Metrics.** For a horizon of H steps, the H-th step forecast of every test window forms a continuous H-step-ahead trajectory for each test scenario. MAE and RMSE are computed per scenario and averaged over the 15 test scenarios.

```bash
python NuclearResearch/src/result_process/visualization/visualize_experiment.py --exp all
python NuclearResearch/src/result_process/result_extraction/extract_metrics.py
```

This writes per-scenario plots and summaries to `NuclearResearch/results/Visualization/<exp>/<30s|300s|600s>_ahead/` and the metric table to `NuclearResearch/results_summary/all_experiments_metrics.csv`. For the seeded runs:

```bash
python NuclearResearch/src/result_process/visualization/visualize_seeded_rerun.py --exp exp0 exp4
python NuclearResearch/src/result_process/result_extraction/extract_seeded_metrics.py
python NuclearResearch/src/result_process/visualization/visualize_seeded_uncertainty.py --exp exp4 --model SCINet Crossformer --horizon 300s 600s
```

**6. Inference latency** (exp4 checkpoints):

```bash
bash NuclearResearch/scripts/multi_horizon_final/benchmark_inference_exp4.sh
bash NuclearResearch/scripts/multi_horizon_final/benchmark_inference_zeroshot_exp4.sh
```

**7. Uncertainty quantification and analysis figures:**

```bash
python NuclearResearch/analysis/uncertainty_batch_processor.py   # needs results/ from steps 1-3
python NuclearResearch/analysis/run_full_analysis.py
```

The first command computes split conformal intervals and residual statistics from `results/`. The second writes the figures and tables to `NuclearResearch/results_summary/analysis_output/`. With the summaries shipped in `NuclearResearch/results_summary/`, the second command can be run on its own without retraining.

**8. Data figures:**

```bash
python NuclearResearch/scripts/plot_loca_parameters_evolution.py
python NuclearResearch/scripts/generate_correlation_heatmap_reduced.py --mode normal
python NuclearResearch/scripts/generate_correlation_heatmap_reduced.py --mode loca_only
```

## Paper results

| Paper item | Produced by | Output |
|---|---|---|
| Result tables for Exp0-Exp4, foundation-model table | step 5 | `results_summary/all_experiments_metrics.csv` |
| Prediction plots (e.g. Exp0 SCINet 300 s) | step 5 | `results/Visualization/<exp>/<H>_ahead/<Model>_overview_<H>_ahead.png` |
| Seed sensitivity table | step 5 (seeded) | `results_summary/seeded_rerun_aggregated_metrics.csv` |
| Seed sensitivity plots | step 5 (seeded) | `results/Visualization_seeded/exp4/<H>_ahead/uncertainty_plots/` |
| Improvement histograms, top-10 and percentile plots | step 7 | `results_summary/analysis_output/figures/{comparisons,rankings}/` |
| Conformal intervals, error distribution statistics | step 7 | `results_summary/analysis_output/uncertainty/`, `results_summary/analysis_output/figures/uncertainty/` |
| Inference and training time, accuracy-efficiency plots | steps 6-7 | `results_summary/analysis_output/figures/timing/` |
| Parameter evolution and correlation heatmaps | step 8 | `results_summary/analysis_output/figures/data/` |

Paths in the last column are relative to `NuclearResearch/`; `<H>` is `30s`, `300s` or `600s`.

## Running on a Slurm cluster

Each experiment script can be submitted as one job, for example:

```bash
sbatch --gres=gpu:1 --wrap "bash NuclearResearch/scripts/multi_horizon_final/train_exp0_all_horizons.sh"
```

The scripts that launch many independent runs (ARIMA, fine-tuning, seeded reruns) accept a `LAUNCHER` prefix. `submit_slurm.sh` submits each run as its own job:

```bash
LAUNCHER="bash NuclearResearch/scripts/multi_horizon_final/submit_slurm.sh" SBATCH_OPTS="--gres=gpu:1" \
    bash NuclearResearch/scripts/multi_horizon_final/train_exp4_seeded_rerun.sh
```

## Acknowledgements and license

This work builds on the Time Series Library (Wu et al., 2024; https://github.com/thuml/Time-Series-Library) and uses the NPPAD dataset (Qi et al., *Scientific Data* 9, 766, 2022; https://doi.org/10.1038/s41597-022-01879-1). Please cite both if you use this code. The code is released under the MIT License (see `LICENSE`), which includes the original TSLib copyright notice.
