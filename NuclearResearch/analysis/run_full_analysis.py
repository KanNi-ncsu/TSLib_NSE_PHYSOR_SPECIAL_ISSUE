#!/usr/bin/env python
"""
Main Orchestrator for exp0-4 Analysis Suite

Runs the complete analysis pipeline: data loading, metrics calculation,
cross-experiment analysis, statistical tests, visualizations, and reports.

Usage:
    python run_full_analysis.py [--skip-viz] [--skip-reports] [--timing-only]
"""

import argparse
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

import config
from data_loader import DataLoader
from metrics_calculator import MetricsCalculator
from cross_experiment_analyzer import CrossExperimentAnalyzer
from architecture_analyzer import ArchitectureAnalyzer
from statistical_tests import StatisticalTests
from visualization.heatmap_generator import HeatmapGenerator
from visualization.comparison_plots import ComparisonPlots
from visualization.ranking_visualizer import RankingVisualizer
from reports.insight_finder import InsightFinder


def print_header(text: str, char: str = "="):
    """Print formatted section header."""
    print("\n" + char * 80)
    print(text)
    print(char * 80)


def run_timing_analysis(metrics_df=None):
    """Run timing analysis and generate timing plots.

    Args:
        metrics_df: Extended metrics DataFrame for efficiency cross-reference.
                    If None, efficiency analysis is skipped.

    Returns:
        TimingAnalyzer instance or None if timing data not found.
    """
    from timing_analyzer import TimingAnalyzer
    from visualization.timing_plots import TimingPlots

    timing_analyzer = TimingAnalyzer(metrics_df=metrics_df)
    print(f"  Training timing: {len(timing_analyzer.training_df)} entries")
    print(f"  Inference timing: {len(timing_analyzer.inference_df)} entries")

    print("\n  Computing timing analysis...")
    timing_results = timing_analyzer.analyze()
    timing_analyzer.save_all()
    print(f"  Saved {len(timing_results)} timing CSVs")

    print("\n  Generating timing plots...")
    timing_plotter = TimingPlots(timing_analyzer)
    timing_plots = timing_plotter.generate_all()
    print(f"  Generated {len(timing_plots)} timing plot PNG files")

    return timing_analyzer


def main(skip_viz: bool = False, skip_reports: bool = False,
         timing_only: bool = False):
    """Run complete analysis pipeline."""

    print_header("EXP0-4 COMPREHENSIVE ANALYSIS SUITE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output Directory: {config.OUTPUT_DIR}")

    # Ensure all output directories exist
    config.create_output_directories()

    # ==================================================================
    # TIMING-ONLY MODE
    # ==================================================================
    if timing_only:
        print_header("TIMING-ONLY ANALYSIS MODE")

        # Load metrics for MAE cross-reference (optional)
        metrics_df = None
        try:
            print("\n  Loading metrics for efficiency cross-reference...")
            loader = DataLoader()
            df = loader.load()
            calculator = MetricsCalculator(df)
            metrics_df = calculator.calculate_all()
            print(f"  Loaded {len(metrics_df)} metric entries")
        except Exception as e:
            print(f"  WARNING: Could not load metrics ({e})")
            print("  Efficiency analysis (time vs MAE) will be skipped")

        run_timing_analysis(metrics_df=metrics_df)

        print_header("TIMING ANALYSIS COMPLETE")
        print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nOutputs:")
        print(f"  CSVs: {config.METRICS_DIR}/timing_*.csv")
        print(f"  Plots: {config.TIMING_PLOTS_DIR}/")
        return

    # ==================================================================
    # FULL PIPELINE
    # ==================================================================

    print(f"Input CSV: {config.INPUT_CSV}")

    # ========================================================================
    # PHASE 1: Data Loading & Metrics Calculation
    # ========================================================================

    print_header("PHASE 1: Data Loading & Extended Metrics", "-")

    print("\nStep 1.1: Loading experimental results...")
    loader = DataLoader()
    df = loader.load()

    print("\nStep 1.2: Calculating extended metrics...")
    calculator = MetricsCalculator(df)
    extended_df = calculator.calculate_all()

    print("\nStep 1.3: Saving metrics CSVs...")
    calculator.save_extended_metrics()
    calculator.save_family_summary()
    calculator.save_experiment_summary()

    summary = loader.get_summary_stats()
    print(f"\nDataset Summary:")
    print(f"  Total rows: {summary['n_total']}")
    print(f"  Experiments: {', '.join(summary['experiments'])}")
    print(f"  Models: {summary['n_models']}")
    print(f"  Horizons: {', '.join(summary['horizons'])}")

    # ========================================================================
    # PHASE 2: Analysis
    # ========================================================================

    print_header("PHASE 2: Cross-Experiment & Statistical Analysis", "-")

    print("\nStep 2.1: Running cross-experiment analysis...")
    cross_analyzer = CrossExperimentAnalyzer(extended_df)
    cross_results = cross_analyzer.analyze()
    cross_analyzer.save_all()
    print(f"  Computed and saved {len(cross_results)} cross-experiment results")

    print("\nStep 2.2: Running architecture family analysis...")
    arch_analyzer = ArchitectureAnalyzer(extended_df)
    arch_results = arch_analyzer.analyze()
    arch_analyzer.save_all()
    print(f"  Computed and saved {len(arch_results)} architecture analysis results")

    print("\nStep 2.3: Running statistical significance tests...")
    stats_tester = StatisticalTests(extended_df)
    stats_results = stats_tester.run_all()
    stats_tester.save_all()
    print(f"  Computed and saved {len(stats_results)} statistical test results")

    # ========================================================================
    # PHASE 2.5: Timing Analysis
    # ========================================================================

    print_header("PHASE 2.5: Timing Analysis", "-")

    timing_analyzer = None
    try:
        timing_analyzer = run_timing_analysis(metrics_df=extended_df)
    except FileNotFoundError as e:
        print(f"  WARNING: Timing data not found, skipping ({e})")

    # ========================================================================
    # PHASE 3: Visualizations (Optional)
    # ========================================================================

    if not skip_viz:
        print_header("PHASE 3: Visualization Generation", "-")

        print("\nStep 3.1: Generating heatmaps...")
        heatmap_gen = HeatmapGenerator(extended_df)
        heatmaps = heatmap_gen.generate_all()
        print(f"  Generated {len(heatmaps)} heatmap PNG files (expected: 13)")

        print("\nStep 3.2: Generating comparison plots...")
        comparison_gen = ComparisonPlots(extended_df)
        comparisons = comparison_gen.generate_all()
        print(f"  Generated {len(comparisons)} comparison plot PNG files (expected: 10)")

        print("\nStep 3.3: Generating ranking visualizations...")
        ranking_gen = RankingVisualizer(extended_df)
        rankings = ranking_gen.generate_all()
        print(f"  Generated {len(rankings)} ranking visualization PNG files (expected: 10)")

        total_viz = len(heatmaps) + len(comparisons) + len(rankings)

        # Step 3.4: Uncertainty visualizations (if UQ data exists)
        uq_csv_path = config.UNCERTAINTY_DIR / 'uncertainty_metrics.csv'
        if uq_csv_path.exists():
            from visualization.uncertainty_plots import UncertaintyPlots
            print("\nStep 3.4: Generating uncertainty visualizations...")
            uq_plotter = UncertaintyPlots(config.UNCERTAINTY_DIR)
            uq_plots = uq_plotter.generate_all()
            print(f"  Generated {len(uq_plots)} uncertainty plot PNG files")
            total_viz += len(uq_plots)
        else:
            print("\nStep 3.4: Skipping uncertainty plots (run uncertainty_batch_processor.py first)")

        print(f"\n  Total visualizations: {total_viz} PNG files")
    else:
        print_header("PHASE 3: Visualization Generation [SKIPPED]", "-")

    # ========================================================================
    # PHASE 4: Reports (Optional)
    # ========================================================================

    if not skip_reports:
        print_header("PHASE 4: Data-Driven Insight Discovery", "-")

        print("\nStep 4.1: Running insight finder...")
        # Optionally load uncertainty data if available
        uq_df = None
        uq_path = config.UNCERTAINTY_DIR / 'uncertainty_metrics.csv'
        if uq_path.exists():
            import pandas as _pd
            uq_df = _pd.read_csv(uq_path)
            print(f"  Loaded uncertainty data: {len(uq_df)} entries")
        else:
            print("  No uncertainty data found (run uncertainty_batch_processor.py to generate)")

        finder = InsightFinder(extended_df, uncertainty_df=uq_df)
        report_files = finder.save(config.REPORTS_DIR)
        print(f"  Generated {len(report_files)} output files")
    else:
        print_header("PHASE 4: Data-Driven Insight Discovery [SKIPPED]", "-")

    # ========================================================================
    # Summary
    # ========================================================================

    print_header("ANALYSIS COMPLETE")

    print("\nGenerated Outputs:")
    print(f"  Metrics CSVs: {config.METRICS_DIR}")
    print(f"  Statistical CSVs: {config.STATISTICAL_DIR}")

    if not skip_viz:
        print(f"  Visualizations: {config.FIGURES_DIR}")

    if not skip_reports:
        print(f"  Reports: {config.REPORTS_DIR}")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nOK All analysis tasks completed successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run complete exp0-4 analysis pipeline'
    )
    parser.add_argument(
        '--skip-viz',
        action='store_true',
        help='Skip visualization generation (faster execution)'
    )
    parser.add_argument(
        '--skip-reports',
        action='store_true',
        help='Skip text report generation'
    )
    parser.add_argument(
        '--timing-only',
        action='store_true',
        help='Run only timing analysis (skip metrics, statistical, and other viz)'
    )

    args = parser.parse_args()

    try:
        main(skip_viz=args.skip_viz, skip_reports=args.skip_reports,
             timing_only=args.timing_only)
    except Exception as e:
        print(f"\nERROR: Analysis failed with exception:")
        print(f"  {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
