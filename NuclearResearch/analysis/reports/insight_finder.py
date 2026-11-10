"""
Data-Driven Insight Finder for exp0-4 Analysis Suite

Replaces template-based summary_report.py and uncertainty_benchmark_report.py
with algorithmic insight discovery. No hardcoded model names, experiment IDs,
or predefined narratives -- every finding is computed from data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy import stats
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).parent.parent))
import config


class InsightFinder:
    """Data-driven insight discovery from experimental results.

    Scans the extended metrics DataFrame and optionally the uncertainty
    DataFrame to algorithmically surface non-obvious findings. Findings
    are ranked by effect size so the most important appear first.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        uncertainty_df: pd.DataFrame = None
    ):
        """
        Initialize insight finder.

        Args:
            df: Extended metrics DataFrame (from MetricsCalculator)
            uncertainty_df: Optional uncertainty metrics (from uncertainty_batch_processor)
        """
        self.df = df.copy()
        self.uq_df = uncertainty_df.copy() if uncertainty_df is not None else None
        self.findings = []

        # Derive available dimensions from data
        self.experiments = sorted(self.df['experiment'].unique())
        self.models = sorted(self.df['model'].unique())
        self.horizons = sorted(self.df['horizon'].unique())
        self.families = sorted(self.df['family'].unique()) if 'family' in self.df.columns else []

        # Identify foundation vs specialist models from config
        foundation_set = set(config.MODEL_FAMILIES.get('Foundation', []))
        self.foundation_models = [m for m in self.models if m in foundation_set]
        self.specialist_models = [m for m in self.models if m not in foundation_set]

    def discover_all(self) -> pd.DataFrame:
        """
        Run all discovery methods.

        Returns:
            DataFrame of findings sorted by effect_size (descending).
        """
        self.findings = []

        # Performance insights
        self._find_foundation_vs_specialist()
        self._find_rank_shifts()
        self._find_horizon_degradation()
        self._find_experiment_effects()
        self._find_pareto_front()
        self._find_family_gaps()

        # Uncertainty insights (only if UQ data provided)
        if self.uq_df is not None:
            self._find_ood_signals()
            self._find_tightest_uncertainty()
            self._find_coverage_anomalies()
            self._find_uncertainty_vs_accuracy()

        findings_df = pd.DataFrame(self.findings)
        if len(findings_df) > 0:
            findings_df = findings_df.sort_values('effect_size', ascending=False)
            findings_df = findings_df.reset_index(drop=True)
        return findings_df

    def save(self, output_dir: Path = None) -> Dict[str, Path]:
        """
        Save findings to CSV and auto-generated summary text.

        Args:
            output_dir: Output directory (default: config.REPORTS_DIR)

        Returns:
            Dictionary mapping output names to file paths.
        """
        if output_dir is None:
            output_dir = config.REPORTS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        findings_df = self.discover_all()

        output_files = {}

        # Save CSV
        csv_path = output_dir / 'findings.csv'
        findings_df.to_csv(csv_path, index=False)
        output_files['findings_csv'] = csv_path
        print(f"OK Saved {len(findings_df)} findings to: {csv_path.name}")

        # Generate and save summary text
        txt_path = output_dir / 'findings_summary.txt'
        self._write_summary(findings_df, txt_path)
        output_files['findings_summary'] = txt_path
        print(f"OK Saved summary to: {txt_path.name}")

        return output_files

    # ========================================================================
    # PERFORMANCE INSIGHTS
    # ========================================================================

    def _find_foundation_vs_specialist(self):
        """Per (experiment, horizon): where do foundation models rank?"""
        if not self.foundation_models:
            return

        for horizon in self.horizons:
            for exp in self.experiments:
                subset = self.df[
                    (self.df['experiment'] == exp) &
                    (self.df['horizon'] == horizon)
                ].copy()

                if len(subset) == 0:
                    continue

                # Rank all models by MAE (rank 1 = best)
                subset = subset.sort_values('mae')
                n_total = len(subset)

                for fm in self.foundation_models:
                    fm_row = subset[subset['model'] == fm]
                    if len(fm_row) == 0:
                        continue

                    fm_mae = fm_row.iloc[0]['mae']
                    rank = (subset['mae'] < fm_mae).sum() + 1
                    percentile = (n_total - rank) / n_total * 100

                    # How many specialists does it beat?
                    specialist_maes = subset[
                        subset['model'].isin(self.specialist_models)
                    ]['mae']
                    n_beaten = (specialist_maes > fm_mae).sum()
                    n_specialists = len(specialist_maes)

                    self._add_finding(
                        category='foundation_vs_specialist',
                        finding=(
                            f"{fm} ranks {rank}/{n_total} in {exp}/{horizon} "
                            f"(percentile {percentile:.0f}%, beats "
                            f"{n_beaten}/{n_specialists} specialists)"
                        ),
                        evidence=f"MAE={fm_mae:.2f}, rank={rank}/{n_total}",
                        effect_size=percentile / 100,
                        models=fm,
                        experiments=exp,
                        horizons=horizon
                    )

    def _find_rank_shifts(self):
        """Models with large rank changes across experiments."""
        if len(self.experiments) < 2:
            return

        for horizon in self.horizons:
            horizon_data = self.df[self.df['horizon'] == horizon].copy()
            if len(horizon_data) == 0:
                continue

            # Build rank table: model -> {exp: rank}
            rank_table = {}
            for exp in self.experiments:
                exp_data = horizon_data[
                    horizon_data['experiment'] == exp
                ].sort_values('mae')
                for i, (_, row) in enumerate(exp_data.iterrows(), 1):
                    rank_table.setdefault(row['model'], {})[exp] = i

            # Find models with large rank std
            for model, ranks in rank_table.items():
                if len(ranks) < 2:
                    continue
                rank_values = list(ranks.values())
                rank_std = np.std(rank_values)
                rank_range = max(rank_values) - min(rank_values)
                best_exp = min(ranks, key=ranks.get)
                worst_exp = max(ranks, key=ranks.get)

                if rank_range >= 8:
                    self._add_finding(
                        category='rank_shift',
                        finding=(
                            f"{model} rank varies by {rank_range} positions at "
                            f"{horizon}: rank {ranks[best_exp]} in {best_exp} "
                            f"vs rank {ranks[worst_exp]} in {worst_exp}"
                        ),
                        evidence=(
                            f"rank_std={rank_std:.1f}, range={rank_range}, "
                            f"ranks={ranks}"
                        ),
                        effect_size=rank_range / max(len(self.models), 1),
                        models=model,
                        experiments=f"{best_exp},{worst_exp}",
                        horizons=horizon
                    )

    def _find_horizon_degradation(self):
        """Models that degrade unusually fast or slow with horizon."""
        if len(self.horizons) < 2:
            return

        first_horizon = self.horizons[0]
        last_horizon = self.horizons[-1]

        for exp in self.experiments:
            exp_data = self.df[self.df['experiment'] == exp]
            ratios = {}

            for model in self.models:
                first_mae = exp_data[
                    (exp_data['model'] == model) &
                    (exp_data['horizon'] == first_horizon)
                ]['mae']
                last_mae = exp_data[
                    (exp_data['model'] == model) &
                    (exp_data['horizon'] == last_horizon)
                ]['mae']

                if len(first_mae) > 0 and len(last_mae) > 0:
                    first_val = first_mae.iloc[0]
                    if first_val > 0:
                        ratios[model] = last_mae.iloc[0] / first_val

            if len(ratios) < 3:
                continue

            ratio_values = np.array(list(ratios.values()))
            median_ratio = np.median(ratio_values)
            mad = np.median(np.abs(ratio_values - median_ratio))

            for model, ratio in ratios.items():
                if mad > 0:
                    z_score = abs(ratio - median_ratio) / (mad * 1.4826)
                else:
                    z_score = 0

                if z_score >= 2.0:
                    direction = "degrades slower" if ratio < median_ratio else "degrades faster"
                    self._add_finding(
                        category='horizon_degradation',
                        finding=(
                            f"{model} {direction} than typical in {exp}: "
                            f"MAE ratio ({last_horizon}/{first_horizon}) = "
                            f"{ratio:.1f}x vs median {median_ratio:.1f}x"
                        ),
                        evidence=f"z_score={z_score:.1f}, ratio={ratio:.1f}, median={median_ratio:.1f}",
                        effect_size=z_score / 5.0,
                        models=model,
                        experiments=exp,
                        horizons=f"{first_horizon},{last_horizon}"
                    )

    def _find_experiment_effects(self):
        """Per model: which experiment helps/hurts most vs exp0."""
        if 'exp0' not in self.experiments or len(self.experiments) < 2:
            return

        for horizon in self.horizons:
            for model in self.models:
                exp0_row = self.df[
                    (self.df['model'] == model) &
                    (self.df['experiment'] == 'exp0') &
                    (self.df['horizon'] == horizon)
                ]
                if len(exp0_row) == 0:
                    continue

                exp0_mae = exp0_row.iloc[0]['mae']
                if exp0_mae == 0:
                    continue

                for exp in self.experiments:
                    if exp == 'exp0':
                        continue

                    exp_row = self.df[
                        (self.df['model'] == model) &
                        (self.df['experiment'] == exp) &
                        (self.df['horizon'] == horizon)
                    ]
                    if len(exp_row) == 0:
                        continue

                    exp_mae = exp_row.iloc[0]['mae']
                    pct_change = (exp_mae - exp0_mae) / exp0_mae * 100

                    # Only report large effects (>20% change)
                    if abs(pct_change) >= 20:
                        direction = "improves" if pct_change < 0 else "degrades"
                        self._add_finding(
                            category='experiment_effect',
                            finding=(
                                f"{model} {direction} {abs(pct_change):.1f}% "
                                f"at {horizon} from exp0 to {exp} "
                                f"(MAE {exp0_mae:.2f} -> {exp_mae:.2f})"
                            ),
                            evidence=f"pct_change={pct_change:+.1f}%, exp0_mae={exp0_mae:.2f}, {exp}_mae={exp_mae:.2f}",
                            effect_size=abs(pct_change) / 100,
                            models=model,
                            experiments=f"exp0,{exp}",
                            horizons=horizon
                        )

    def _find_pareto_front(self):
        """Models on the Pareto front (not dominated across all horizons)."""
        for exp in self.experiments:
            exp_data = self.df[self.df['experiment'] == exp]

            # Build MAE matrix: model -> [mae_h1, mae_h2, mae_h3]
            mae_matrix = {}
            for model in self.models:
                maes = []
                for h in self.horizons:
                    row = exp_data[
                        (exp_data['model'] == model) &
                        (exp_data['horizon'] == h)
                    ]
                    if len(row) == 0:
                        break
                    maes.append(row.iloc[0]['mae'])

                if len(maes) == len(self.horizons):
                    mae_matrix[model] = maes

            if len(mae_matrix) < 2:
                continue

            # Find Pareto front
            pareto_models = []
            models_list = list(mae_matrix.keys())

            for model in models_list:
                dominated = False
                for other in models_list:
                    if other == model:
                        continue
                    # other dominates model if it is <= on all horizons and < on at least one
                    all_leq = all(
                        mae_matrix[other][i] <= mae_matrix[model][i]
                        for i in range(len(self.horizons))
                    )
                    any_lt = any(
                        mae_matrix[other][i] < mae_matrix[model][i]
                        for i in range(len(self.horizons))
                    )
                    if all_leq and any_lt:
                        dominated = True
                        break

                if not dominated:
                    pareto_models.append(model)

            if pareto_models:
                avg_rank = np.mean([
                    sorted(mae_matrix.keys(),
                           key=lambda m: np.mean(mae_matrix[m])).index(pm) + 1
                    for pm in pareto_models
                ])

                self._add_finding(
                    category='pareto_front',
                    finding=(
                        f"{exp} Pareto-optimal models ({len(pareto_models)}/{len(mae_matrix)}): "
                        f"{', '.join(pareto_models)}"
                    ),
                    evidence=f"n_pareto={len(pareto_models)}, n_total={len(mae_matrix)}",
                    effect_size=1.0 - len(pareto_models) / len(mae_matrix),
                    models=','.join(pareto_models),
                    experiments=exp,
                    horizons=','.join(self.horizons)
                )

    def _find_family_gaps(self):
        """Per horizon: significant performance gaps between architecture families."""
        if len(self.families) < 2:
            return

        for horizon in self.horizons:
            for exp in self.experiments:
                subset = self.df[
                    (self.df['experiment'] == exp) &
                    (self.df['horizon'] == horizon)
                ]

                # Collect MAEs by family
                family_maes = {}
                for family in self.families:
                    maes = subset[subset['family'] == family]['mae'].values
                    if len(maes) >= 2:
                        family_maes[family] = maes

                if len(family_maes) < 2:
                    continue

                # Find best and worst families by median MAE
                family_medians = {
                    f: np.median(m) for f, m in family_maes.items()
                }
                best_family = min(family_medians, key=family_medians.get)
                worst_family = max(family_medians, key=family_medians.get)

                best_maes = family_maes[best_family]
                worst_maes = family_maes[worst_family]

                # Mann-Whitney U test
                if len(best_maes) >= 2 and len(worst_maes) >= 2:
                    try:
                        u_stat, p_value = stats.mannwhitneyu(
                            best_maes, worst_maes, alternative='less'
                        )
                    except ValueError:
                        continue

                    gap_pct = (
                        (family_medians[worst_family] - family_medians[best_family])
                        / family_medians[best_family] * 100
                    )

                    if p_value < config.ALPHA:
                        self._add_finding(
                            category='family_gap',
                            finding=(
                                f"{best_family} significantly outperforms "
                                f"{worst_family} at {horizon}/{exp}: "
                                f"median MAE {family_medians[best_family]:.2f} vs "
                                f"{family_medians[worst_family]:.2f} "
                                f"({gap_pct:.0f}% gap, p={p_value:.4f})"
                            ),
                            evidence=f"U={u_stat:.0f}, p={p_value:.4f}, gap={gap_pct:.1f}%",
                            effect_size=gap_pct / 100,
                            models=f"{best_family} vs {worst_family}",
                            experiments=exp,
                            horizons=horizon
                        )

    # ========================================================================
    # UNCERTAINTY INSIGHTS
    # ========================================================================

    def _find_ood_signals(self):
        """Experiments where uncertainty is abnormally wide vs exp0."""
        if self.uq_df is None or 'conf_90_radius' not in self.uq_df.columns:
            return
        if 'exp0' not in self.uq_df['experiment'].values:
            return

        for horizon in self.uq_df['horizon'].unique():
            for model in self.uq_df['model'].unique():
                exp0_row = self.uq_df[
                    (self.uq_df['model'] == model) &
                    (self.uq_df['experiment'] == 'exp0') &
                    (self.uq_df['horizon'] == horizon)
                ]
                if len(exp0_row) == 0:
                    continue

                exp0_radius = exp0_row.iloc[0]['conf_90_radius']
                if exp0_radius <= 0:
                    continue

                for exp in self.uq_df['experiment'].unique():
                    if exp == 'exp0':
                        continue

                    exp_row = self.uq_df[
                        (self.uq_df['model'] == model) &
                        (self.uq_df['experiment'] == exp) &
                        (self.uq_df['horizon'] == horizon)
                    ]
                    if len(exp_row) == 0:
                        continue

                    exp_radius = exp_row.iloc[0]['conf_90_radius']
                    amplification = exp_radius / exp0_radius

                    if amplification >= 2.0:
                        self._add_finding(
                            category='ood_signal',
                            finding=(
                                f"{model} 90% interval {amplification:.1f}x wider "
                                f"in {exp} vs exp0 at {horizon} "
                                f"(+/-{exp0_radius:.2f} -> +/-{exp_radius:.2f})"
                            ),
                            evidence=f"amplification={amplification:.1f}x, exp0_radius={exp0_radius:.2f}, {exp}_radius={exp_radius:.2f}",
                            effect_size=amplification / 10.0,
                            models=model,
                            experiments=f"exp0,{exp}",
                            horizons=horizon
                        )

    def _find_tightest_uncertainty(self):
        """Per (experiment, horizon): models with smallest conformal radius."""
        if self.uq_df is None or 'conf_90_radius' not in self.uq_df.columns:
            return

        for horizon in self.uq_df['horizon'].unique():
            for exp in self.uq_df['experiment'].unique():
                subset = self.uq_df[
                    (self.uq_df['experiment'] == exp) &
                    (self.uq_df['horizon'] == horizon)
                ].sort_values('conf_90_radius')

                if len(subset) < 3:
                    continue

                top3 = subset.head(3)
                models_str = ', '.join(
                    f"{r['model']}(+/-{r['conf_90_radius']:.2f})"
                    for _, r in top3.iterrows()
                )

                self._add_finding(
                    category='tightest_uncertainty',
                    finding=(
                        f"Tightest 90% intervals at {horizon}/{exp}: {models_str}"
                    ),
                    evidence=f"top3_radii={top3['conf_90_radius'].tolist()}",
                    effect_size=0.3,
                    models=','.join(top3['model'].tolist()),
                    experiments=exp,
                    horizons=horizon
                )

    def _find_coverage_anomalies(self):
        """Models with empirical coverage deviating >5pp from target."""
        if self.uq_df is None or 'coverage_90_empirical' not in self.uq_df.columns:
            return

        for _, row in self.uq_df.iterrows():
            coverage = row['coverage_90_empirical']
            gap = abs(coverage - 0.90)

            if gap >= 0.05:
                direction = "over-covers" if coverage > 0.90 else "under-covers"
                self._add_finding(
                    category='coverage_anomaly',
                    finding=(
                        f"{row['model']} {direction} at "
                        f"{row['horizon']}/{row['experiment']}: "
                        f"{coverage*100:.1f}% empirical vs 90% target "
                        f"(gap={gap*100:.1f}pp)"
                    ),
                    evidence=f"coverage={coverage:.3f}, target=0.90, gap={gap:.3f}",
                    effect_size=gap,
                    models=row['model'],
                    experiments=row['experiment'],
                    horizons=row['horizon']
                )

    def _find_uncertainty_vs_accuracy(self):
        """Correlation between MAE and conformal radius. Flag mismatches."""
        if self.uq_df is None or 'conf_90_radius' not in self.uq_df.columns:
            return
        if 'mae' not in self.uq_df.columns:
            return

        for horizon in self.uq_df['horizon'].unique():
            subset = self.uq_df[self.uq_df['horizon'] == horizon].copy()
            if len(subset) < 5:
                continue

            corr, p_val = stats.spearmanr(subset['mae'], subset['conf_90_radius'])

            self._add_finding(
                category='uncertainty_accuracy_correlation',
                finding=(
                    f"MAE vs uncertainty correlation at {horizon}: "
                    f"Spearman rho={corr:.3f} (p={p_val:.4f}) -- "
                    f"{'strong' if abs(corr) > 0.7 else 'moderate' if abs(corr) > 0.4 else 'weak'} "
                    f"{'positive' if corr > 0 else 'negative'} association"
                ),
                evidence=f"spearman_rho={corr:.3f}, p={p_val:.4f}, n={len(subset)}",
                effect_size=abs(corr),
                models='all',
                experiments='all',
                horizons=horizon
            )

            # Flag specific models with low MAE but wide intervals (or vice versa)
            mae_rank = subset['mae'].rank(pct=True)
            radius_rank = subset['conf_90_radius'].rank(pct=True)
            mismatch = (mae_rank - radius_rank).abs()

            for idx in mismatch.nlargest(3).index:
                row = subset.loc[idx]
                mae_pct = mae_rank.loc[idx] * 100
                radius_pct = radius_rank.loc[idx] * 100

                if mismatch.loc[idx] >= 0.4:
                    if mae_pct < radius_pct:
                        desc = "low MAE but wide intervals (overconfident errors)"
                    else:
                        desc = "high MAE but narrow intervals (well-calibrated despite poor accuracy)"

                    self._add_finding(
                        category='uncertainty_mismatch',
                        finding=(
                            f"{row['model']} at {horizon}/{row['experiment']}: "
                            f"{desc} -- MAE percentile={mae_pct:.0f}%, "
                            f"interval percentile={radius_pct:.0f}%"
                        ),
                        evidence=f"mae_pct={mae_pct:.0f}, radius_pct={radius_pct:.0f}, mismatch={mismatch.loc[idx]:.2f}",
                        effect_size=mismatch.loc[idx],
                        models=row['model'],
                        experiments=row['experiment'],
                        horizons=horizon
                    )

    # ========================================================================
    # HELPERS
    # ========================================================================

    def _add_finding(
        self,
        category: str,
        finding: str,
        evidence: str,
        effect_size: float,
        models: str,
        experiments: str,
        horizons: str
    ):
        """Add a finding to the internal list."""
        self.findings.append({
            'category': category,
            'finding': finding,
            'evidence': evidence,
            'effect_size': round(effect_size, 4),
            'models': models,
            'experiments': experiments,
            'horizons': horizons
        })

    def _write_summary(self, findings_df: pd.DataFrame, output_path: Path):
        """Auto-generate human-readable summary from findings DataFrame."""
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DATA-DRIVEN FINDINGS SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total findings: {len(findings_df)}\n")
            f.write(f"Experiments: {', '.join(self.experiments)}\n")
            f.write(f"Models: {len(self.models)} ({len(self.specialist_models)} specialist, "
                    f"{len(self.foundation_models)} foundation)\n")
            f.write(f"Horizons: {', '.join(self.horizons)}\n")
            if self.uq_df is not None:
                f.write(f"Uncertainty data: available ({len(self.uq_df)} entries)\n")
            else:
                f.write("Uncertainty data: not available\n")
            f.write("\n")

            if len(findings_df) == 0:
                f.write("No findings discovered.\n")
                return

            # Group by category and write
            categories = findings_df['category'].unique()

            # Category display names
            cat_names = {
                'foundation_vs_specialist': 'Foundation vs Specialist Models',
                'rank_shift': 'Rank Instability Across Experiments',
                'horizon_degradation': 'Horizon Degradation Outliers',
                'experiment_effect': 'Transfer Learning Effects (vs exp0)',
                'pareto_front': 'Pareto-Optimal Models',
                'family_gap': 'Architecture Family Significance',
                'ood_signal': 'Out-of-Distribution Signals',
                'tightest_uncertainty': 'Most Confident Predictions',
                'coverage_anomaly': 'Coverage Calibration Anomalies',
                'uncertainty_accuracy_correlation': 'Uncertainty-Accuracy Correlation',
                'uncertainty_mismatch': 'Uncertainty-Accuracy Mismatches',
            }

            for cat in categories:
                cat_findings = findings_df[findings_df['category'] == cat]
                cat_name = cat_names.get(cat, cat)

                f.write("-" * 80 + "\n")
                f.write(f"{cat_name} ({len(cat_findings)} findings)\n")
                f.write("-" * 80 + "\n\n")

                # Show top findings (up to 15 per category)
                for i, (_, row) in enumerate(cat_findings.head(15).iterrows(), 1):
                    f.write(f"  {i}. {row['finding']}\n")
                    f.write(f"     [{row['evidence']}]\n\n")

                remaining = len(cat_findings) - 15
                if remaining > 0:
                    f.write(f"  ... and {remaining} more (see findings.csv)\n\n")

            f.write("=" * 80 + "\n")
            f.write("END OF FINDINGS\n")
            f.write("=" * 80 + "\n")


if __name__ == '__main__':
    print("=" * 80)
    print("InsightFinder Test")
    print("=" * 80)
    print()

    from data_loader import DataLoader
    from metrics_calculator import MetricsCalculator

    # Load data
    loader = DataLoader()
    df = loader.load()
    calculator = MetricsCalculator(df)
    extended_df = calculator.calculate_all()

    # Optionally load UQ data
    uq_df = None
    uq_path = config.UNCERTAINTY_DIR / 'uncertainty_metrics.csv'
    if uq_path.exists():
        uq_df = pd.read_csv(uq_path)
        print(f"Loaded uncertainty data: {len(uq_df)} entries")
    else:
        print("No uncertainty data found (run uncertainty_batch_processor.py first)")

    # Run insight finder
    finder = InsightFinder(extended_df, uncertainty_df=uq_df)
    output_files = finder.save()

    print()
    print("=" * 80)
    for name, path in output_files.items():
        print(f"  OK {name}: {path.name}")
    print("=" * 80)
    print("\nOK InsightFinder test successful!")
