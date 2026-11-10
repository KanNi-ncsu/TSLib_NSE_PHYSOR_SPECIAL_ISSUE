"""
Configuration for exp0-4 Analysis Suite

Contains all configuration parameters, model families, color schemes, and paths.
"""

from pathlib import Path

# ============================================================================
# PATHS (derived from script location -- no hardcoded absolute paths)
# ============================================================================

_SCRIPT_DIR = Path(__file__).resolve().parent       # NuclearResearch/analysis/
_NUCLEAR_DIR = _SCRIPT_DIR.parent                   # NuclearResearch/
_PROJECT_ROOT = _NUCLEAR_DIR.parent                 # repository root

INPUT_CSV = _NUCLEAR_DIR / 'results_summary' / 'all_experiments_metrics.csv'
OUTPUT_DIR = _NUCLEAR_DIR / 'results_summary' / 'analysis_output'
RESULTS_DIR = _PROJECT_ROOT / 'results'             # raw result directories

# Output subdirectories
METRICS_DIR = OUTPUT_DIR / 'metrics'
FIGURES_DIR = OUTPUT_DIR / 'figures'
STATISTICAL_DIR = OUTPUT_DIR / 'statistical'
REPORTS_DIR = OUTPUT_DIR / 'reports'
UNCERTAINTY_DIR = OUTPUT_DIR / 'uncertainty'

# Figure subdirectories
HEATMAPS_DIR = FIGURES_DIR / 'heatmaps'
COMPARISONS_DIR = FIGURES_DIR / 'comparisons'
RANKINGS_DIR = FIGURES_DIR / 'rankings'
UNCERTAINTY_PLOTS_DIR = FIGURES_DIR / 'uncertainty'
TIMING_PLOTS_DIR = FIGURES_DIR / 'timing'

# Timing data directory
TIMING_DIR = _NUCLEAR_DIR / 'results_summary' / 'timing'

# ============================================================================
# MODEL FAMILIES
# ============================================================================

MODEL_FAMILIES = {
    'Transformers': [
        'Transformer', 'TFT', 'Nonstationary', 'PatchTST', 'Crossformer',
        'Pyraformer', 'Reformer', 'Informer', 'iTransformer', 'FEDformer',
        'Autoformer', 'ETSformer', 'MultiPatchFormer'
    ],
    'CNNs': ['SCINet', 'TimesNet', 'MICN'],
    'RNNs': ['LSTM', 'GRU', 'SegRNN'],
    'MLPs': ['DLinear', 'TSMixer', 'TimeMixer', 'FreTS'],
    'Baselines': ['ARIMA', 'MovingAverage', 'Naive'],
    'Foundation': ['Chronos', 'Chronos2', 'TimesFM'],
    'Foundation-FT': ['Chronos2_FT', 'Chronos_FT']
}

# Finetune CSV model name -> config model name
FINETUNE_MODEL_MAP = {
    'Chronos2_multivar': 'Chronos2_FT',
    'ChronosBolt_univar': 'Chronos_FT',
}

# Reverse mapping: model -> family
MODEL_TO_FAMILY = {}
for family, models in MODEL_FAMILIES.items():
    for model in models:
        MODEL_TO_FAMILY[model] = family

# ============================================================================
# EXPERIMENTS METADATA
# ============================================================================

EXPERIMENTS = {
    'exp0': {
        'name': 'LOCA-only',
        'description': 'Pure LOCA training (75 scenarios)',
        'n_scenarios': 75,
        'composition': 'LOCA only'
    },
    'exp1': {
        'name': 'LOCA+Normal',
        'description': 'LOCA + Normal operation (95 scenarios)',
        'n_scenarios': 95,
        'composition': '75 LOCA + 20 Normal'
    },
    'exp2': {
        'name': 'LOCAC+Normal',
        'description': 'Cold-leg LOCA + Normal (95 scenarios)',
        'n_scenarios': 95,
        'composition': '75 LOCAC + 20 Normal'
    },
    'exp3': {
        'name': 'FLB+Normal',
        'description': 'Feedline Break + Normal (95 scenarios)',
        'n_scenarios': 95,
        'composition': '75 FLB + 20 Normal'
    },
    'exp4': {
        'name': 'Multi-accident',
        'description': 'Multi-accident diversity (395 scenarios)',
        'n_scenarios': 395,
        'composition': '5 accident types + Normal'
    }
}

# Experiment order for plotting
EXPERIMENT_ORDER = ['exp0', 'exp1', 'exp2', 'exp3', 'exp4']

# ============================================================================
# HORIZONS
# ============================================================================

HORIZONS = {
    '30s': {'pred_len': 3, 'seconds': 30, 'label': '30s ahead'},
    '300s': {'pred_len': 30, 'seconds': 300, 'label': '300s ahead'},
    '600s': {'pred_len': 60, 'seconds': 600, 'label': '600s ahead'}
}

HORIZON_ORDER = ['30s', '300s', '600s']

# Reverse lookup: pred_len -> horizon label
PRED_LEN_TO_HORIZON = {v['pred_len']: k for k, v in HORIZONS.items()}

# ============================================================================
# COLOR SCHEMES
# ============================================================================

# Architecture family colors
FAMILY_COLORS = {
    'Transformers': '#1f77b4',  # Blue
    'CNNs': '#ff7f0e',          # Orange
    'RNNs': '#2ca02c',          # Green
    'MLPs': '#d62728',          # Red
    'Baselines': '#9467bd',      # Purple
    'Foundation': '#8c564b',      # Brown
    'Foundation-FT': '#e377c2'    # Pink
}

# Experiment colors
EXPERIMENT_COLORS = {
    'exp0': '#e74c3c',  # Red (baseline)
    'exp1': '#3498db',  # Blue
    'exp2': '#2ecc71',  # Green
    'exp3': '#f39c12',  # Orange
    'exp4': '#9b59b6'   # Purple (most diverse)
}

# Heatmap color schemes
HEATMAP_CMAP = 'RdYlGn_r'  # Red=bad, Yellow=medium, Green=good
CORRELATION_CMAP = 'coolwarm'  # Blue=negative, Red=positive

# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================

# Figure sizes (width, height in inches)
FIGURE_SIZES = {
    'heatmap_large': (16, 12),
    'heatmap_medium': (12, 8),
    'heatmap_small': (8, 6),
    'comparison_plot': (10, 6),
    'ranking_plot': (12, 8),
    'boxplot': (10, 6)
}

# Font sizes
FONT_SIZES = {
    'title': 16,
    'axis_label': 14,
    'tick_label': 10,
    'legend': 12,
    'annotation': 9
}

# DPI for saving figures
FIGURE_DPI = 300

# ============================================================================
# STATISTICAL PARAMETERS
# ============================================================================

# Significance threshold for p-values
ALPHA = 0.05

# Number of bootstrap iterations for confidence intervals
N_BOOTSTRAP = 1000

# Confidence level for intervals
CONFIDENCE_LEVEL = 0.95

# ============================================================================
# TYPICAL PCT VALUE RANGE (for normalized metrics)
# ============================================================================

# Typical Peak Cladding Temperature range in Kelvin
PCT_MIN = 300  # K (normal operation)
PCT_MAX = 1500  # K (severe accident)
PCT_MEAN = 900  # K (approximate average)

# ============================================================================
# REPORTING PARAMETERS
# ============================================================================

# Number of top models to highlight in reports
N_TOP_MODELS = 5

# Number of decimal places for metrics
METRIC_DECIMALS = 2

# LaTeX table format
LATEX_TABLE_FORMAT = 'booktabs'

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_family(model: str) -> str:
    """Get architecture family for a model."""
    return MODEL_TO_FAMILY.get(model, 'Unknown')


def get_family_models(family: str) -> list:
    """Get all models in a family."""
    return MODEL_FAMILIES.get(family, [])


def get_experiment_label(exp: str) -> str:
    """Get short label for experiment."""
    return EXPERIMENTS[exp]['name']


def get_horizon_label(horizon: str) -> str:
    """Get label for horizon."""
    return HORIZONS[horizon]['label']


def create_output_directories():
    """Create all output directories."""
    dirs = [
        OUTPUT_DIR, METRICS_DIR, FIGURES_DIR, STATISTICAL_DIR, REPORTS_DIR,
        UNCERTAINTY_DIR, HEATMAPS_DIR, COMPARISONS_DIR, RANKINGS_DIR,
        UNCERTAINTY_PLOTS_DIR, TIMING_PLOTS_DIR, REPORTS_DIR / 'latex_tables'
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    # Test configuration
    print("="*80)
    print("Configuration Test")
    print("="*80)
    print(f"\nInput CSV: {INPUT_CSV}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"\nModel families:")
    for family, models in MODEL_FAMILIES.items():
        print(f"  {family}: {len(models)} models")
    print(f"\nExperiments: {len(EXPERIMENTS)}")
    print(f"Horizons: {len(HORIZONS)}")
    print("\nCreating output directories...")
    create_output_directories()
    print("OK - Configuration loaded successfully!")
