"""
Vance County Analysis Configuration

This module contains configuration parameters for analyzing Vance County
NDVI data, including statistical parameters and visualization settings.
"""
from pathlib import Path
import logging

# CRS Configuration
STANDARD_CRS = "EPSG:4326"  # WGS84 - Standard CRS for all operations

# Logging Configuration
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = logging.INFO
LOG_FILE = "vance_analysis.log"

# Analysis Years
ANALYSIS_YEARS = [2018, 2020, 2022]

# Statistical Parameters
SIGNIFICANCE_LEVEL = 0.05  # For statistical tests
MIN_SAMPLES = 30  # Minimum samples for parametric tests

# Visualization Parameters
FIGURE_SIZE = (12, 6)  # Default figure size
PLOT_DPI = 300  # Plot resolution
COLOR_SCHEME = {
    'heirs': 'blue',
    'non_heirs': 'red'
}
LINE_STYLES = {
    'heirs': '-',
    'non_heirs': '--'
}

# File Paths
DATA_DIR = Path("data/processed/vance_county")
OUTPUT_DIR = Path("output/analysis")
PROCESSED_NDVI = DATA_DIR / "vance_processed_ndvi.parquet"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Analysis Output Files
STATS_OUTPUT = OUTPUT_DIR / "ndvi_comparison_stats.csv"
TREND_PLOT = OUTPUT_DIR / "ndvi_trend_comparison.png"
DISTRIBUTION_PLOT = OUTPUT_DIR / "ndvi_distribution.png"
SUMMARY_REPORT = OUTPUT_DIR / "analysis_summary.md" 