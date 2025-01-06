"""
Vance County Prototype Analysis Configuration
"""
from pathlib import Path
from typing import Dict, List, Tuple

# Vance County Bounds (NC State Plane - EPSG:2264)
VANCE_BOUNDS: Dict[str, float] = {
    'minx': 2118000,  # Eastern bound
    'miny': 764000,   # Southern bound
    'maxx': 2155000,  # Western bound
    'maxy': 795000    # Northern bound
}

# NDVI Analysis Years
ANALYSIS_YEARS: List[int] = [2018, 2020, 2022]

# Processing Parameters
MAX_PROPERTIES: int = 102  # Known number of properties in coverage
BATCH_SIZE: int = 10      # Process properties in batches of 10
MAX_WORKERS: int = 4      # Parallel processing workers

# Matching Parameters
MATCH_DISTANCE_M: float = 2000.0  # Maximum distance for neighbor matching
MATCH_SIZE_RATIO: Tuple[float, float] = (0.6, 1.4)  # Size ratio range for matching

# File Paths
DATA_DIR = Path("data/processed/vance_county")
OUTPUT_DIR = Path("output/vance_analysis")
NDVI_DIR = Path("data/raw/ndvi/vance")

# Create directories if they don't exist
for directory in [DATA_DIR, OUTPUT_DIR, NDVI_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Validation Parameters
MIN_VALID_PIXELS: int = 10  # Minimum pixels for valid NDVI calculation
MAX_INVALID_RATIO: float = 0.3  # Maximum ratio of invalid pixels allowed

# Statistical Analysis Parameters
SIGNIFICANCE_LEVEL: float = 0.05  # For statistical tests
MIN_SAMPLES: int = 30  # Minimum samples for parametric tests 