"""
Vance County Data Processing Configuration

This module contains configuration parameters for processing Vance County
property and NDVI data. The processed data will be used for subsequent
analysis modules.
"""
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# CRS Configuration
STANDARD_CRS = "EPSG:4326"  # WGS84 - Standard CRS for all operations

# Logging Configuration
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = logging.INFO
LOG_FILE = "vance_processing.log"

# Vance County Bounds (WGS84 - EPSG:4326)
VANCE_BOUNDS: Dict[str, float] = {
    'minx': -78.51124666236885,  # Western bound
    'miny': 36.24948309810118,   # Southern bound
    'maxx': -78.27717264878582,  # Eastern bound
    'maxy': 36.54384305040146    # Northern bound
}

# NDVI Analysis Years
ANALYSIS_YEARS: List[int] = [2018, 2020, 2022]

# Processing Parameters
MAX_PROPERTIES: int = 102  # Known number of properties in coverage
BATCH_SIZE: int = 10      # Process properties in batches of 10
MAX_WORKERS: int = 20      # Parallel processing workers

# Matching Parameters
MATCH_DISTANCE_M: float = 1000.0  # Maximum distance (meters) for neighbor matching
MATCH_SIZE_RATIO: Tuple[float, float] = (0.9, 1.1)  # Size ratio range for matching

# File Paths
DATA_DIR = Path("data/processed/vance_county")
OUTPUT_DIR = Path("output/vance_processing")
NDVI_DIR = Path("data/raw/ndvi")

# Create directories if they don't exist
for directory in [DATA_DIR, OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Validation Parameters
MIN_VALID_PIXELS: int = 10  # Minimum pixels for valid NDVI calculation
MAX_INVALID_RATIO: float = 0.3  # Maximum ratio of invalid pixels allowed

# Statistical Analysis Parameters
SIGNIFICANCE_LEVEL: float = 0.05  # For statistical tests
MIN_SAMPLES: int = 30  # Minimum samples for parametric tests 