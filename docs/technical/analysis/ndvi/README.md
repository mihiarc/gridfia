# NDVI Analysis System

## Important: Project Maintenance

Before working with this system, always:

1. Read `CHANGELOG.md` and `PROJECT_SCOPE.md` at the start of any work session
   - These files contain critical project context
   - They track the project's evolution and current state
   - They define project boundaries and limitations

2. Update documentation when making changes:
   - Record all changes in `CHANGELOG.md`
   - Update scope/limitations in `PROJECT_SCOPE.md` if needed
   - Keep component documentation current
   - Document any new dependencies or requirements

3. Follow the self-maintenance instructions in these files to ensure project continuity

## System Description

This document describes the NDVI (Normalized Difference Vegetation Index) analysis system for comparing heirs and non-heirs properties.

## Overview

The system is split into four main components, each handling a specific part of the analysis pipeline:

1. Data Preparation (`data_prep.py`)
2. Property Matching (`property_matcher.py`)
3. NDVI Processing (`ndvi_processor.py`)
4. Statistical Analysis (`stats_analyzer.py`)

## Components

### 1. Data Preparation (`data_prep.py`)

Handles loading and preparing property data for analysis.

**Key Features:**
- Loads heirs and parcels data from parquet files
- Converts geometries to NC State Plane (EPSG:2264) for accurate measurements
- Validates geometries and removes invalid/zero-area properties
- Calculates and validates property areas
- Saves processed data to `output/processed/`

**Usage:**
```bash
python3 src/analysis/data_prep.py
```

### 2. Property Matching (`property_matcher.py`)

Matches heirs properties with comparable non-heirs properties based on size and distance.

**Key Features:**
- Size-based filtering (60-140% of heir property area)
- Distance-based filtering (within 2000m)
- Excludes self-matches
- Supports sample-based testing
- Saves matches to `output/matches/`

**Usage:**
```bash
python3 src/analysis/property_matcher.py
```

### 3. NDVI Processing (`ndvi_processor.py`)

Processes NDVI data for matched property pairs.

**Key Features:**
- Loads NDVI raster files for 2018, 2020, and 2022
- Extracts NDVI values for properties
- Handles CRS conversion for raster data
- Validates pixel data
- Saves NDVI values to `output/ndvi/`

**Usage:**
```bash
python3 src/analysis/ndvi_processor.py
```

### 4. Statistical Analysis (`stats_analyzer.py`)

Analyzes NDVI statistics and creates visualizations.

**Key Features:**
- Calculates summary statistics
- Performs paired t-tests
- Creates visualizations:
  - Time series comparison
  - Distribution plots
  - Difference boxplots
- Saves results to `output/analysis/`

**Usage:**
```bash
python3 src/analysis/stats_analyzer.py
```

## Running the Analysis

1. First, prepare the data:
```bash
python3 src/analysis/data_prep.py
```

2. Then find comparable property pairs:
```bash
python3 src/analysis/property_matcher.py
```

3. Extract NDVI values:
```bash
python3 src/analysis/ndvi_processor.py
```

4. Finally, analyze the results:
```bash
python3 src/analysis/stats_analyzer.py
```

## Output Structure

```
output/
├── processed/
│   ├── heirs_processed.parquet
│   └── parcels_processed.parquet
├── matches/
│   └── sample_matches.csv
├── ndvi/
│   └── sample_ndvi.csv
└── analysis/
    ├── statistics.txt
    ├── ndvi_time_series.png
    ├── ndvi_distributions.png
    └── ndvi_differences.png
```

## Testing

Each component supports sample-based testing by setting the `sample_size` parameter. This allows quick validation of the pipeline with a small subset of properties before running the full analysis.

Example:
```python
# In any of the main() functions:
sample_size = 5  # Process only 5 properties
```

## Dependencies

- geopandas
- pandas
- numpy
- rasterio
- matplotlib
- seaborn
- scipy

## Data Requirements

- Heirs properties: `data/raw/gis/nc-hp_v2.parquet`
- All parcels: `data/raw/gis/nc-parcels.parquet`
- NDVI rasters: `data/raw/ndvi/*.tif`

## Notes

1. All spatial operations use NC State Plane (EPSG:2264) for accurate area and distance calculations.
2. Property matching uses flexible criteria (60-140% size, 2000m distance) to ensure sufficient matches.
3. NDVI processing handles missing data and invalid pixels gracefully.
4. Statistical analysis includes both numerical summaries and visual representations. 