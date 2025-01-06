# Prototype Analysis Plan

## Overview
This document outlines the development plan for building a prototype of the full analysis system using test data from Vance County. The primary objective is to compare temporal NDVI trends between heirs property parcels and their neighbors.

## Data Assessment Results

### Available Data Coverage
- NDVI rasters for Vance County:
  - Years: 2018, 2020, 2022
  - Resolution: 1m (8.98315e-06 degrees)
  - Full county coverage
- Property Data:
  - 24,349 total heirs properties in NC
  - 1,000 heirs properties in Vance County
  - 102 heirs properties within NDVI coverage
  - 2,562 neighbor parcels within NDVI coverage
- Neighbor Relationships:
  - 29,646 total relationships
  - 29,250 relationships for Vance County properties

### Data Quality Considerations
1. Spatial Coverage:
   - Only 10.2% of Vance County heirs properties have NDVI coverage
   - We expext full county coverage, so we need to figure why its not overlapping correctly
   - Need to assess spatial distribution for potential bias
   - Consider impact on statistical significance

2. Temporal Resolution:
   - Two-year intervals (2018-2020, 2020-2022)
   - Sufficient for trend analysis but may miss seasonal variations

3. Control Group Selection:
   - Large pool of neighbor parcels (2,562)
   - Need criteria for selecting comparable properties
   - Consider property size, location, and land use

## Implementation Strategy

### Implemented Workflow
The analysis pipeline has been implemented with four main components:

1. Data Preparation (`data_prep.py`)
   ```python
   class DataPrep:
       """Prepare and validate property data."""
       def process_data(self):
           # - Load heirs and parcels data
           # - Convert to NC State Plane (EPSG:2264)
           # - Validate geometries
           # - Remove invalid/zero-area properties
           # - Save to output/processed/
   ```

2. Property Matching (`property_matcher.py`)
   ```python
   class PropertyMatcher:
       """Match heirs properties with comparable neighbors."""
       def find_matches(self):
           # - Size-based filtering (60-140% of heir property)
           # - Distance-based filtering (within 2000m)
           # - Exclude self-matches
           # - Save to output/matches/
   ```

3. NDVI Processing (`ndvi_processor.py`)
   ```python
   class NDVIProcessor:
       """Process NDVI data for properties."""
       def __init__(self, ndvi_dir: str = "data/raw/ndvi"):
           # Initialize with NDVI files for 2018, 2020, 2022
           
       def process_properties(self, property_ids: List[int]):
           # - Extract NDVI values using rasterio
           # - Handle CRS conversion
           # - Validate pixel data
           # - Process in memory-efficient batches
           # - Calculate temporal trends
           # - Save to output/ndvi/
   ```

4. Statistical Analysis (`stats_analyzer.py`)
   ```python
   class StatsAnalyzer:
       """Analyze NDVI patterns and create visualizations."""
       def calculate_statistics(self, matches_df: pd.DataFrame, ndvi_df: pd.DataFrame):
           # - Merge property matches with NDVI data
           # - Calculate summary statistics
           # - Perform paired t-tests
           # - Generate visualizations:
           #   * Time series comparison
           #   * Distribution plots
           #   * Difference boxplots
           # - Save to output/analysis/
   ```

### Execution Pipeline
```bash
# Complete analysis workflow
1. python3 src/analysis/data_prep.py      # Data preparation
2. python3 src/analysis/property_matcher.py # Find matches
3. python3 src/analysis/ndvi_processor.py  # Process NDVI
4. python3 src/analysis/stats_analyzer.py  # Analyze results
```

### Key Features Implemented
1. Robust Error Handling
   - Comprehensive validation at each step
   - Graceful error recovery
   - Detailed error logging

2. Memory Management
   - Batch processing for large datasets
   - Memory-mapped raster reading
   - Efficient data structures

3. Progress Tracking
   - Detailed logging at each stage
   - Progress bars for long operations
   - Status reporting

4. Data Validation
   - Geometry validation
   - CRS consistency checks
   - NDVI value range validation
   - Coverage verification

### Output Structure
```
output/
├── processed/          # Processed property data
│   ├── heirs_processed.parquet
│   └── parcels_processed.parquet
├── matches/           # Property match results
│   └── sample_matches.csv
├── ndvi/             # NDVI analysis results
│   └── sample_ndvi.csv
└── analysis/         # Final results
    ├── statistics.txt
    ├── ndvi_time_series.png
    ├── ndvi_distributions.png
    └── ndvi_differences.png
```

### Phase 3: Visualization Framework
1. Spatial Visualization
   - Property distribution maps
   - NDVI change maps
   - Neighbor relationship networks

2. Statistical Plots
   - Time series comparisons
   - Distribution comparisons
   - Trend analysis plots

3. Interactive Components
   - Property selection
   - Temporal comparison
   - Statistical summary views

## Success Metrics
1. Data Quality
   - Complete property dataset with validated geometries
   - Validated NDVI statistics for all properties
   - Documented coverage limitations

2. Analysis Framework
   - Robust statistical comparison methodology
   - Validated control group selection
   - Reproducible analysis pipeline

3. Visualization
   - Clear presentation of spatial patterns
   - Effective statistical visualizations
   - Interactive exploration capabilities

## Next Steps
1. Implement spatial distribution analysis
2. Develop control group selection criteria
3. Create initial visualization prototypes
4. Document statistical methodology

## Technical Requirements
1. Data Processing
   - Memory-efficient raster processing
   - Robust error handling
   - Progress tracking and logging

2. Analysis Tools
   - Statistical analysis packages
   - Spatial analysis tools
   - Visualization libraries

3. Documentation
   - Analysis methodology
   - Data limitations
   - Implementation details

## Expected Outcomes
1. Validated analysis methodology
2. Initial comparative results
3. Visualization prototypes
4. Scaling recommendations
5. Econometric dataset for external use

## Documentation Requirements
1. Data Coverage Report
   - Spatial distribution analysis
   - Temporal coverage assessment
   - Quality metrics

2. Methodology Documentation
   - Property selection criteria
   - Statistical approach
   - Validation procedures

3. Results Documentation
   - Initial findings
   - Limitations
   - Recommendations for scaling 
