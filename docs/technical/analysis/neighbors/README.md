# Neighbor Analysis System

## Overview
The neighbor analysis system identifies and analyzes relationships between heirs properties and nearby non-heirs properties in Vance County, NC. This system is a critical component for comparing NDVI trends between heirs and non-heirs properties.

## Implementation Details

### Core Components

1. **Neighbor Finder (`scripts/analyze_neighbors.py`)**
   - Uses R-tree spatial indexing for efficient neighbor identification
   - Supports configurable buffer distances (1-10km)
   - Calculates key metrics:
     - Distance between parcels
     - Cardinal direction
     - Shared boundaries
     - Area relationships

2. **Property Selector (`src/analysis/analyze_properties.py`)**
   - Filters comparable properties based on:
     - Size tolerance (±20% by default)
     - Maximum distance (1km default)
     - Land use characteristics (when available)
     - Limits to 5 matches per property

3. **NDVI Integration**
   - Combines neighbor relationships with NDVI data
   - Calculates weighted NDVI statistics based on distance
   - Supports temporal analysis across years (2018-2022)

### Data Flow

1. **Input Data**
   - Heirs properties: `data/raw/gis/nc-hp_v2.parquet`
   - NC parcels: `data/raw/gis/nc-parcels.parquet`
   - NDVI data: `data/raw/ndvi/*.tif`

2. **Processing Steps**
   ```
   Load Data → Filter Vance County → Project to NC State Plane →
   Find Neighbors → Calculate Metrics → Match Properties →
   Extract NDVI → Generate Statistics
   ```

3. **Output Files**
   - Neighbor relationships: `data/processed/neighbors_*.parquet`
   - Analysis results: `data/processed/neighbor_analysis_results.json`

## Usage

### Running the Analysis

1. **Generate Neighbor Relationships**
   ```bash
   python scripts/analyze_neighbors.py
   ```
   - Creates neighbor relationship files for each buffer distance
   - Generates comprehensive statistics

2. **Analyze Properties with NDVI**
   ```bash
   python src/analysis/analyze_properties.py
   ```
   - Matches comparable properties
   - Extracts and compares NDVI values
   - Produces statistical comparisons

### Configuration

1. **Buffer Distances**
   - Default: 1km, 2.5km, 5km, 7.5km, 10km
   - Configurable in `NeighborAnalyzer` initialization

2. **Property Matching**
   - Size tolerance: ±20% (configurable)
   - Maximum distance: 1km (configurable)
   - Maximum matches: 5 per property (configurable)

3. **Spatial Reference**
   - Uses NC State Plane (EPSG:2264) for accurate distance calculations
   - Input data automatically projected from WGS84

## Technical Details

### Spatial Indexing
- Uses R-tree indexing for efficient spatial queries
- Significantly improves performance for large datasets
- Implemented in `create_spatial_index` method

### Distance Calculations
- Performed in NC State Plane projection
- Ensures accurate metric distances
- Includes both point-to-point and shared boundary calculations

### NDVI Processing
- Handles split NDVI files per year
- Combines data from multiple files when properties span file boundaries
- Uses weighted averaging based on distance for neighbor comparisons

### Performance Considerations
- Spatial indexing reduces query time from O(n²) to O(n log n)
- Chunked processing for large datasets
- Memory-efficient NDVI extraction

## Data Quality

### Validation Checks
1. **Geometry Validation**
   - Checks for valid geometries
   - Handles null geometries
   - Reports invalid cases

2. **Coverage Analysis**
   - Verifies NDVI data availability
   - Checks temporal coverage
   - Validates spatial relationships

3. **Statistical Validation**
   - Outlier detection
   - Coverage completeness
   - Relationship verification

### Known Limitations
1. **Coverage**
   - Limited to Vance County
   - NDVI coverage for ~10.2% of properties
   - Temporal gaps in NDVI data

2. **Processing**
   - Memory intensive for large buffer distances
   - Computationally expensive for full NC dataset
   - Some properties may lack suitable matches

## Future Improvements

1. **Planned Enhancements**
   - Parallel processing for large datasets
   - Advanced property matching criteria
   - Enhanced visualization tools

2. **Potential Extensions**
   - Additional counties
   - More sophisticated matching algorithms
   - Integration with other environmental data

## References

1. **Key Files**
   - `scripts/analyze_neighbors.py`: Main neighbor analysis
   - `src/analysis/analyze_properties.py`: Property selection and NDVI analysis
   - `docs/technical/analysis/neighbors/`: Additional documentation

2. **Related Documentation**
   - Project scope: `PROJECT_SCOPE.md`
   - Change history: `CHANGELOG.md`
   - Technical specs: `docs/technical/` 