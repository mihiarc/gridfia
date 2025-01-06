# NDVI Raster Data Documentation

## Vance County NDVI Dataset

### Overview
The dataset consists of NDVI (Normalized Difference Vegetation Index) rasters derived from NAIP (National Agriculture Imagery Program) imagery for Vance County, North Carolina. The data spans three time periods, providing temporal analysis capabilities.

### Available Files

1. `ndvi_NAIP_Vance_County_2018-p2.tif` (454 MB)
   - Year: 2018
   - Source: NAIP imagery
   - Processing: Part 2 of processing pipeline

2. `ndvi_NAIP_Vance_County_2020-p2.tif` (478 MB)
   - Year: 2020
   - Source: NAIP imagery
   - Processing: Part 2 of processing pipeline

3. `ndvi_NAIP_Vance_County_2022-p2.tif` (446 MB)
   - Year: 2022
   - Source: NAIP imagery
   - Processing: Part 2 of processing pipeline

### Temporal Coverage
- Spans 2018-2022
- Two-year intervals
- Three distinct time points for change analysis

### Data Characteristics
- File Format: GeoTIFF
- Source: NAIP aerial imagery
- Processing Level: p2 (indicating second phase processing)
- File Size Range: 446-478 MB
- Geographic Coverage: Vance County, NC

### Analysis Potential
1. Temporal Analysis
   - Forest health changes over 4-year period
   - Vegetation pattern changes
   - Impact assessment between 2018-2022

2. Property Analysis
   - NDVI values for individual properties
   - Comparative analysis between years
   - Forest health metric development

### Usage Notes
1. Data Processing
   - Files are already processed to NDVI values
   - '-p2' suffix indicates secondary processing
   - Ready for property-level analysis

2. Considerations
   - Large file sizes require efficient processing
   - Temporal alignment needed for change detection
   - Coordinate system verification recommended

### Next Steps
1. Technical Validation
   - Verify coordinate systems
   - Check for data gaps or artifacts
   - Validate NDVI value ranges

2. Integration Tasks
   - Align with property boundaries
   - Develop extraction workflows
   - Set up batch processing 