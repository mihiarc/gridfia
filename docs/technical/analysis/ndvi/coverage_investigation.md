# NDVI Coverage Investigation Plan

## Overview
This document outlines the investigation and resolution plan for the NDVI coverage of Vance County heirs properties. We have confirmed that we have 102 heirs properties (10.2%) with NDVI coverage, which will serve as our prototype dataset.

## Current Status
- Total heirs properties in NC: 24,349
- Heirs properties in Vance County: 1,000
- Properties with NDVI coverage: 102 (10.2%)
- NDVI temporal coverage: 2018, 2020, 2022
- NDVI resolution: 1m (8.98315e-06 degrees)

## Investigation Results
1. **Data Validation**
   - Confirmed all NDVI rasters are complete and valid
   - Property data is loading correctly
   - Intersection process is working as designed
   - Low coverage is due to partial NDVI coverage of county

2. **Coverage Analysis**
   - Current NDVI data covers a portion of Vance County
   - All 102 properties have complete NDVI coverage for all years
   - No data quality issues identified
   - Sample size is sufficient for prototype analysis

## Next Steps

### 1. Prototype Development
1. **Process Current Dataset**
   - Analyze 102 properties with full coverage
   - Develop and test analysis methodology
   - Create visualization prototypes
   - Validate statistical approaches

2. **Documentation**
   - Document analysis procedures
   - Record methodology decisions
   - Note any limitations or considerations
   - Prepare for scaling to full dataset

3. **Quality Control**
   - Validate NDVI calculations
   - Verify temporal comparisons
   - Test neighbor analysis methods
   - Document accuracy metrics

### 2. Future Expansion
1. **Data Collection**
   - Plan for full county NDVI coverage
   - Document additional data requirements
   - Prepare processing pipeline for scaling

2. **Methodology Refinement**
   - Use prototype results to improve methods
   - Identify potential scaling issues
   - Optimize processing workflows
   - Enhance analysis techniques

## Success Criteria
1. **Prototype Analysis**
   - Complete analysis of 102 properties
   - Validated methodology
   - Reproducible results
   - Clear documentation

2. **Data Quality**
   - Verified NDVI calculations
   - Validated temporal trends
   - Documented accuracy metrics
   - Clear limitations and assumptions

3. **Documentation**
   - Complete methodology documentation
   - Clear scaling requirements
   - Updated technical specifications
   - Processing guidelines

## Technical Requirements
- Python with GeoPandas and Rasterio
- PostGIS for spatial operations
- Visualization tools (Matplotlib, Folium)
- Memory-efficient processing capabilities 