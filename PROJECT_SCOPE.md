# Heirs Property - Project Scope

---

## **Project Objectives**
1. Quantify the forest health of heirs property parcels in North Carolina
2. Quantify the forest health of non-heirs property parcels in North Carolina
3. Create a dataset for comparing heirs property parcels to neighboring non-heirs property parcels

---

## **IMPORTANT: PROJECT CONTINUITY**  
To maintain project context across conversations, always start a new chat with the following instructions:  

```
You are working on the Heirs Property project
Read CHANGELOG.md and PROJECT_SCOPE.md now, report your findings, and strictly follow all instructions found in these documents.  
You must complete the check-in process before proceeding with any task.  

Begin check-in process and document analysis.
```

---

## **IMPORTANT: SELF-MAINTENANCE INSTRUCTIONS**  

### **Before Taking Any Action or Making Suggestions**  
1. **Read Both Files**:  
   - Read `CHANGELOG.md` and `PROJECT_SCOPE.md`.  
   - Immediately report:  
     ```
     Initializing new conversation...  
     Read [filename]: [key points relevant to current task]  
     Starting conversation history tracking...
     ```

2. **Review Context**:  
   - Assess existing features, known issues, and architectural decisions.  

3. **Inform Responses**:  
   - Use the gathered context to guide your suggestions or actions.  

4. **Proceed Only After Context Review**:  
   - Ensure all actions align with the project’s scope and continuity requirements.

---

### **After Making ANY Code Changes**  
1. **Update Documentation Immediately**:  
   - Add new features/changes to the `[Unreleased]` section of `CHANGELOG.md`.  
   - Update `PROJECT_SCOPE.md` if there are changes to architecture, features, or limitations.

2. **Report Documentation Updates**:  
   - Use the following format to report updates:  
     ```
     Updated CHANGELOG.md: [details of what changed]  
     Updated PROJECT_SCOPE.md: [details of what changed] (if applicable)
     ```

3. **Ensure Alignment**:  
   - Verify that all changes align with existing architecture and features.

4. **Document All Changes**:  
   - Include specific details about:
     - New features or improvements
     - Bug fixes
     - Error handling changes
     - UI/UX updates
     - Technical implementation details

5. **Adhere to the Read-First/Write-After Approach**:  
   - Maintain explicit update reporting for consistency and continuity.

---

## **Project Overview**
A spatial data analysis system for tracking and analyzing heirs property parcels in North Carolina.

## **Documentation Structure**
```
docs/
├── core/               # Core project documentation
│   ├── status.md      # Project status and phases
│   ├── next_steps.md  # Future implementation plans
│   └── development.md # Development setup & guidelines
├── technical/         # Technical documentation
│   ├── analysis/     # Analysis documentation
│   │   ├── ndvi/    # NDVI analysis docs
│   │   └── neighbors/ # Neighbor analysis docs
│   ├── data_pipeline/ # Pipeline documentation
│   ├── database/     # Database configuration
│   ├── testing/      # Testing documentation
│   └── storage/      # Storage planning
└── archive/          # Archived documentation
```

## **Project Goals**

1. **Data Processing**
   - Process NC parcel data efficiently
   - Handle large GIS datasets
   - Maintain data quality and integrity

2. **Spatial Analysis**
   - Identify heirs property patterns
   - Analyze forest health indicators
   - Generate spatial statistics

3. **Visualization**
   - Create interactive maps
   - Generate analysis reports
   - Provide data dashboards

### **Project Status**

### Current Phase: Phase 2 (Analysis Prototype)

### Recent Achievements
1. Data Preparation
   - Successfully processed all property data with GEOS 3.10.6
   - Implemented robust geometry validation

2. Property Matching
   - Completed matching for Vance County properties
   - Achieved 100% coverage with high-quality matches
   - Implemented efficient spatial search with two-stage filtering
   - Generated comprehensive match statistics

3. NDVI Processing
   - Split processing into modular components:
     - MosaicCreator for efficient NDVI mosaicking
     - NDVITrendAnalyzer for temporal analysis
   - Implemented memory-efficient virtual rasters (VRT)
   - Added parallel processing with automatic CPU scaling
   - Enhanced progress tracking and error handling
   - Improved data management and statistics collection
   - Enhanced property ID tracking and validation
   - Strengthened spatial data lineage

### Processing Statistics
- Data Processing:
  - 24,349 heirs properties processed
  - 5,556,642 parcels processed
  - 1.63GB total processed data
  - 99.95% processing success rate

- Property Matching:
  - 92,342 total matches found
  - 92.3 average matches per property
  - 1,137.6m average distance
  - 0.97 average area ratio
  - 100% coverage for Vance County

- NDVI Processing:
  - Memory-efficient VRT mosaics for 2018, 2020, 2022
  - Optimized property filtering using NDVI coverage bounds
  - Processing only properties within NDVI coverage (102 properties)
  - Parallel processing with CPU-optimized workers
  - Batch-based property processing (100 properties/batch)
  - Comprehensive trend statistics and validation
  - Separate storage for trends and yearly data

### Next Steps
1. NDVI Processing
   - Rerun the full data pipeline for Vance County subset

2. Statistical Analysis
   - Compare NDVI trends between heirs vs neighbor properties
   - Analyze temporal patterns
   - Generate visualization outputs
   - Create summary reports

### Known Issues
- NDVI coverage limited to subset of Vance County (102 properties)
- Memory usage spikes during large batch processing
- Long processing times for full dataset analysis

### **Implementation Timeline**

#### Phase 2: Data Pipeline
1. Vector Data Processing
   - Parcel data preparation
   - Table creation and indexing
   - Spatial analysis setup
   - Performance optimization

2. Point Data Integration
   - FIA point data processing
   - Spatial joins configuration
   - Statistical analysis setup
   - Report generation

3. Heirs Property Integration
   - Data preparation and import
   - Relationship management
   - Analysis framework setup
   - Validation procedures

4. Raster Processing
   - NDVI calculation pipeline
   - Integration with vector data

### **Current Focus**
1. Processing NDVI data for matched properties
2. Implementing statistical analysis framework
3. Documenting analysis procedures

### **Implementation Progress**
- Data Pipeline: 80% Complete
  - ✅ Data Preparation
    - Successfully processed 24,349 heirs properties
    - Successfully processed 5,556,642 parcels
  
  - ✅ Property Matching
    - Completed Vance County property matching
    - Found 92,342 total matches
    - Achieved 92.3 average matches per property
    - Maintained high match quality (0.97 area ratio)
    - 100% coverage for target properties
  
  - 🔄 NDVI Processing
    - Optimized processing for Vance County prototype
    - Efficient spatial filtering using NDVI coverage bounds
    - Memory-efficient property loading
    - Processing only properties within NDVI coverage (102 properties)
    - Multi-year analysis framework in place
    - Validation system implemented

---

## **Technical Architecture**

### **NDVI Processing Architecture**

#### Core Components
1. Data Management
   - Separate storage for trends and yearly stats
   - Parquet-based results storage
   - Automatic cleanup of temporary files
   - Progress tracking and logging

2. Mosaic Creation System
   - Virtual raster (VRT) support for memory efficiency
   - Automatic file grouping by year
   - Temporary file management
   - Progress tracking and validation

3. Property Processing
   - Load matched properties
   - Extract NDVI values
   - Calculate temporal trends
   - Generate statistics
   - Save results

### **Project Structure**
```
heirs-property/
├── data/               # Data directory
│   ├── raw/           # Raw input data
│   ├── processed/     # Processed data
│   └── interim/       # Intermediate data
├── docs/              # Documentation
│   ├── debug/                    # Debugging documentation
│   │   └── map_visualization.md  # Map visualization debugging plan
│   ├── core/                     # Core documentation
│   ├── technical/               # Technical documentation
│   └── reference/              # Reference documentation
├── notebooks/         # Jupyter notebooks
├── src/              # Source code
│   ├── processing/   # Data processing scripts
│   └── analysis/     # Analysis tools
├── tests/            # Test suite
└── docker/           # Docker configuration
```

---

## **Data Structures**

### **Core Data Sources**
1. Property Data
   - `nc-parcels.parquet` (874MB): North Carolina parcel database
   - `nc-hp_v2.parquet` (11MB): Heirs property dataset
   
2. NDVI Data:
   - `ndvi_NAIP_Vanve_County_2018-p1.tif` (1.2GB): NDVI data for 2018
   - `ndvi_NAIP_Vanve_County_2018-p2.tif` (1.2GB): NDVI data for 2018
   - `ndvi_NAIP_Vanve_County_2020-p1.tif` (1.2GB): NDVI data for 2020
   - `ndvi_NAIP_Vanve_County_2020-p2.tif` (1.2GB): NDVI data for 2020
   - `ndvi_NAIP_Vanve_County_2022-p1.tif` (1.2GB): NDVI data for 2022
   - `ndvi_NAIP_Vanve_County_2022-p2.tif` (1.2GB): NDVI data for 2022

### **Pipeline Directory Structure**
```
src/analysis/
├── data_integrator.py
├── data_prep.py
|--- property_matcher.py
├── run_ndvi_analysis.py
     ├── mosaic_creator.py
     ├── ndvi_processor.py.py
├── stats_analyzer.py

```
---

## **Phase 2: Prototype Status 🔄**

### **Current Status**
- Completed initial data assessment
  - NDVI coverage for Vance County (2018, 2020, 2022)
  - 102 heirs properties within NDVI coverage
  - 2,562 neighbor parcels for comparison
- Implemented property filtering and validation
- Developed NDVI extraction pipeline
- Created neighbor relationship analysis