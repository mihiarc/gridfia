# Heirs Property Analysis Data Pipeline

This document outlines the data processing pipeline for analyzing heirs properties in North Carolina, focusing on land management patterns and forest health indicators.

## Pipeline Overview

```mermaid
graph TD
    A[Raw GIS Data] --> B[Convert to Parquet]
    B --> C[Process Heirs Properties]
    D[FIA Plot Data] --> E[Analyze FIA Plots]
    C --> F[Neighbor Analysis]
    G[NDVI Data] --> H[Process NDVI]
    C --> I[Final Analysis]
    E --> I
    F --> I
    H --> I
```

## Data Sources

### 1. GIS Data
- Location: `src/data/raw/gis/`
- Files:
  - `NC.gdb`: North Carolina parcel data
  - `HP_Deliverables.gdb`: Heirs property data
  - `HP_Deliverables.gdb.zip`: Backup of heirs property data

### 2. FIA Plot Data
- Location: `src/data/raw/`
- Files:
  - `nc-fia-plots.csv`: Forest Inventory Analysis plot data
  - `nc-plots.csv`: Additional plot information

### 3. NDVI Data
- Location: `src/data/raw/ndvi/`
- Files:
  - NAIP imagery for Vance County (2018, 2020, 2022)
  - Tree canopy cover data

## Pipeline Stages

### Stage 1: GDB to Parquet Conversion
```mermaid
graph LR
    A[NC.gdb] --> B[Convert]
    C[HP_Deliverables.gdb] --> B
    B --> D[Parquet Files]
    D --> E[nc-parcels.parquet]
    D --> F[nc-hp.parquet]
```

**Process:**
1. Load GDB files using GeoPandas
2. Clean and standardize fields
3. Convert to Parquet format for efficient processing

### Stage 2: Heirs Property Processing
```mermaid
graph LR
    A[nc-hp.parquet] --> B[Process HP Data]
    B --> C[Clean Data]
    C --> D[Add Features]
    D --> E[nc-hp_v2.parquet]
```

**Process:**
1. Clean heirs property data
2. Add derived features
3. Validate property information

### Stage 3: FIA Plot Analysis
```mermaid
graph LR
    A[FIA Plot Data] --> B[Spatial Join]
    C[Heirs Properties] --> B
    B --> D[Plot Analysis]
    D --> E[Forest Health Metrics]
```

**Process:**
1. Load and clean FIA plot data
2. Perform spatial join with properties
3. Calculate forest health metrics

### Stage 4: Neighbor Analysis
```mermaid
graph TD
    A[Property Data] --> B[Find Neighbors]
    B --> C[Calculate Distance]
    C --> D[Create Buffer]
    D --> E[Spatial Join]
    E --> F[parcels_within_1_mile.parquet]
```

**Process:**
1. Identify neighboring properties
2. Calculate distances
3. Create property buffers
4. Analyze management patterns

### Stage 5: NDVI Processing
```mermaid
graph LR
    A[NAIP Imagery] --> B[Calculate NDVI]
    B --> C[Temporal Analysis]
    C --> D[Property Statistics]
    D --> E[Forest Health Indicators]
```

**Process:**
1. Process NAIP imagery
2. Calculate NDVI values
3. Extract temporal patterns
4. Generate property-level statistics

## Output Products

1. **Processed Datasets:**
   - Cleaned heirs property data
   - Property neighbor relationships
   - Forest health indicators

2. **Analysis Results:**
   - Forest management patterns
   - Temporal NDVI changes
   - Property characteristic comparisons

3. **Visualizations:**
   - NDVI change maps
   - Forest health distribution
   - Property relationship networks

## Running the Pipeline

The pipeline can be executed using:

```python
from src.pipeline.data_pipeline import HeirsPropertyPipeline, PipelineConfig

config = PipelineConfig(
    raw_gis_dir=Path("src/data/raw/gis"),
    processed_dir=Path("src/data/processed"),
    ndvi_dir=Path("src/data/raw/ndvi")
)

pipeline = HeirsPropertyPipeline(config)
pipeline.run_pipeline()
```

## Dependencies

- GeoPandas
- Pandas
- Rasterio
- NumPy
- Shapely

## Error Handling

The pipeline includes comprehensive error handling for:
- Missing input files
- Invalid data formats
- Processing failures
- Output validation

## Monitoring

Pipeline progress is monitored through:
- Logging to file and console
- Progress indicators for long-running processes
- Data validation checks
- Error reporting 

# Heirs Property Analysis Data Pipeline

## Complete Data Flow
```mermaid
flowchart TD
    subgraph Input
        GIS[GIS Data\nNC.gdb\nHP_Deliverables.gdb]
        FIA[FIA Plot Data\nnc-fia-plots.csv]
        NDVI[NDVI Data\nNAIP Imagery]
    end

    subgraph Processing
        CONV[GDB Conversion]
        HP[Heirs Property\nProcessing]
        PLOT[Plot Analysis]
        NEIGH[Neighbor Analysis]
        NDVI_PROC[NDVI Processing]
    end

    subgraph Output
        PAR[Parquet Files]
        FOREST[Forest Health\nMetrics]
        VIZ[Visualizations]
    end

    GIS --> CONV
    CONV --> PAR
    PAR --> HP
    FIA --> PLOT
    HP --> PLOT
    HP --> NEIGH
    NDVI --> NDVI_PROC
    PLOT --> FOREST
    NDVI_PROC --> FOREST
    NEIGH --> VIZ
    FOREST --> VIZ

    style Input fill:#e6f3ff,stroke:#4d94ff
    style Processing fill:#fff2e6,stroke:#ffb366
    style Output fill:#e6ffe6,stroke:#66cc66
```

## Data Processing Detail
```mermaid
flowchart LR
    subgraph GIS Processing
        direction TB
        GDB[GDB Files] --> CLEAN[Clean & Standardize]
        CLEAN --> PAR[Parquet Format]
    end

    subgraph Heirs Analysis
        direction TB
        HP[Heirs Properties] --> FEAT[Feature Engineering]
        FEAT --> VALID[Validation]
        VALID --> FINAL[Final Dataset]
    end

    subgraph Forest Health
        direction TB
        FIA[FIA Plots] --> JOIN[Spatial Join]
        JOIN --> METRICS[Health Metrics]
        NDVI[NDVI Data] --> TEMP[Temporal Analysis]
        TEMP --> METRICS
    end

    GIS Processing --> Heirs Analysis
    Heirs Analysis --> Forest Health

    style GIS Processing fill:#e6f3ff
    style Heirs Analysis fill:#fff2e6
    style Forest Health fill:#e6ffe6
```

## NDVI Processing Pipeline
```mermaid
flowchart LR
    subgraph Input
        NAIP[NAIP Imagery]
        TCC[Tree Canopy Cover]
    end

    subgraph Processing
        CALC[Calculate NDVI]
        TEMP[Temporal Analysis]
        STAT[Statistics]
    end

    subgraph Output
        MAPS[NDVI Maps]
        CHANGE[Change Detection]
        HEALTH[Health Indicators]
    end

    NAIP --> CALC
    TCC --> STAT
    CALC --> TEMP
    TEMP --> STAT
    STAT --> MAPS
    STAT --> CHANGE
    STAT --> HEALTH

    style Input fill:#e6f3ff
    style Processing fill:#fff2e6
    style Output fill:#e6ffe6
```

## Neighbor Analysis Detail
```mermaid
flowchart TD
    subgraph Input
        HP[Heirs Properties]
        PAR[All Parcels]
    end

    subgraph Spatial Processing
        BUF[Create Buffers]
        DIST[Distance Calculation]
        JOIN[Spatial Join]
    end

    subgraph Analysis
        COMP[Property Comparison]
        PATTERN[Management Patterns]
        NET[Network Analysis]
    end

    HP --> BUF
    PAR --> DIST
    BUF --> JOIN
    DIST --> JOIN
    JOIN --> COMP
    COMP --> PATTERN
    COMP --> NET

    style Input fill:#e6f3ff
    style Spatial Processing fill:#fff2e6
    style Analysis fill:#e6ffe6
```

## Data Validation Flow
```mermaid
flowchart TD
    subgraph Validation Steps
        INPUT[Input Validation]
        PROC[Processing Validation]
        OUT[Output Validation]
    end

    subgraph Checks
        SCHEMA[Schema Check]
        GEOM[Geometry Check]
        NULL[Null Check]
        TYPE[Type Check]
        RANGE[Range Check]
        SPATIAL[Spatial Validity]
    end

    INPUT --> SCHEMA
    INPUT --> NULL
    PROC --> GEOM
    PROC --> TYPE
    OUT --> RANGE
    OUT --> SPATIAL

    style Validation Steps fill:#fff2e6
    style Checks fill:#e6f3ff
```