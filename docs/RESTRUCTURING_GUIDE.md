# Heirs Property Pipeline Restructuring Guide

## Overview
This guide outlines the simplified architecture of the Heirs Property Analysis Pipeline. The restructuring focuses on core functionality while maintaining clean code practices and the Single Responsibility Principle (SRP).

## Architecture Diagrams

### Component Architecture
```mermaid
graph TB
    subgraph Configuration
        C[config.py]
        Y[config.yaml]
        Y --> C
    end

    subgraph Processing
        DP[DataProcessor]
        L[Load Data]
        V[Validate]
        P[Process]
        S[Save]
        DP --> L
        L --> V
        V --> P
        P --> S
    end

    subgraph Pipeline
        M[main.py]
        M --> C
        C --> DP
        DP --> R[Results]
    end

    subgraph Storage
        I[data/raw/*.shp]
        O[data/processed/*.gpkg]
        I --> L
        S --> O
    end

    style C fill:#f9f,stroke:#333
    style DP fill:#bbf,stroke:#333
    style M fill:#bfb,stroke:#333
```

### Data Flow
```mermaid
flowchart LR
    subgraph Input
        A[Raw Shapefile] --> B[Load]
    end

    subgraph Processing
        B --> C{Validate}
        C -->|Valid| D[Process]
        C -->|Invalid| E[Error]
        D --> F[Calculate Area]
        D --> G[Extract Centroids]
        F --> H[Combine]
        G --> H
    end

    subgraph Output
        H --> I[GeoPackage]
        H --> J[Logs]
        H --> K[Status Report]
    end

    style A fill:#f96,stroke:#333
    style I fill:#9f6,stroke:#333
    style E fill:#f66,stroke:#333
```

### Class Relationships
```mermaid
classDiagram
    class Config {
        +Path config_path
        +Dict settings
        +load_config()
        +input_dir
        +output_dir
        +years
        +required_fields
    }
    
    class DataProcessor {
        +Config config
        +Logger logger
        +load_property_data()
        +process_properties()
        +save_results()
        +run()
    }
    
    class Main {
        +main()
    }
    
    Main --> Config : uses
    Main --> DataProcessor : uses
    DataProcessor --> Config : depends on
```

## Directory Structure
```
heirs-property/
├── src/
│   ├── main.py              # Pipeline entry point
│   ├── config.py            # Configuration management
│   ├── data_processing/     
│   │   ├── processor.py     # Core processing logic
│   │   └── __init__.py
│   └── archive/            # Legacy code storage
├── data/
│   ├── raw/                # Input data directory
│   └── processed/          # Output data directory
├── requirements.txt        # Project dependencies
└── README.md              # Project documentation
```

## Core Components

### 1. Configuration Management (`config.py`)
```python
class Config:
    """Manages pipeline configuration via YAML or defaults"""
    
    # Key responsibilities:
    # - Load configuration from YAML
    # - Provide default settings
    # - Manage input/output paths
    # - Define required fields
```

Key Features:
- YAML-based configuration with sensible defaults
- Type-hinted property accessors
- Path management for data I/O
- Required field specifications

### 2. Data Processing (`processor.py`)
```python
class DataProcessor:
    """Handles core geospatial data processing"""
    
    # Key responsibilities:
    # - Load property data
    # - Validate required fields
    # - Process spatial features
    # - Save results
```

Key Features:
- Unified data processing workflow
- Built-in logging
- Error handling
- GeoPackage output format

### 3. Pipeline Entry Point (`main.py`)
```python
def main():
    """Orchestrates the complete pipeline"""
    
    # Key responsibilities:
    # - Initialize configuration
    # - Run processing pipeline
    # - Report results
```

Key Features:
- Simple command-line interface
- Clear success/failure reporting
- Pipeline orchestration

## Data Flow
1. **Input**: Raw property data (Shapefile/GeoJSON) in `data/raw/`
2. **Processing**:
   - Load and validate spatial data
   - Compute geometric properties
   - Extract features (area, centroids)
3. **Output**: Processed GeoPackage in `data/processed/`

## Implementation Guidelines

### Configuration
- Use YAML for external configuration
- Provide sensible defaults
- Type-hint all properties
- Keep configuration minimal

### Data Processing
- Focus on core spatial operations
- Validate early
- Log operations clearly
- Handle errors gracefully

### Code Style
- Use type hints
- Write clear docstrings
- Follow PEP 8
- Keep functions focused and small

## Dependencies
Essential packages only:
```
geopandas>=0.13.2
pandas>=2.0.0
pyyaml>=6.0.1
shapely>=2.0.0
```

## Usage Example
```python
# Load and run pipeline
from config import Config
from data_processing.processor import DataProcessor

# Initialize with default config
config = Config()
processor = DataProcessor(config)

# Run pipeline
results = processor.run()



