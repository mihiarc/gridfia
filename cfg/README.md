# BigMap Configuration Files

This directory contains configuration files for various BigMap analyses and operations.

## Configuration Naming Conventions

All configuration files follow these standards:
- End with `_config.yaml`
- Use underscores for word separation
- Use descriptive names indicating purpose
- Use suffixes for variants (_simple, _corrected, etc.)

## Directory Structure

```
cfg/
├── paths_config.yaml              # Infrastructure configuration
├── analysis/                      # Analysis configurations
│   ├── comparison_config.yaml
│   ├── diversity_analysis_config.yaml
│   └── total_biomass_config.yaml
├── species/                       # Species-specific configurations
│   ├── species_proportion_config.yaml
│   ├── species_proportion_corrected_config.yaml
│   ├── southern_yellow_pine_config.yaml
│   └── southern_yellow_pine_simple_config.yaml
└── data/                          # Data processing configurations
    ├── counties_config.yaml
    └── mosaic_config.yaml
```

## Available Configurations

### Infrastructure Configurations
- **paths_config.yaml** - Central directory and file path definitions for the project

### Analysis Configurations (`analysis/`)
- **comparison_config.yaml** - Statistical comparison between heirs and non-heirs properties
- **diversity_analysis_config.yaml** - Comprehensive diversity metrics with multiple thresholds
- **total_biomass_config.yaml** - Basic total biomass and richness calculations

### Species-Specific Configurations (`species/`)
- **species_proportion_config.yaml** - Template for species and group proportions
- **species_proportion_corrected_config.yaml** - Corrected version handling pre-calculated totals
- **southern_yellow_pine_config.yaml** - Detailed analysis of 4 Southern Yellow Pine species
- **southern_yellow_pine_simple_config.yaml** - Simplified version for confirmed SYP regions

### Data Processing Configurations (`data/`)
- **counties_config.yaml** - List of eastern and western NC counties
- **mosaic_config.yaml** - County-specific NDVI TIFF mosaic processing

## Configuration Structure

All configuration files use YAML format and typically include:
- `name` - Configuration identifier
- `description` - Purpose of the configuration
- `output_dir` - Where results will be saved
- `calculations` - List of enabled calculations with parameters

## Usage

Load configurations in Python:
```python
from bigmap.config import load_settings

settings = load_settings("cfg/diversity_analysis_config.yaml")
```

Or use with CLI:
```bash
bigmap calculate data.zarr --config cfg/diversity_analysis_config.yaml
```