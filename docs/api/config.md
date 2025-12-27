# Configuration

GridFIA uses Pydantic v2 for type-safe configuration management. Settings can be
loaded from files, environment variables, or created programmatically.

## Overview

| Class | Description |
|-------|-------------|
| [`GridFIASettings`](#gridfiasettings) | Main settings class |
| [`CalculationConfig`](#calculationconfig) | Individual calculation configuration |
| [`ProcessingConfig`](#processingconfig) | Processing parameters |
| [`VisualizationConfig`](#visualizationconfig) | Visualization parameters |
| [`OutputFormat`](#outputformat) | Supported output formats |

## Quick Start

```python
from gridfia import GridFIA, GridFIASettings
from gridfia.config import CalculationConfig

# Use default settings
api = GridFIA()

# Load from file
api = GridFIA(config="config/production.yaml")

# Programmatic configuration
settings = GridFIASettings(
    output_dir="results",
    calculations=[
        CalculationConfig(name="species_richness", enabled=True),
        CalculationConfig(name="shannon_diversity", enabled=True),
    ]
)
api = GridFIA(config=settings)
```

## GridFIASettings

Main settings class for GridFIA application.

::: gridfia.config.GridFIASettings
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - app_name
        - debug
        - verbose
        - data_dir
        - output_dir
        - cache_dir
        - visualization
        - processing
        - calculations
        - species_codes
        - get_output_path
        - get_temp_path

### Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `app_name` | `str` | `"GridFIA"` | Application name |
| `debug` | `bool` | `False` | Enable debug mode |
| `verbose` | `bool` | `False` | Enable verbose output |
| `data_dir` | `Path` | `"data"` | Base directory for data files |
| `output_dir` | `Path` | `"output"` | Base directory for output files |
| `cache_dir` | `Path` | `".cache"` | Directory for caching |
| `visualization` | `VisualizationConfig` | (defaults) | Visualization parameters |
| `processing` | `ProcessingConfig` | (defaults) | Processing parameters |
| `calculations` | `List[CalculationConfig]` | (defaults) | Calculations to perform |
| `species_codes` | `List[str]` | `[]` | Valid species codes |

### Environment Variables

Settings can be configured via environment variables with the `GRIDFIA_` prefix:

```bash
export GRIDFIA_DEBUG=true
export GRIDFIA_VERBOSE=true
export GRIDFIA_OUTPUT_DIR=/data/results
export GRIDFIA_DATA_DIR=/data/input
export GRIDFIA_CACHE_DIR=/tmp/gridfia_cache
```

```python
from gridfia import GridFIA

# Settings loaded from environment
api = GridFIA()
print(f"Debug: {api.settings.debug}")
print(f"Output: {api.settings.output_dir}")
```

### Configuration Files

Settings can be loaded from YAML or JSON files:

=== "YAML Configuration"

    ```yaml
    # config.yaml
    app_name: GridFIA Analysis
    debug: false
    verbose: true
    output_dir: results/
    data_dir: data/

    visualization:
      default_dpi: 300
      default_figure_size: [16, 12]
      color_maps:
        biomass: viridis
        diversity: plasma
        richness: Spectral_r

    processing:
      max_workers: 4
      memory_limit_gb: 16.0

    calculations:
      - name: species_richness
        enabled: true
        parameters:
          biomass_threshold: 0.5
      - name: shannon_diversity
        enabled: true
        output_format: geotiff
      - name: total_biomass
        enabled: true
    ```

=== "JSON Configuration"

    ```json
    {
      "app_name": "GridFIA Analysis",
      "debug": false,
      "verbose": true,
      "output_dir": "results/",
      "calculations": [
        {
          "name": "species_richness",
          "enabled": true,
          "parameters": {"biomass_threshold": 0.5}
        },
        {
          "name": "shannon_diversity",
          "enabled": true
        }
      ]
    }
    ```

## CalculationConfig

Configuration for individual forest metric calculations.

::: gridfia.config.CalculationConfig
    options:
      show_root_heading: false
      heading_level: 3

### Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | (required) | Name of the calculation |
| `enabled` | `bool` | `True` | Whether calculation is enabled |
| `parameters` | `Dict[str, Any]` | `{}` | Calculation-specific parameters |
| `output_format` | `OutputFormat` | `GEOTIFF` | Output format for results |
| `output_name` | `Optional[str]` | `None` | Custom output filename |

### Example

```python
from gridfia.config import CalculationConfig, OutputFormat

# Basic calculation
calc = CalculationConfig(name="species_richness")

# With parameters
calc = CalculationConfig(
    name="species_richness",
    enabled=True,
    parameters={"biomass_threshold": 1.0},
    output_format=OutputFormat.GEOTIFF,
    output_name="richness_map"
)

# Disabled calculation
calc = CalculationConfig(
    name="total_biomass",
    enabled=False
)
```

## ProcessingConfig

Configuration for data processing parameters.

::: gridfia.config.ProcessingConfig
    options:
      show_root_heading: false
      heading_level: 3

### Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_workers` | `Optional[int]` | `None` | Max worker processes (auto-detect) |
| `memory_limit_gb` | `float` | `8.0` | Memory limit in GB |
| `temp_dir` | `Optional[Path]` | `None` | Temporary directory |

### Example

```python
from gridfia.config import ProcessingConfig
from pathlib import Path

# Default processing
config = ProcessingConfig()

# High-performance configuration
config = ProcessingConfig(
    max_workers=8,
    memory_limit_gb=32.0,
    temp_dir=Path("/fast_ssd/tmp")
)
```

## VisualizationConfig

Configuration for visualization parameters.

::: gridfia.config.VisualizationConfig
    options:
      show_root_heading: false
      heading_level: 3

### Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `default_dpi` | `int` | `300` | Default DPI for images (72-600) |
| `default_figure_size` | `Tuple[float, float]` | `(16, 12)` | Figure size in inches |
| `color_maps` | `Dict[str, str]` | (see below) | Default colormaps |
| `font_size` | `int` | `12` | Default font size (8-24) |

Default color maps:

```python
{
    "biomass": "viridis",
    "diversity": "plasma",
    "richness": "Spectral_r"
}
```

### Example

```python
from gridfia.config import VisualizationConfig

# Publication-quality settings
config = VisualizationConfig(
    default_dpi=600,
    default_figure_size=(12, 10),
    font_size=14,
    color_maps={
        "biomass": "YlGn",
        "diversity": "RdYlBu",
        "richness": "Spectral"
    }
)
```

## OutputFormat

Enum for supported output formats.

::: gridfia.config.OutputFormat
    options:
      show_root_heading: false
      heading_level: 3

### Values

| Value | String | Description |
|-------|--------|-------------|
| `GEOTIFF` | `"geotiff"` | GeoTIFF format (best for GIS) |
| `ZARR` | `"zarr"` | Zarr format (best for large outputs) |
| `NETCDF` | `"netcdf"` | NetCDF format (best for xarray) |

### Example

```python
from gridfia.config import CalculationConfig, OutputFormat

# GeoTIFF output (default)
calc = CalculationConfig(
    name="species_richness",
    output_format=OutputFormat.GEOTIFF
)

# Zarr output for large datasets
calc = CalculationConfig(
    name="total_biomass",
    output_format=OutputFormat.ZARR
)

# NetCDF for xarray workflows
calc = CalculationConfig(
    name="shannon_diversity",
    output_format=OutputFormat.NETCDF
)
```

## Helper Functions

### load_settings

::: gridfia.config.load_settings
    options:
      show_root_heading: false
      heading_level: 4

```python
from gridfia.config import load_settings
from pathlib import Path

# Load from YAML file
settings = load_settings(Path("config/production.yaml"))

# Load from JSON file
settings = load_settings(Path("config/settings.json"))

# Load from environment/defaults
settings = load_settings()
```

### save_settings

::: gridfia.config.save_settings
    options:
      show_root_heading: false
      heading_level: 4

```python
from gridfia.config import GridFIASettings, save_settings, CalculationConfig
from pathlib import Path

settings = GridFIASettings(
    output_dir="results/",
    calculations=[
        CalculationConfig(name="species_richness", enabled=True),
        CalculationConfig(name="shannon_diversity", enabled=True),
    ]
)

# Save to JSON file
save_settings(settings, Path("config/my_settings.json"))
```

## Usage Patterns

### Complete Configuration Example

```python
from gridfia import GridFIA, GridFIASettings
from gridfia.config import (
    CalculationConfig,
    ProcessingConfig,
    VisualizationConfig,
    OutputFormat
)
from pathlib import Path

# Create comprehensive settings
settings = GridFIASettings(
    app_name="Forest Diversity Analysis",
    debug=False,
    verbose=True,
    data_dir=Path("data"),
    output_dir=Path("results"),
    cache_dir=Path(".cache"),

    processing=ProcessingConfig(
        max_workers=4,
        memory_limit_gb=16.0
    ),

    visualization=VisualizationConfig(
        default_dpi=300,
        default_figure_size=(16, 12),
        font_size=12
    ),

    calculations=[
        CalculationConfig(
            name="species_richness",
            enabled=True,
            parameters={"biomass_threshold": 0.5},
            output_format=OutputFormat.GEOTIFF,
            output_name="richness"
        ),
        CalculationConfig(
            name="shannon_diversity",
            enabled=True,
            output_format=OutputFormat.GEOTIFF
        ),
        CalculationConfig(
            name="simpson_diversity",
            enabled=True
        ),
        CalculationConfig(
            name="total_biomass",
            enabled=True,
            output_format=OutputFormat.ZARR
        ),
    ]
)

# Use with GridFIA
api = GridFIA(config=settings)
results = api.calculate_metrics("data/forest.zarr")
```

### Dynamic Configuration

```python
from gridfia import GridFIASettings
from gridfia.config import CalculationConfig

# Start with defaults
settings = GridFIASettings()

# Modify settings
settings.output_dir = Path("new_results")
settings.processing.memory_limit_gb = 32.0

# Add calculations dynamically
available_calcs = ["species_richness", "shannon_diversity", "evenness"]
settings.calculations = [
    CalculationConfig(name=calc, enabled=True)
    for calc in available_calcs
]
```

### Environment-Based Configuration

```python
import os
from gridfia import GridFIA, GridFIASettings

# Set environment variables for different environments
if os.getenv("ENVIRONMENT") == "production":
    os.environ["GRIDFIA_OUTPUT_DIR"] = "/data/production/results"
    os.environ["GRIDFIA_DEBUG"] = "false"
else:
    os.environ["GRIDFIA_OUTPUT_DIR"] = "./dev_results"
    os.environ["GRIDFIA_DEBUG"] = "true"

# Settings automatically loaded from environment
api = GridFIA()
```

## Validation

Pydantic automatically validates all configuration values:

```python
from gridfia.config import GridFIASettings, VisualizationConfig
from pydantic import ValidationError

# Invalid DPI (must be 72-600)
try:
    config = VisualizationConfig(default_dpi=1000)
except ValidationError as e:
    print(f"Validation error: {e}")

# Empty calculations list (must have at least 1)
try:
    settings = GridFIASettings(calculations=[])
except ValidationError as e:
    print(f"Validation error: {e}")
```

## See Also

- [GridFIA Class](gridfia.md) - Main API using configuration
- [Data Models](models.md) - Related Pydantic models
- [Calculations](calculations.md) - Available calculation names
