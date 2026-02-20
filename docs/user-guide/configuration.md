# Configuration

GridFIA uses [Pydantic v2](https://docs.pydantic.dev/) for type-safe configuration management. Settings can be loaded from YAML files, environment variables, or created programmatically.

## Quick Start

```python
from gridfia import GridFIA

# Use default settings
api = GridFIA()

# Load from YAML file
api = GridFIA(config="config/production.yaml")

# Programmatic configuration
from gridfia import GridFIASettings
from gridfia.config import CalculationConfig

settings = GridFIASettings(
    output_dir="results",
    calculations=[
        CalculationConfig(name="species_richness", enabled=True),
        CalculationConfig(name="shannon_diversity", enabled=True),
    ]
)
api = GridFIA(config=settings)
```

## YAML Configuration

Create a YAML file with your analysis settings:

```yaml
# my_analysis.yaml
app_name: GridFIA Analysis
debug: false
verbose: true
output_dir: results/diversity
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
    output_format: geotiff

  - name: shannon_diversity
    enabled: true
    output_format: geotiff

  - name: total_biomass
    enabled: true
    output_format: geotiff
```

Load and use:

```python
from gridfia import GridFIA
from gridfia.config import load_settings
from pathlib import Path

settings = load_settings(Path("my_analysis.yaml"))
api = GridFIA(config=settings)
results = api.calculate_metrics("data/forest.zarr")
```

## Environment Variables

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

# Settings automatically loaded from environment
api = GridFIA()
print(f"Debug: {api.settings.debug}")
print(f"Output: {api.settings.output_dir}")
```

## Saving and Loading Settings

```python
from gridfia.config import load_settings, save_settings, GridFIASettings
from pathlib import Path

# Save current settings to JSON
settings = GridFIASettings(output_dir="results/")
save_settings(settings, Path("config/backup.json"))

# Load from file
restored = load_settings(Path("config/backup.json"))
```

## Dynamic Configuration

```python
from gridfia import GridFIASettings
from gridfia.config import CalculationConfig
from pathlib import Path

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

## Reproducibility

Set a random seed for deterministic results in bootstrap and permutation tests:

```python
# At initialization
api = GridFIA(seed=42)

# Or later
api.set_seed(42)

# Now calculate_metrics_with_stats produces reproducible results
results = api.calculate_metrics_with_stats("data/forest.zarr")
```

## See Also

- [API Configuration Reference](../api/config.md) - Full auto-generated reference for all config classes
- [GridFIA Class](../api/gridfia.md) - Main API interface
