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

## Cloud Storage Configuration

GridFIA supports streaming forest data from cloud storage. The cloud backend is configured via `CloudStorageConfig`:

```python
from gridfia import GridFIA, GridFIASettings
from gridfia.config import CloudStorageConfig, CloudStorageBackend

# Configure cloud storage
cloud = CloudStorageConfig(
    backend=CloudStorageBackend.BACKBLAZE_B2,
    bucket="gridfia-data",
    public_url="https://f004.backblazeb2.com/file/gridfia-data",
)

settings = GridFIASettings(cloud=cloud)
api = GridFIA(config=settings)

# Now load_state and load_from_cloud use this config
store = api.load_state("RI")
```

Cloud settings can also be configured via environment variables:

```bash
export GRIDFIA_CLOUD_BACKEND=b2
export GRIDFIA_CLOUD_BUCKET=gridfia-data
export GRIDFIA_CLOUD_PUBLIC_URL=https://f004.backblazeb2.com/file/gridfia-data
export GRIDFIA_CLOUD_ENDPOINT_URL=https://s3.us-west-004.backblazeb2.com
export GRIDFIA_CLOUD_ACCESS_KEY=your_key       # Optional, for private buckets
export GRIDFIA_CLOUD_SECRET_KEY=your_secret     # Optional, for private buckets
```

### Supported Backends

| Backend | Enum Value | Description |
|---------|-----------|-------------|
| Backblaze B2 | `CloudStorageBackend.BACKBLAZE_B2` | Low-cost S3-compatible storage (default) |
| Cloudflare R2 | `CloudStorageBackend.CLOUDFLARE_R2` | Zero egress fees |
| AWS S3 | `CloudStorageBackend.AWS_S3` | Standard cloud storage |
| HTTP | `CloudStorageBackend.HTTP` | Any publicly accessible HTTP URL |

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
- [Cloud Data Access](cloud-data.md) - Using cloud-hosted datasets
- [GridFIA Class](../api/gridfia.md) - Main API interface
