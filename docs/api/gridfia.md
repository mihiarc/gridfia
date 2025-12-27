# GridFIA API

The `GridFIA` class is the primary interface for all GridFIA functionality. It provides methods
for downloading species data, creating Zarr stores, calculating forest metrics, and generating
visualizations.

## Overview

GridFIA follows an API-first design pattern, providing a single clean interface that:

- Downloads species biomass rasters from the FIA BIGMAP service
- Converts GeoTIFF files to cloud-optimized Zarr arrays
- Calculates diversity metrics (Shannon, Simpson, richness)
- Generates publication-ready maps and visualizations

## Quick Start

```python
from gridfia import GridFIA

# Initialize the API
api = GridFIA()

# Download species data for a state
files = api.download_species(
    state="Montana",
    species_codes=["0202", "0122"],  # Douglas-fir, Ponderosa pine
    output_dir="downloads/montana"
)

# Create a Zarr store from downloaded data
zarr_path = api.create_zarr(
    input_dir="downloads/montana",
    output_path="data/montana.zarr"
)

# Calculate forest metrics
results = api.calculate_metrics(
    zarr_path,
    calculations=["species_richness", "shannon_diversity"]
)

# Create visualization maps
maps = api.create_maps(
    zarr_path,
    map_type="diversity",
    state="MT"
)
```

## Class Reference

::: gridfia.api.GridFIA
    options:
      show_root_heading: true
      show_source: false
      members_order: source
      heading_level: 3
      show_signature_annotations: true
      separate_signature: true

## Method Details

### Downloading Species Data

The `download_species` method supports multiple ways to specify the geographic extent:

=== "By State"

    ```python
    # Download all species for an entire state
    files = api.download_species(
        state="California",
        output_dir="data/california"
    )

    # Download specific species
    files = api.download_species(
        state="Oregon",
        species_codes=["0202", "0122", "0015"],
        output_dir="data/oregon"
    )
    ```

=== "By County"

    ```python
    # Download for a specific county
    files = api.download_species(
        state="North Carolina",
        county="Wake",
        species_codes=["0131", "0068"],
        output_dir="data/wake_county"
    )
    ```

=== "By Bounding Box"

    ```python
    # Download for a custom bounding box (WGS84)
    files = api.download_species(
        bbox=(-123.5, 45.0, -122.0, 46.5),
        crs="4326",
        output_dir="data/custom_region"
    )
    ```

=== "From Config File"

    ```python
    # Use a pre-configured location
    files = api.download_species(
        location_config="config/montana_study_area.yaml",
        output_dir="data/study_area"
    )
    ```

### Working with Zarr Stores

The `create_zarr` method converts GeoTIFF files to cloud-optimized Zarr arrays:

```python
# Basic usage
zarr_path = api.create_zarr(
    input_dir="downloads/",
    output_path="data/forest.zarr"
)

# With custom chunking for large datasets
zarr_path = api.create_zarr(
    input_dir="downloads/",
    output_path="data/forest.zarr",
    chunk_size=(1, 2000, 2000),  # Larger chunks for faster reads
    compression="zstd",
    compression_level=3
)

# Validate the created store
info = api.validate_zarr(zarr_path)
print(f"Shape: {info['shape']}")
print(f"Species count: {info['num_species']}")
print(f"CRS: {info['crs']}")
```

### Calculating Metrics

GridFIA provides 15+ forest metrics through the calculation registry:

```python
# List all available calculations
calcs = api.list_calculations()
print(f"Available: {calcs}")
# ['species_richness', 'shannon_diversity', 'simpson_diversity',
#  'evenness', 'total_biomass', 'dominant_species', ...]

# Run specific calculations
results = api.calculate_metrics(
    zarr_path,
    calculations=["species_richness", "shannon_diversity", "total_biomass"],
    output_dir="output/metrics"
)

# Process results
for result in results:
    print(f"{result.name}: {result.output_path}")
```

### Creating Visualizations

Generate publication-ready maps with various options:

```python
# Species biomass map
maps = api.create_maps(
    zarr_path,
    map_type="species",
    species=["0202"],
    state="MT",
    dpi=300
)

# Diversity map with basemap
maps = api.create_maps(
    zarr_path,
    map_type="diversity",
    state="MT",
    basemap="CartoDB.Positron"
)

# Species comparison (side-by-side)
maps = api.create_maps(
    zarr_path,
    map_type="comparison",
    species=["0202", "0122", "0116"]
)

# All species in store
maps = api.create_maps(
    zarr_path,
    map_type="species",
    show_all=True,
    output_dir="maps/all_species"
)
```

## Configuration

The `GridFIA` class accepts an optional configuration:

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

## Thread Safety

The `GridFIA` class is thread-safe. Internal components (REST client, processor) use
double-checked locking for lazy initialization:

```python
from concurrent.futures import ThreadPoolExecutor
from gridfia import GridFIA

api = GridFIA()

def process_state(state):
    files = api.download_species(state=state, output_dir=f"data/{state}")
    return len(files)

# Safe to use from multiple threads
with ThreadPoolExecutor(max_workers=4) as executor:
    results = executor.map(process_state, ["MT", "WY", "ID", "WA"])
```

## Error Handling

All methods raise domain-specific exceptions for clear error handling:

```python
from gridfia import GridFIA
from gridfia.exceptions import (
    InvalidZarrStructure,
    SpeciesNotFound,
    CalculationFailed,
    APIConnectionError,
    InvalidLocationConfig,
    DownloadError,
)

api = GridFIA()

try:
    files = api.download_species(state="Montana", species_codes=["9999"])
except SpeciesNotFound as e:
    print(f"Species not found: {e.species_code}")

try:
    results = api.calculate_metrics("invalid.zarr")
except InvalidZarrStructure as e:
    print(f"Invalid Zarr store: {e.zarr_path}")

try:
    api.list_species()
except APIConnectionError as e:
    print(f"API error (status {e.status_code}): {e.message}")
```

## See Also

- [Data Models](models.md) - `SpeciesInfo` and `CalculationResult` classes
- [Configuration](config.md) - `GridFIASettings` and related classes
- [Exceptions](exceptions.md) - Exception hierarchy
- [Calculations](calculations.md) - Available forest metrics
