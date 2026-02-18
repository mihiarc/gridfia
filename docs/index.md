---
title: GridFIA - Python API for Spatial Forest Biomass Analysis
description: GridFIA provides efficient Zarr-based storage and processing for USDA Forest Service BIGMAP forest biomass data. Download, analyze, and visualize 30-meter resolution species biomass across the United States.
---

# GridFIA Documentation

Welcome to GridFIA - a Python API for spatial forest analysis using USDA Forest Service BIGMAP data.

## What is GridFIA?

GridFIA is a user-friendly wrapper that makes it easy to work with [BIGMAP 2018](https://data.fs.usda.gov/geodata/rastergateway/biomass/) forest biomass data. BIGMAP provides 30-meter resolution estimates of tree species biomass across the contiguous United States, and GridFIA gives you a clean Python API to:

- Download species biomass rasters for any state, county, or custom region
- Store data efficiently in cloud-optimized Zarr format
- Calculate diversity metrics (Shannon, Simpson, richness, evenness)
- Generate publication-ready maps and visualizations

## Quick Start

```bash
# Install with uv (recommended)
uv pip install gridfia

# Or with pip
pip install gridfia
```

```python
from gridfia import GridFIA

api = GridFIA()

# Download species data for Montana
files = api.download_species(
    state="Montana",
    species_codes=["0202", "0122"],  # Douglas-fir, Ponderosa pine
    output_dir="downloads/"
)

# Create Zarr store
zarr_path = api.create_zarr("downloads/", "data/montana.zarr")

# Calculate diversity metrics
results = api.calculate_metrics(
    zarr_path,
    calculations=["species_richness", "shannon_diversity"]
)

# Create maps
maps = api.create_maps(zarr_path, map_type="diversity", state="MT")
```

## Key Features

### Simple API

One class with a clean, discoverable interface:

```python
api = GridFIA()

# Core workflow
api.list_species()                # See available species
api.download_species()            # Download raster data
api.create_zarr()                 # Convert to Zarr format
api.calculate_metrics()           # Run forest calculations
api.calculate_metrics_with_stats()  # Metrics with confidence intervals
api.create_maps()                 # Generate visualizations

# Cloud & sample data
api.load_from_cloud()             # Stream data from cloud storage
api.load_state()                  # Load pre-hosted state data
api.download_sample()             # Download sample datasets
api.list_sample_datasets()        # List available samples
api.list_state_datasets()         # List available states

# Configuration & utilities
api.get_location_config()         # Configure geographic extents
api.list_calculations()           # See available metrics
api.validate_zarr()               # Validate data stores
api.set_seed()                    # Set seed for reproducibility
```

### 15+ Forest Metrics

| Category | Metrics |
|----------|---------|
| Diversity | Species richness, Shannon index, Simpson index, Evenness |
| Biomass | Total biomass, Species proportion, Threshold analysis |
| Species | Dominant species, Presence/absence, Rare/common species |

### Cloud-Optimized Storage

GridFIA uses [Zarr](https://zarr.dev/) for efficient storage and processing of large raster datasets with configurable chunking and compression.

### Cloud Data Access

Stream pre-hosted forest data directly from cloud storage -- no downloads required:

```python
# Load a pre-hosted state dataset (streaming, only fetches chunks you access)
store = api.load_state("RI")

# Load a sample dataset for quick testing
store = api.load_from_cloud(sample="durham_nc")
```

### Any Geographic Extent

Download data for any US location:

```python
# Entire state
api.download_species(state="California")

# Specific county
api.download_species(state="Texas", county="Harris")

# Custom bounding box
api.download_species(bbox=(-123.5, 45.0, -122.0, 46.5), crs="EPSG:4326")
```

## Documentation

- **[Installation](getting-started/installation.md)** - Setup and requirements
- **[Quick Start](user-guide/getting-started.md)** - First steps with GridFIA
- **User Guide**
  - [Configuration](user-guide/configuration.md) - Settings and options
  - [Cloud Data Access](user-guide/cloud-data.md) - Stream pre-hosted data
  - [Data Pipeline](user-guide/data-pipeline.md) - Download, convert, and process
- **[API Reference](api/index.md)** - Complete API documentation
- **[Tutorials](tutorials/species-diversity-analysis.md)** - Step-by-step guides
- **[Contributing](contributing.md)** - Development setup and guidelines

## About BIGMAP Data

BIGMAP (Biomass and Carbon Mapping) provides modeled estimates of live tree biomass at 30-meter resolution. The data is derived from:

- FIA plot measurements
- Landsat imagery
- Topographic variables
- Climate data

Species-level biomass estimates are available for 300+ tree species. See the [FIA BIGMAP documentation](https://data.fs.usda.gov/geodata/rastergateway/biomass/) for methodology details.

## Contributing

We welcome contributions! See our [Contributing Guide](contributing.md) for development setup and guidelines, or visit our [GitHub repository](https://github.com/mihiarc/gridfia) to report issues and submit pull requests.

## Learn More

- **[GitHub](https://github.com/mihiarc/gridfia)** - Source code and issue tracker
- **[PyPI](https://pypi.org/project/gridfia/)** - Package installation

## License

GridFIA is released under the MIT License.

---

<div align="center">
<sub>Built by <a href="https://github.com/mihiarc">Chris Mihiar</a> · USDA Forest Service Southern Research Station</sub>
</div>
