---
title: GridFIA - Python API for Spatial Forest Biomass Analysis
description: GridFIA provides efficient Zarr-based storage and processing for USDA Forest Service BIGMAP forest biomass data. Download, analyze, and visualize 30-meter resolution species biomass across the United States.
---

# GridFIA Documentation

Welcome to GridFIA - a Python API for spatial forest analysis using USDA Forest Service BIGMAP data.

!!! tip "Part of the FIAtools Ecosystem"
    GridFIA is one of four integrated Python tools for forest inventory analysis. Visit **[fiatools.org](https://fiatools.org)** to explore the complete ecosystem and see how the tools work together.

## What is GridFIA?

GridFIA is a user-friendly wrapper that makes it easy to work with [BIGMAP 2018](https://data.fs.usda.gov/geodata/rastergateway/biomass/) forest biomass data. BIGMAP provides 30-meter resolution estimates of tree species biomass across the contiguous United States, and GridFIA gives you a clean Python API to:

- Download species biomass rasters for any state, county, or custom region
- Store data efficiently in cloud-optimized Zarr format
- Calculate diversity metrics (Shannon, Simpson, richness, evenness)
- Generate publication-ready maps and visualizations

## The FIAtools Python Ecosystem

| Tool | Purpose | Key Features |
|------|---------|--------------|
| [**pyFIA**](https://fiatools.org) | Survey & plot data | DuckDB backend, 10-100x faster than EVALIDator |
| [**gridFIA**](https://fiatools.org) | Spatial raster analysis | 327 species at 30m resolution, Zarr storage |
| [**pyFVS**](https://fiatools.org) | Growth simulation | Chapman-Richards curves, yield projections |
| [**askFIA**](https://fiatools.org) | AI interface | Natural language queries for forest data |

[:material-arrow-right: Explore the full ecosystem at fiatools.org](https://fiatools.org){ .md-button .md-button--primary }

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

One class, eight methods - that's all you need:

```python
api = GridFIA()
api.list_species()        # See available species
api.download_species()    # Download raster data
api.create_zarr()         # Convert to Zarr format
api.calculate_metrics()   # Run forest calculations
api.create_maps()         # Generate visualizations
api.get_location_config() # Configure geographic extents
api.list_calculations()   # See available metrics
api.validate_zarr()       # Validate data stores
```

### 15+ Forest Metrics

| Category | Metrics |
|----------|---------|
| Diversity | Species richness, Shannon index, Simpson index, Evenness |
| Biomass | Total biomass, Species proportion, Threshold analysis |
| Species | Dominant species, Presence/absence, Rare/common species |

### Cloud-Optimized Storage

GridFIA uses [Zarr](https://zarr.dev/) for efficient storage and processing of large raster datasets with configurable chunking and compression.

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

- **[Getting Started](user-guide/getting-started.md)** - Installation and first steps
- **[API Reference](api/index.md)** - Complete API documentation
  - [GridFIA Class](api/gridfia.md) - Main API interface
  - [Data Models](api/models.md) - SpeciesInfo, CalculationResult
  - [Configuration](api/config.md) - Settings and options
  - [Calculations](api/calculations.md) - Available metrics
- **[Tutorials](tutorials/species-diversity-analysis.md)** - Step-by-step guides

## About BIGMAP Data

BIGMAP (Biomass and Carbon Mapping) provides modeled estimates of live tree biomass at 30-meter resolution. The data is derived from:

- FIA plot measurements
- Landsat imagery
- Topographic variables
- Climate data

Species-level biomass estimates are available for 300+ tree species. See the [FIA BIGMAP documentation](https://data.fs.usda.gov/geodata/rastergateway/biomass/) for methodology details.

## Contributing

We welcome contributions! See our [GitHub repository](https://github.com/mihiarc/gridfia) to:

- Report issues
- Submit pull requests
- Request features

## Learn More

- **[fiatools.org](https://fiatools.org)** - Explore the complete FIA Python ecosystem
- **[GitHub](https://github.com/mihiarc/gridfia)** - Source code and issue tracker
- **[PyPI](https://pypi.org/project/gridfia/)** - Package installation

## License

GridFIA is released under the MIT License.

---

<div align="center">
<strong><a href="https://fiatools.org">fiatools.org</a></strong> · Python Ecosystem for Forest Inventory Analysis
</div>
