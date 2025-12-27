<div align="center">
  <a href="https://fiatools.org"><img src="https://fiatools.org/logos/gridfia_logo.png" alt="gridFIA" width="400"></a>

  <p><strong>Spatial raster analysis for USDA Forest Service BIGMAP data</strong></p>

  <p>
    <a href="https://fiatools.org"><img src="https://img.shields.io/badge/FIAtools-Ecosystem-2E7D32" alt="FIAtools Ecosystem"></a>
    <a href="https://pypi.org/project/gridfia/"><img src="https://img.shields.io/pypi/v/gridfia?color=006D6D&label=PyPI" alt="PyPI"></a>
    <a href="https://pypi.org/project/gridfia/"><img src="https://img.shields.io/pypi/dm/gridfia?color=006D6D&label=Downloads" alt="PyPI Downloads"></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-006D6D" alt="License: MIT"></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9+-006D6D" alt="Python 3.9+"></a>
    <a href="https://mihiarc.github.io/gridfia/"><img src="https://img.shields.io/badge/docs-GitHub%20Pages-006D6D" alt="Documentation"></a>
  </p>

  <p>
    <strong>Part of the <a href="https://fiatools.org">FIAtools Python Ecosystem</a></strong><br>
    <a href="https://fiatools.org/tools/pyfia/">pyFIA</a> ·
    <a href="https://fiatools.org/tools/gridfia/">gridFIA</a> ·
    <a href="https://fiatools.org/tools/pyfvs/">pyFVS</a> ·
    <a href="https://fiatools.org/tools/askfia/">askFIA</a>
  </p>
</div>

---

GridFIA provides efficient Zarr-based storage and processing for localized forest biomass analysis using USDA Forest Service BIGMAP data.

## About BIGMAP

[BIGMAP](https://data.fs.usda.gov/geodata/rastergateway/bigmap/index.php) (FIA Tree Species Aboveground Biomass Layers) provides tree species biomass estimates at 30-meter resolution across the continental United States.

| Attribute | Value |
|-----------|-------|
| **Resolution** | 30 meters |
| **Species** | 327 individual tree species + total biomass |
| **Coverage** | Coterminous United States (CONUS) |
| **Data Year** | 2018 |
| **Units** | Tons per acre |
| **Source Data** | Landsat 8 OLI (2014-2018) + 212,978 FIA plots |

The methodology uses harmonic regression to characterize vegetation phenology from Landsat time series imagery, then K-nearest neighbors imputation to associate pixels with similar FIA plots based on ecological gradients across 36 ecological provinces.

> Wilson, B.T., Knight, J.F., and McRoberts, R.E., 2018. "Harmonic regression of Landsat time series for modeling attributes from national forest inventory data." *ISPRS Journal of Photogrammetry and Remote Sensing*, 137: 29-46.

## What GridFIA Does

- **Converts** BIGMAP GeoTIFF data into cloud-optimized Zarr arrays
- **Enables** localized analysis for any US state, county, or custom region
- **Calculates** forest diversity metrics (Shannon, Simpson, richness)
- **Optimizes** data access patterns for scientific computing workflows
- **Visualizes** publication-ready maps with automatic boundary detection

## Installation

```bash
# Using uv (recommended)
uv venv
uv pip install -e ".[dev]"

# Using pip
pip install -e ".[dev]"
```

## Quick Start

```python
from gridfia import GridFIA

# Initialize API
api = GridFIA()

# List available species
species = api.list_species()

# Download species data for a location
files = api.download_species(
    state="North Carolina",
    county="Wake",
    species_codes=["0131", "0068"],  # Loblolly Pine, Red Maple
    output_dir="data/wake"
)

# Create Zarr store from downloaded data
zarr_path = api.create_zarr(
    input_dir="data/wake",
    output_path="data/wake_forest.zarr"
)

# Calculate forest metrics
results = api.calculate_metrics(
    zarr_path=zarr_path,
    calculations=["species_richness", "shannon_diversity", "total_biomass"]
)

# Create visualization maps
maps = api.create_maps(
    zarr_path=zarr_path,
    map_type="diversity",
    output_dir="maps/"
)
```

### Using Bounding Boxes

```python
from gridfia import GridFIA

api = GridFIA()

# Download using explicit bounding box (Web Mercator)
files = api.download_species(
    bbox=(-8792000, 4274000, -8732000, 4334000),
    crs="3857",
    species_codes=["0131"],
    output_dir="data/custom"
)
```

## Supported Locations

- **All 50 US States** with automatic State Plane CRS detection
- **Any US County** within a state
- **Custom Regions** via bounding box
- **Multi-State Regions** by combining multiple states

## Available Calculations

| Calculation | Description | Units |
|------------|-------------|--------|
| `species_richness` | Number of tree species per pixel | count |
| `shannon_diversity` | Shannon diversity index | index |
| `simpson_diversity` | Simpson diversity index | index |
| `evenness` | Pielou's evenness (J) | ratio |
| `total_biomass` | Total biomass across all species | Mg/ha |
| `dominant_species` | Most abundant species by biomass | species_id |
| `species_proportion` | Proportion of specific species | ratio |

## API Reference

### GridFIA Class

```python
from gridfia import GridFIA
from gridfia.config import GridFIASettings, CalculationConfig

# Initialize with default settings
api = GridFIA()

# Initialize with custom settings
settings = GridFIASettings(
    output_dir=Path("output"),
    calculations=[
        CalculationConfig(name="species_richness", enabled=True),
        CalculationConfig(name="shannon_diversity", enabled=True)
    ]
)
api = GridFIA(config=settings)
```

### Methods

| Method | Description |
|--------|-------------|
| `list_species()` | List available species from BIGMAP |
| `download_species()` | Download species data for a location |
| `create_zarr()` | Create Zarr store from GeoTIFF files |
| `calculate_metrics()` | Run forest metric calculations |
| `create_maps()` | Create visualization maps |
| `validate_zarr()` | Validate a Zarr store |
| `get_location_config()` | Get location configuration |

## Integration with pyFIA

```python
from pyfia import FIA
from gridfia import GridFIA

# Get species information from pyFIA
with FIA() as fia:
    species_info = fia.species()

# Use species codes with GridFIA
api = GridFIA()
files = api.download_species(
    state="Oregon",
    species_codes=species_info["spcd"].tolist()
)
```

## Development

```bash
# Run tests
uv run pytest

# Format code
uv run black gridfia/
uv run isort gridfia/

# Type checking
uv run mypy gridfia/

# Build documentation
uv run mkdocs serve
```

## The FIAtools Ecosystem

GridFIA is part of the [FIAtools Python ecosystem](https://fiatools.org) - a unified suite of open-source tools for forest inventory analysis:

| Tool | Purpose | Key Features |
|------|---------|--------------|
| [**pyFIA**](https://fiatools.org) | Survey & plot data | DuckDB backend, 10-100x faster than EVALIDator |
| [**gridFIA**](https://fiatools.org) | Spatial raster analysis | 327 species at 30m resolution, Zarr storage |
| [**pyFVS**](https://fiatools.org) | Growth simulation | Chapman-Richards curves, yield projections |
| [**askFIA**](https://fiatools.org) | AI interface | Natural language queries for forest data |

**[Explore the full ecosystem at fiatools.org](https://fiatools.org)**

## Citation

```bibtex
@software{gridfia2025,
  title = {GridFIA: Spatial Raster Analysis for USDA Forest Service BIGMAP Data},
  author = {Mihiar, Christopher},
  year = {2025},
  url = {https://fiatools.org}
}
```

---

<div align="center">
  <a href="https://fiatools.org"><strong>fiatools.org</strong></a> · Python Ecosystem for Forest Inventory Analysis<br>
  <sub>Built by <a href="https://github.com/mihiarc">Chris Mihiar</a> · USDA Forest Service Southern Research Station</sub>
</div>
