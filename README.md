# GridFIA

**GridFIA** is a spatial raster analysis tool for USDA Forest Service BIGMAP data, providing efficient Zarr-based storage and processing for localized forest biomass analysis.

**Part of the FIA Python Ecosystem:**
- **PyFIA**: Survey/plot data analysis ([github.com/mihiarc/pyfia](https://github.com/mihiarc/pyfia))
- **GridFIA**: Spatial raster analysis (this package)
- **PyFVS**: Growth/yield simulation ([github.com/mihiarc/pyfvs](https://github.com/mihiarc/pyfvs))
- **AskFIA**: AI conversational interface ([github.com/mihiarc/askfia](https://github.com/mihiarc/askfia))

## About BIGMAP

The USDA Forest Service's BIGMAP project provides tree species aboveground biomass estimates at 30-meter resolution across the continental United States. This data, derived from Landsat 8 imagery (2014-2018) and 212,978 FIA plots, represents biomass for 327 individual tree species in tons per acre.

## What This Project Does

GridFIA bridges the gap between the BIGMAP REST API and local analysis by:
- **Converting** raster data from the FIA BIGMAP ImageServer into efficient Zarr stores
- **Enabling** localized analysis for any US state, county, or custom region
- **Providing** ready-to-use tools for calculating forest diversity metrics
- **Optimizing** data access patterns for scientific computing workflows

## Key Features

- **Zarr Storage**: Converts BIGMAP GeoTIFF data into cloud-optimized Zarr arrays for fast local analysis
- **REST API Integration**: Direct access to FIA BIGMAP ImageServer (327 tree species, 30m resolution)
- **Location Flexibility**: Analyze any US state, county, or custom geographic region
- **Analysis Ready**: Pre-configured calculations for diversity indices, biomass totals, and species distributions
- **Performance**: Chunked storage with compression for efficient data access patterns
- **Visualization**: Create publication-ready maps with automatic boundary detection

## Installation

```bash
# Using uv (recommended)
uv venv
uv pip install -e ".[dev]"

# Using pip
pip install -e ".[dev]"
```

## Quick Start

### Python API

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

GridFIA supports analysis for:
- **All 50 US States**: Automatic State Plane CRS detection
- **Counties**: Any US county within a state
- **Custom Regions**: Define your own bounding box
- **Multi-State Regions**: Combine multiple states

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

The main API interface for all GridFIA functionality:

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

- `list_species()` - List available species from BIGMAP
- `download_species()` - Download species data for a location
- `create_zarr()` - Create Zarr store from GeoTIFF files
- `calculate_metrics()` - Run forest metric calculations
- `create_maps()` - Create visualization maps
- `validate_zarr()` - Validate a Zarr store
- `get_location_config()` - Get location configuration

## Integration with FIA Ecosystem

GridFIA works seamlessly with other FIA Python tools:

```python
# Use with PyFIA for survey data
from pyfia import FIAData
from gridfia import GridFIA

# Get species information from PyFIA
fia_data = FIAData()
species_info = fia_data.query_species()

# Use species codes with GridFIA
api = GridFIA()
files = api.download_species(
    state="Oregon",
    species_codes=species_info["species_code"].tolist()
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

## Data Sources

### FIA BIGMAP (2018)
This project accesses the USDA Forest Service FIA BIGMAP Tree Species Aboveground Biomass layers:
- **Resolution**: 30 meters
- **Species**: 327 individual tree species
- **Coverage**: Continental United States
- **Units**: Tons per acre (converted to Mg/ha in processing)
- **Source**: Landsat 8 OLI (2014-2018) + 212,978 FIA plots
- **REST API**: `https://di-usfsdata.img.arcgis.com/arcgis/rest/services/FIA_BIGMAP_2018_Tree_Species_Aboveground_Biomass/ImageServer`

## Citation

If you use GridFIA in your research, please cite:

```bibtex
@software{gridfia2025,
  title = {GridFIA: Spatial Raster Analysis for USDA Forest Service BIGMAP Data},
  year = {2025},
  url = {https://github.com/mihiarc/gridfia}
}
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

## Support

- [Issue Tracker](https://github.com/mihiarc/gridfia/issues)
- [Discussions](https://github.com/mihiarc/gridfia/discussions)
