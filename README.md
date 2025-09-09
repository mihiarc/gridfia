# BigMap Zarr

🌲 **BigMap Zarr** makes USDA Forest Service FIA BIGMAP data analysis-ready by providing efficient Zarr-based storage and processing tools for localized forest biomass analysis.

## About BIGMAP

The USDA Forest Service's BIGMAP project provides tree species aboveground biomass estimates at 30-meter resolution across the continental United States. This data, derived from Landsat 8 imagery (2014-2018) and 212,978 FIA plots, represents biomass for 327 individual tree species in tons per acre.

## What This Project Does

BigMap Zarr bridges the gap between the BIGMAP REST API and local analysis by:
- **Converting** raster data from the FIA BIGMAP ImageServer into efficient Zarr stores
- **Enabling** localized analysis for any US state, county, or custom region
- **Providing** ready-to-use tools for calculating forest diversity metrics
- **Optimizing** data access patterns for scientific computing workflows

## Key Features

- 📦 **Zarr Storage**: Converts BIGMAP GeoTIFF data into cloud-optimized Zarr arrays for fast local analysis
- 🌐 **REST API Integration**: Direct access to FIA BIGMAP ImageServer (327 tree species, 30m resolution)
- 📍 **Location Flexibility**: Analyze any US state, county, or custom geographic region
- 📊 **Analysis Ready**: Pre-configured calculations for diversity indices, biomass totals, and species distributions
- 🚀 **Performance**: Chunked storage with compression for efficient data access patterns
- 🗺️ **Visualization**: Create publication-ready maps with automatic boundary detection

## Installation

```bash
# Using uv (recommended)
uv venv
uv pip install -e ".[dev]"

# Using pip
pip install -e ".[dev]"
```

## How It Works

BigMap Zarr transforms BIGMAP REST API data into analysis-ready Zarr stores:

```
BIGMAP ImageServer → GeoTIFF Downloads → Zarr Arrays → Local Analysis
        ↓                    ↓                ↓              ↓
   (REST API)         (Species Rasters)  (Chunked Storage) (Fast Access)
```

The Zarr format provides:
- **Chunked arrays** for partial data loading
- **Compression** to reduce storage requirements  
- **Parallel access** for multi-threaded processing
- **Metadata preservation** for CRS and spatial info

## Quick Start

### 1. Create Location Configuration

```bash
# For a state
bigmap location create --state California --output california.yaml

# For a county
bigmap location create --state Texas --county Harris --output houston.yaml

# For a custom region (bbox in WGS84)
bigmap location create --bbox "-104.5,44.0,-104.0,44.5" --output custom.yaml

# List all available states
bigmap location list
```

### 2. Download Species Data

```bash
# Download for a specific state
bigmap download --state "North Carolina" --species 0131 --species 0068

# Download using location config
bigmap download --location-config california.yaml --output data/california/

# Download for custom bbox
bigmap download --bbox "-9200000,4000000,-8400000,4400000" --crs 3857
```

### 3. Build Zarr Store

```bash
# Build from downloaded GeoTIFFs
bigmap build-zarr --data-dir downloads/ --output forest.zarr

# Specify chunk size and compression
bigmap build-zarr --chunk-size "1,1000,1000" --compression lz4
```

### 4. Calculate Forest Metrics

```bash
# Run all calculations
bigmap calculate forest.zarr --config config.yaml

# Run specific calculations
bigmap calculate forest.zarr --calc shannon_diversity --calc species_richness

# List available calculations
bigmap calculate forest.zarr --list
```

### 5. Create Maps

```bash
# Create species distribution maps
bigmap map forest.zarr --type species --species 0131 --state NC

# Create diversity maps
bigmap map forest.zarr --type diversity --output maps/

# Create richness map with basemap
bigmap map forest.zarr --type richness --basemap CartoDB
```

## Supported Locations

BigMap supports analysis for:
- **All 50 US States**: Automatic State Plane CRS detection
- **Counties**: Any US county within a state
- **Custom Regions**: Define your own bounding box
- **Multi-State Regions**: Combine multiple states

### Example State Configurations

Pre-configured templates are available for:
- North Carolina (`config/templates/north_carolina.yaml`)
- Texas (`config/templates/texas.yaml`)
- California (`config/templates/california.yaml`)
- Montana (`config/montana_project.yml`)

## Available Calculations

| Calculation | Description | Units |
|------------|-------------|--------|
| `species_richness` | Number of tree species per pixel | count |
| `shannon_diversity` | Shannon diversity index | index |
| `simpson_diversity` | Simpson diversity index | index |
| `total_biomass` | Total biomass across all species | Mg/ha |
| `dominant_species` | Most abundant species by biomass | species_id |
| `species_proportion` | Proportion of specific species | ratio |

## API Reference

### Python API

```python
from bigmap.utils.location_config import LocationConfig
from bigmap.api import BigMapRestClient
from bigmap.core.processors import ForestMetricsProcessor

# Create configuration for any state
config = LocationConfig.from_state("Oregon")

# Download species data
client = BigMapRestClient()
client.batch_export_location_species(
    bbox=config.web_mercator_bbox,
    output_dir=Path("data/oregon"),
    location_name="oregon"
)

# Process metrics
processor = ForestMetricsProcessor(settings)
results = processor.run_calculations("oregon.zarr")
```

### CLI Commands

- `bigmap location` - Manage location configurations
- `bigmap download` - Download species data from REST API
- `bigmap build-zarr` - Build Zarr store from GeoTIFFs
- `bigmap calculate` - Run forest metric calculations
- `bigmap map` - Create visualization maps
- `bigmap list-species` - List available species codes

## Configuration

### Location Configuration Structure

```yaml
location:
  type: state        # state, county, or custom
  name: California
  abbreviation: CA

crs:
  target: EPSG:26943  # Auto-detected State Plane CRS

bounding_boxes:
  wgs84:             # Latitude/Longitude
  state_plane:       # State-specific projection
  web_mercator:      # For web mapping

species:             # Species of interest
  - code: '0202'
    name: Douglas-fir
```

## Development

```bash
# Run tests
uv run pytest

# Format code
uv run black bigmap/
uv run isort bigmap/

# Type checking
uv run mypy bigmap/

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
- **REST API**: `https://apps.fs.usda.gov/arcx/rest/services/RDW_Biomass/`

### Data Processing Pipeline
1. **Download**: Fetch species-specific rasters via BIGMAP ImageServer REST API
2. **Convert**: Transform GeoTIFF files into chunked Zarr arrays
3. **Analyze**: Apply forest metrics calculations on local Zarr stores
4. **Visualize**: Generate maps and statistics for regions of interest

## Citation

If you use BigMap in your research, please cite:

```bibtex
@software{bigmap2024,
  title = {BigMap: Forest Biomass and Diversity Analysis Toolkit},
  year = {2024},
  url = {https://github.com/yourusername/bigmap-zarr}
}
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

## Support

- 📖 [Documentation](https://yourdocs.com)
- 🐛 [Issue Tracker](https://github.com/yourusername/bigmap-zarr/issues)
- 💬 [Discussions](https://github.com/yourusername/bigmap-zarr/discussions)