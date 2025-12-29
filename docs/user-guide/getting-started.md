# Getting Started with GridFIA

GridFIA is a Python API for analyzing forest biomass and species diversity using USDA Forest Service BIGMAP 2018 data at 30-meter resolution for any US state, county, or custom region.

## Installation

### Using pip

```bash
pip install gridfia
```

### Using uv (recommended)

```bash
uv pip install gridfia
```

### Development Installation

```bash
git clone https://github.com/mihiarc/gridfia.git
cd gridfia
uv pip install -e ".[dev,test,docs]"
```

## Quick Start

### 1. Initialize the API

```python
from gridfia import GridFIA

api = GridFIA()
```

### 2. List Available Species

BIGMAP provides biomass data for 300+ tree species. List them all:

```python
# Get all available species from BIGMAP
species = api.list_species()

# Display first 10 species
for s in species[:10]:
    print(f"{s.species_code}: {s.common_name} ({s.scientific_name})")
```

### 3. Download Species Data

Download biomass rasters for specific species and locations:

```python
# Download for an entire state
files = api.download_species(
    state="North Carolina",
    species_codes=["0131", "0068"],  # Loblolly Pine, Eastern White Pine
    output_dir="data/nc"
)
print(f"Downloaded {len(files)} files")

# Download for a specific county
files = api.download_species(
    state="North Carolina",
    county="Wake",
    species_codes=["0131", "0068"],
    output_dir="data/wake"
)

# Download with custom bounding box (WGS84)
files = api.download_species(
    bbox=(-79.5, 35.5, -78.5, 36.5),
    crs="EPSG:4326",
    species_codes=["0131"],
    output_dir="data/custom"
)
```

### 4. Create Zarr Store

Convert downloaded GeoTIFF files to cloud-optimized Zarr format:

```python
# Create Zarr store from downloaded rasters
zarr_path = api.create_zarr(
    input_dir="data/wake",
    output_path="data/wake_forest.zarr"
)

# Validate the created store
info = api.validate_zarr(zarr_path)
print(f"Species: {info['num_species']}")
print(f"Shape: {info['shape']}")
print(f"CRS: {info['crs']}")
```

### 5. Calculate Forest Metrics

Run diversity and biomass calculations:

```python
# List available calculations
calculations = api.list_calculations()
print(f"Available: {calculations}")

# Run specific calculations
results = api.calculate_metrics(
    zarr_path="data/wake_forest.zarr",
    calculations=["species_richness", "shannon_diversity", "total_biomass"],
    output_dir="output/metrics"
)

# View results
for result in results:
    print(f"{result.name}: {result.output_path}")
```

### 6. Create Visualizations

Generate publication-ready maps:

```python
# Create diversity map
maps = api.create_maps(
    zarr_path="data/wake_forest.zarr",
    map_type="diversity",
    output_dir="output/maps"
)

# Create species biomass map
maps = api.create_maps(
    zarr_path="data/wake_forest.zarr",
    map_type="species",
    species=["0131"],  # Loblolly Pine
    output_dir="output/maps"
)
```

## Complete Example

Here's a full workflow from download to visualization:

```python
from gridfia import GridFIA
from pathlib import Path

# Initialize API
api = GridFIA()

# Define species of interest
pine_species = ["0131", "0110", "0132"]  # Loblolly, Shortleaf, Longleaf

# Download data for Wake County, NC
files = api.download_species(
    state="North Carolina",
    county="Wake",
    species_codes=pine_species,
    output_dir="tutorial_data"
)
print(f"Downloaded {len(files)} species files")

# Create Zarr store
zarr_path = api.create_zarr(
    input_dir="tutorial_data",
    output_path="tutorial_data/wake_pines.zarr"
)

# Calculate diversity metrics
results = api.calculate_metrics(
    zarr_path=zarr_path,
    calculations=[
        "species_richness",
        "shannon_diversity",
        "simpson_diversity",
        "total_biomass"
    ],
    output_dir="output"
)

# Create maps
maps = api.create_maps(
    zarr_path=zarr_path,
    map_type="diversity",
    output_dir="output/maps"
)

print("Analysis complete!")
```

## Using Sample Datasets

GridFIA provides pre-hosted sample datasets for quick testing:

```python
from gridfia import GridFIA

api = GridFIA()

# List available sample datasets
samples = api.list_sample_datasets()
for sample in samples:
    print(f"{sample['name']}: {sample['description']}")

# Download a sample dataset
zarr_path = api.download_sample(
    name="wake_county_nc",
    output_dir="samples"
)

# Or load directly from cloud (no download)
store = api.load_from_cloud(
    url="https://data.example.com/samples/wake_county.zarr"
)
```

## Configuration

### Programmatic Configuration

```python
from gridfia import GridFIA
from gridfia.config import GridFIASettings, CalculationConfig
from pathlib import Path

# Create custom settings
settings = GridFIASettings(
    output_dir=Path("results"),
    calculations=[
        CalculationConfig(name="species_richness", enabled=True),
        CalculationConfig(name="shannon_diversity", enabled=True),
        CalculationConfig(
            name="total_biomass",
            enabled=True,
            output_format="geotiff"
        ),
    ]
)

# Initialize API with custom settings
api = GridFIA(config=settings)
```

### Load Settings from YAML

```python
from gridfia import GridFIA
from gridfia.config import load_settings
from pathlib import Path

# Load settings from file
settings = load_settings(Path("config/my_analysis.yaml"))

# Use with API
api = GridFIA(config=settings)
```

Example YAML configuration:

```yaml
# my_analysis.yaml
debug: false
verbose: true
output_dir: results/diversity

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

## Available Calculations

| Calculation | Description | Units |
|-------------|-------------|-------|
| `species_richness` | Number of species per pixel | count |
| `shannon_diversity` | Shannon diversity index (H') | index |
| `simpson_diversity` | Simpson diversity index | index |
| `evenness` | Pielou's evenness (J) | ratio |
| `total_biomass` | Total biomass across all species | Mg/ha |
| `dominant_species` | Most abundant species by biomass | species_id |
| `species_proportion` | Proportion of specific species | ratio |
| `species_presence` | Binary presence of species | binary |
| `biomass_threshold` | Areas above biomass threshold | binary |

## Next Steps

- [API Reference](../api/index.md) - Complete API documentation
- [Tutorials](../tutorials/species-diversity-analysis.md) - Step-by-step guides
- [Configuration](../api/config.md) - Advanced configuration options
