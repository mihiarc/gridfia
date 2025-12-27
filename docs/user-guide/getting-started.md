# Getting Started with GridFIA

GridFIA is a modern Python framework for analyzing forest biomass and species diversity using BIGMAP 2018 data for North Carolina.

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

### 1. Download Species Data

First, download some species data from the FIA BIGMAP REST API:

```bash
# List available species
gridfia list-species

# Download default NC species
gridfia download --output data/

# Download specific species
gridfia download --species 0131 --species 0068 --output data/
```

### 2. Create Zarr Array

Convert downloaded GeoTIFF files to a zarr array for efficient processing:

```python
import zarr
import rasterio
import numpy as np
from pathlib import Path

# Create zarr array from species rasters
def create_zarr_from_rasters(raster_dir, output_path):
    raster_files = sorted(Path(raster_dir).glob("*.tif"))
    
    # Read first raster for dimensions
    with rasterio.open(raster_files[0]) as src:
        height, width = src.shape
        transform = src.transform
        crs = src.crs
    
    # Create zarr array
    z = zarr.open_array(
        output_path,
        mode='w',
        shape=(len(raster_files), height, width),
        chunks=(1, 1000, 1000),
        dtype='f4'
    )
    
    # Load each species
    species_codes = []
    for i, raster_file in enumerate(raster_files):
        with rasterio.open(raster_file) as src:
            z[i] = src.read(1)
            species_codes.append(raster_file.stem)
    
    # Add metadata
    z.attrs.update({
        'species_codes': species_codes,
        'crs': str(crs),
        'transform': list(transform),
        'units': 'Mg/ha'
    })
    
    return output_path

# Create zarr
zarr_path = create_zarr_from_rasters("data/", "data/nc_biomass.zarr")
```

### 3. Run Calculations

Use the CLI to run forest metric calculations:

```bash
# List available calculations
gridfia calculate data/nc_biomass.zarr --list

# Run specific calculations
gridfia calculate data/nc_biomass.zarr \
    --calc species_richness \
    --calc total_biomass \
    --output results/

# Use a configuration file
gridfia calculate data/nc_biomass.zarr --config config.yaml
```

### 4. View Results

The calculations produce GeoTIFF files that can be viewed in GIS software:

```python
import rasterio
import matplotlib.pyplot as plt

# View species richness map
with rasterio.open("results/species_richness.tif") as src:
    richness = src.read(1)
    
plt.figure(figsize=(10, 8))
plt.imshow(richness, cmap='viridis')
plt.colorbar(label='Number of Species')
plt.title('Forest Species Richness')
plt.show()
```

## Configuration

### Create a Configuration File

```bash
# Create a diversity analysis configuration
gridfia config create --template diversity --output my_config.yaml

# Validate configuration
gridfia config validate --config my_config.yaml

# Show current configuration
gridfia config show
```

### Configuration Example

```yaml
# my_config.yaml
app_name: NC Forest Analysis
output_dir: results/diversity_analysis

calculations:
  - name: species_richness
    enabled: true
    parameters:
      biomass_threshold: 0.5
    output_format: geotiff
    
  - name: shannon_diversity
    enabled: true
    output_format: netcdf
    output_name: shannon_index
    
  - name: dominant_species
    enabled: true
    
  - name: total_biomass
    enabled: true
    output_format: zarr
```

## Python API Usage

### Basic Example

```python
from gridfia.config import GridFIASettings, CalculationConfig
from gridfia.core.processors.forest_metrics import ForestMetricsProcessor

# Configure settings
settings = GridFIASettings(
    output_dir="results",
    calculations=[
        CalculationConfig(name="species_richness", enabled=True),
        CalculationConfig(name="shannon_diversity", enabled=True)
    ]
)

# Run analysis
processor = ForestMetricsProcessor(settings)
results = processor.run_calculations("data/nc_biomass.zarr")

print(f"Completed calculations: {list(results.keys())}")
```

### Advanced Example with Custom Parameters

```python
from gridfia.core.calculations import registry
import numpy as np

# Load zarr data
import zarr
z = zarr.open_array("data/nc_biomass.zarr", mode='r')
biomass_data = z[:]

# Get calculation with custom parameters
richness_calc = registry.get(
    'species_richness', 
    biomass_threshold=1.0,
    exclude_total_layer=True
)

# Run calculation
richness_map = richness_calc.calculate(biomass_data)

# Get metadata
metadata = richness_calc.get_metadata()
print(f"Calculation: {metadata['description']}")
print(f"Units: {metadata['units']}")
```

## Next Steps

- [CLI Reference](../cli-reference.md) - Detailed CLI documentation
- [API Reference](../api/index.md) - Complete API documentation
- [Tutorials](../tutorials/index.md) - Step-by-step tutorials
- [Examples](../examples/index.md) - Example scripts and notebooks