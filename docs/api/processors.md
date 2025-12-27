# Processors API Reference

The processors module provides high-level interfaces for running forest metric calculations on large-scale biomass data.

## ForestMetricsProcessor

The main processor class for running forest calculations on zarr arrays.

### Class Definition

```python
class ForestMetricsProcessor:
    """
    Processor for running forest metric calculations on zarr arrays.
    
    This class handles:
    - Loading and validating zarr arrays
    - Running calculations from the registry
    - Memory-efficient chunked processing
    - Saving results in multiple formats
    """
```

### Constructor

```python
ForestMetricsProcessor(settings: Optional[GridFIASettings] = None)
```

**Parameters:**
- `settings` (GridFIASettings, optional): Configuration settings. If None, uses default settings.

### Methods

#### run_calculations

```python
run_calculations(zarr_path: str) -> Dict[str, str]
```

Run forest metric calculations on zarr data.

**Parameters:**
- `zarr_path` (str): Path to the zarr array containing biomass data

**Returns:**
- Dict[str, str]: Dictionary mapping calculation names to output file paths

**Example:**
```python
from gridfia.config import GridFIASettings, CalculationConfig
from gridfia.core.processors.forest_metrics import ForestMetricsProcessor

# Configure settings
settings = GridFIASettings(
    output_dir="results",
    calculations=[
        CalculationConfig(name="species_richness", enabled=True),
        CalculationConfig(name="total_biomass", enabled=True)
    ]
)

# Run calculations
processor = ForestMetricsProcessor(settings)
results = processor.run_calculations("data/biomass.zarr")

# Results: {'species_richness': 'results/species_richness.tif', ...}
```

### Convenience Function

#### run_forest_analysis

```python
run_forest_analysis(
    zarr_path: str, 
    config_path: Optional[str] = None
) -> Dict[str, str]
```

Run forest analysis with the given configuration.

**Parameters:**
- `zarr_path` (str): Path to zarr array
- `config_path` (str, optional): Path to configuration file

**Returns:**
- Dict[str, str]: Results dictionary mapping calculation names to output paths

## Processing Features

### Chunked Processing

The processor automatically divides large arrays into chunks for memory-efficient processing:

- Default chunk size: `(1, 1000, 1000)` (species, height, width)
- Configurable via `processor.chunk_size` attribute
- Progress tracking with tqdm

### Output Formats

Supports multiple output formats:
- **GeoTIFF** (`.tif`): Default format with spatial metadata
- **NetCDF** (`.nc`): For xarray compatibility
- **Zarr** (`.zarr`): For efficient storage and access

### Zarr Array Requirements

Input zarr arrays must have:
- 3 dimensions: `(species, y, x)`
- Required attributes:
  - `species_codes`: List of species identifiers
  - `crs`: Coordinate reference system
- Optional attributes:
  - `transform`: Affine transformation matrix
  - `bounds`: Spatial extent
  - `species_names`: Human-readable species names

## Error Handling

The processor includes comprehensive error handling:
- Validates zarr array structure and metadata
- Handles missing calculations gracefully
- Logs detailed error information
- Returns partial results if some calculations fail

## Performance Considerations

- **Memory Usage**: Controlled by chunk size
- **Parallel Processing**: Each chunk processed independently
- **I/O Optimization**: Efficient zarr reading and result writing
- **Progress Tracking**: Visual feedback during processing