# Processors API Reference

The processors module provides high-level interfaces for running forest metric calculations on large-scale biomass data.

## ForestMetricsProcessor

::: gridfia.core.processors.forest_metrics.ForestMetricsProcessor
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      members_order: source

### Usage Example

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

!!! note
    Most users should use `GridFIA.calculate_metrics()` instead of this class
    directly. The processor is exposed for advanced use cases that need finer
    control over the processing pipeline.

## Convenience Function

::: gridfia.core.processors.forest_metrics.run_forest_analysis
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

## Processing Features

### Chunked Processing

The processor automatically divides large arrays into chunks for memory-efficient processing:

- Default chunk size: `(1, 1000, 1000)` (species, height, width)
- Progress tracking with Rich

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
