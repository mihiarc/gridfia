# API Reference

The BigMap API provides a comprehensive set of tools for forest biomass and species diversity analysis.

## Core Modules

### [Processors](processors.md)
High-level interfaces for running forest calculations on large datasets.

- `ForestMetricsProcessor`: Main processor for forest metric calculations
- `run_forest_analysis`: Convenience function for quick analysis

### [Calculations](calculations.md)
Flexible framework for implementing forest metrics.

- `ForestCalculation`: Abstract base class for all calculations
- Registry system for managing available calculations
- Built-in calculations for diversity, biomass, and species analysis

### [Configuration](config.md)
Type-safe configuration management using Pydantic v2.

- `BigMapSettings`: Main settings class
- `CalculationConfig`: Individual calculation configuration
- Environment variable and file-based configuration

## Quick Reference

### Running Calculations

```python
from bigmap.config import BigMapSettings, CalculationConfig
from bigmap.core.processors.forest_metrics import ForestMetricsProcessor

# Configure
settings = BigMapSettings(
    output_dir="results",
    calculations=[
        CalculationConfig(name="species_richness", enabled=True),
        CalculationConfig(name="shannon_diversity", enabled=True)
    ]
)

# Process
processor = ForestMetricsProcessor(settings)
results = processor.run_calculations("data.zarr")
```

### Using the Registry

```python
from bigmap.core.calculations import registry

# List available calculations
calcs = registry.list_calculations()

# Get calculation instance
calc = registry.get('species_richness', biomass_threshold=1.0)

# Run calculation
result = calc.calculate(biomass_data)
```

### Custom Calculations

```python
from bigmap.core.calculations import ForestCalculation, register_calculation

class MyMetric(ForestCalculation):
    def __init__(self):
        super().__init__(
            name="my_metric",
            description="Custom forest metric",
            units="custom"
        )
    
    def calculate(self, biomass_data, **kwargs):
        # Implementation
        return result
    
    def validate_data(self, biomass_data):
        return biomass_data.ndim == 3

# Register
register_calculation("my_metric", MyMetric)
```

## Module Structure

```
bigmap/
├── api/              # REST API client
│   └── rest_client.py
├── cli/              # Command-line interface
│   └── main.py
├── config.py         # Configuration management
├── core/
│   ├── calculations/ # Calculation framework
│   │   ├── base.py
│   │   ├── biomass.py
│   │   ├── diversity.py
│   │   ├── registry.py
│   │   └── species.py
│   └── processors/   # High-level processors
│       └── forest_metrics.py
└── utils/           # Utility functions
```

## Key Features

### Memory Efficiency
- Chunked processing for large datasets
- Configurable chunk sizes
- Progress tracking

### Flexibility
- Plugin-based calculation system
- Multiple output formats (GeoTIFF, NetCDF, Zarr)
- Customizable parameters

### Type Safety
- Full type hints throughout
- Pydantic validation
- Runtime type checking

### Integration
- REST API client for data access
- CLI for scripting
- Python API for programmatic use

## Error Handling

All modules include comprehensive error handling:

```python
try:
    results = processor.run_calculations(zarr_path)
except ValueError as e:
    # Handle validation errors
    print(f"Invalid input: {e}")
except Exception as e:
    # Handle other errors
    print(f"Processing failed: {e}")
```

## Performance Tips

1. **Chunk Size**: Adjust based on available memory
   ```python
   processor.chunk_size = (1, 2000, 2000)  # Larger chunks
   ```

2. **Parallel Processing**: Use multiple calculations
   ```python
   # Calculations run independently per chunk
   settings.calculations = [multiple_calculations]
   ```

3. **Output Format**: Choose based on use case
   - GeoTIFF: Best for GIS integration
   - NetCDF: Best for xarray workflows
   - Zarr: Best for large outputs

## See Also

- [User Guide](../user-guide/getting-started.md)
- [CLI Reference](../cli-reference.md)
- [Tutorials](../tutorials/index.md)
- [Examples](https://github.com/yourusername/bigmap/tree/main/examples)