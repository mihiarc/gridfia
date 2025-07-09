# BigMap Documentation

Welcome to BigMap - a modern Python framework for analyzing forest biomass and species diversity using BIGMAP 2018 data.

## Overview

BigMap provides tools for:
- 🌲 Forest biomass analysis at 30m resolution
- 📊 Species diversity calculations (richness, Shannon, Simpson)
- 🗺️ Large-scale spatial data processing
- 🔌 REST API integration with FIA BIGMAP ImageServer
- 📈 Publication-ready visualizations

## Quick Links

- [Getting Started](user-guide/getting-started.md) - Installation and first steps
- [CLI Reference](cli-reference.md) - Command-line interface documentation
- [API Reference](api/index.md) - Python API documentation
- [Tutorials](tutorials/species-diversity-analysis.md) - Step-by-step guides

## Installation

```bash
# Using pip
pip install bigmap

# Using uv (recommended)
uv pip install bigmap

# Development installation
git clone https://github.com/yourusername/bigmap.git
cd bigmap
uv pip install -e ".[dev,test,docs]"
```

## Quick Example

```python
from bigmap.config import BigMapSettings, CalculationConfig
from bigmap.core.processors.forest_metrics import ForestMetricsProcessor

# Configure calculations
settings = BigMapSettings(
    output_dir="results",
    calculations=[
        CalculationConfig(name="species_richness", enabled=True),
        CalculationConfig(name="shannon_diversity", enabled=True)
    ]
)

# Run analysis
processor = ForestMetricsProcessor(settings)
results = processor.run_calculations("data/nc_biomass.zarr")
```

## Features

### 🚀 Modern Architecture
- Type-safe configuration with Pydantic v2
- Plugin-based calculation framework
- Memory-efficient chunked processing
- Comprehensive error handling

### 📊 Built-in Calculations
- Species richness and diversity indices
- Biomass metrics and comparisons
- Species dominance and presence
- Custom calculation support

### 🛠️ Developer Friendly
- Full type hints and docstrings
- Extensive test coverage
- Clear API design
- Rich CLI with progress tracking

## Documentation Structure

- **[User Guide](user-guide/getting-started.md)**: Installation, configuration, and usage
- **[API Reference](api/index.md)**: Detailed API documentation
- **[CLI Reference](cli-reference.md)**: Command-line interface guide
- **[Tutorials](tutorials/species-diversity-analysis.md)**: Step-by-step tutorials
- **[Architecture](architecture/system-design.md)**: System design and internals

## Contributing

We welcome contributions! Please see our [Contributing Guide](https://github.com/yourusername/bigmap/blob/main/CONTRIBUTING.md) for details.

## License

BigMap is released under the MIT License. See [LICENSE](https://github.com/yourusername/bigmap/blob/main/LICENSE) for details.