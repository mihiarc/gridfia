# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

GridFIA is a Python API for forest biomass and species diversity analysis that processes BIGMAP 2018 forest data at 30m resolution for any US state, county, or custom region. It provides a clean programmatic interface for analyzing forest metrics, calculating species diversity indices, and downloading data from the FIA BIGMAP ImageServer.

**Part of the FIA Python Ecosystem:**
- **PyFIA**: Survey/plot data analysis (https://github.com/mihiarc/pyfia)
- **GridFIA**: Spatial raster analysis (this package)
- **PyFVS**: Growth/yield simulation (https://github.com/mihiarc/pyfvs)
- **AskFIA**: AI conversational interface (https://github.com/mihiarc/askfia)

## Architecture

### API-First Design

GridFIA uses a pure API architecture with no CLI, providing a single clean interface through the `GridFIA` class.

### Core Components

- **api.py**: Main API interface - single entry point for all functionality
- **external/**: External service clients (FIA BIGMAP REST client)
- **core/**: Main processing logic
  - **analysis/**: Species presence and statistical analysis modules
  - **calculations/**: Plugin-based calculation framework with registry pattern
  - **processors/**: Forest metrics processing (biomass, diversity indices)
- **utils/**: Parallel processing utilities for large-scale data operations
- **visualization/**: Matplotlib-based visualization components

### Data Flow

1. **Input**: Zarr arrays with forest data or downloads from REST API
2. **Processing**: Plugin-based calculations (Shannon, Simpson, richness indices)
3. **Output**: Analyzed data with statistics and optional visualizations

The system uses a registry pattern for calculations, allowing easy extension with new metrics.

## Development Commands

### Environment Setup
```bash
# Create virtual environment and install (using uv as per global instructions)
uv venv
uv pip install -e ".[dev,test,docs]"
```

### Using the API
```python
from gridfia import GridFIA

# Initialize API
api = GridFIA()

# List available species
species = api.list_species()

# Download species data
files = api.download_species(state="California", species_codes=["0202"])

# Create Zarr store
zarr_path = api.create_zarr("downloads/", "data/california.zarr")

# Calculate metrics
results = api.calculate_metrics(zarr_path, calculations=["species_richness"])

# Create visualizations
maps = api.create_maps(zarr_path, map_type="diversity")

# Get location configuration
config = api.get_location_config(state="Texas", county="Harris")
```

### Testing
```bash
# Run all tests with coverage
uv run pytest

# Run specific test file
uv run pytest tests/unit/test_processors.py

# Run with coverage report
uv run pytest --cov

# Run tests in parallel (if pytest-xdist is installed)
uv run pytest -n auto
```

### Code Quality
```bash
# Format code
uv run black gridfia/ tests/
uv run isort gridfia/ tests/

# Lint code
uv run flake8 gridfia/ tests/

# Type checking
uv run mypy gridfia/
```

### Documentation
```bash
# Serve documentation locally at http://127.0.0.1:8000
uv run mkdocs serve

# Build documentation
uv run mkdocs build
```

## Key Technical Details

### Dependencies
- **Core**: numpy, pandas, xarray, zarr, rasterio, geopandas
- **Visualization**: matplotlib, rich for progress bars
- **Validation**: pydantic v2 for configuration and data models
- **Testing**: pytest with 80% minimum coverage requirement

### Configuration System
- YAML-based configuration files in `cfg/` directory
- Pydantic v2 models for validation
- Support for species-specific configurations

### Location Configuration
The `LocationConfig` in `utils/location_config.py` handles any geographic location:
- Automatic state/county boundary detection
- State Plane CRS detection for each state
- Support for custom bounding boxes
- Template configurations in `config/templates/`

### External Service Integration
The `BigMapRestClient` in `external/fia_client.py` downloads species data from:
- Base URL: https://apps.fs.usda.gov/arcx/rest/services/RDW_Biomass
- Supports any geographic location (state, county, custom bbox)
- Progress tracking and chunked downloads
- Automatic retry logic for failed requests

### Testing Approach
- Unit tests separated from integration tests
- Rich fixtures in conftest.py for test data generation
- Real API calls (no mocking) as per global instructions
- Coverage requirements: 80% minimum

## Common Tasks

### Adding a New Calculation
1. Create a new class inheriting from `Calculation` in `core/calculations/`
2. Implement `calculate()` and `get_stats()` methods
3. Register with `@registry.register("name")` decorator
4. Add tests in `tests/unit/test_calculations.py`

### Processing New Species Data
```python
from gridfia import GridFIA

api = GridFIA()

# Download species data
files = api.download_species(
    state="Montana",
    species_codes=["0202", "0122"],  # Douglas-fir, Ponderosa Pine
    output_dir="data/montana"
)

# Create Zarr store
zarr_path = api.create_zarr("data/montana", "data/montana.zarr")

# Run analysis
results = api.calculate_metrics(
    zarr_path,
    calculations=["species_richness", "shannon_diversity", "total_biomass"]
)
```

### Using in Jupyter Notebooks
GridFIA is designed for interactive use in Jupyter notebooks:

```python
from gridfia import GridFIA
import pandas as pd

api = GridFIA()

# Explore species interactively
species = api.list_species()
species_df = pd.DataFrame([s.dict() for s in species])
species_df.head()

# Process and visualize
zarr_path = api.create_zarr("downloads/", "data.zarr")
results = api.calculate_metrics(zarr_path)
maps = api.create_maps(zarr_path, map_type="diversity")
```
