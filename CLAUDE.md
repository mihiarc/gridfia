# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

BigMap is a North Carolina forest biomass and species diversity analysis toolkit that processes BIGMAP 2018 forest data at 30m resolution. It provides tools for analyzing forest metrics, calculating species diversity indices, and downloading data from the FIA BIGMAP ImageServer REST API.

## Architecture

### Core Components

- **api/**: REST client for FIA BIGMAP ImageServer integration
- **cli/**: Typer-based command-line interface
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

### Running the Application
```bash
# Main CLI commands
uv run bigmap --help                                    # Show all commands
uv run bigmap calculate data.zarr --config config.yaml  # Run calculations
uv run bigmap list-species                              # List available species
uv run bigmap download --species 0131 --output data/    # Download species data
uv run bigmap config --show                             # Show configuration
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
uv run black bigmap/ tests/
uv run isort bigmap/ tests/

# Lint code  
uv run flake8 bigmap/ tests/

# Type checking
uv run mypy bigmap/
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
- **CLI**: typer with rich for beautiful terminal output
- **Validation**: pydantic v2 for configuration and data models
- **Testing**: pytest with 80% minimum coverage requirement

### Configuration System
- YAML-based configuration files in `cfg/` directory
- Pydantic v2 models for validation
- Support for species-specific configurations

### REST API Integration
The `BigMapRestClient` in `api/` downloads species data from:
- Base URL: https://apps.fs.usda.gov/arcx/rest/services/RDW_Biomass
- Supports progress tracking and chunked downloads
- Automatic retry logic for failed requests

### Testing Approach
- Unit tests separated from integration tests
- Rich fixtures in conftest.py for test data generation
- Real API calls (no mocking) as per global instructions
- Coverage requirements: 80% minimum

## Cursor Rules Integration

This project includes Cursor rules that affect code style:
- **Logic explanation**: Responses start with "Here is my logic:"
- **Root cause analysis**: Focus on root causes with "The root cause is:"
- **Teaching mode**: Educational responses begin with "Teacher mode:"

## Common Tasks

### Adding a New Calculation
1. Create a new class inheriting from `Calculation` in `core/calculations/`
2. Implement `calculate()` and `get_stats()` methods
3. Register with `@registry.register("name")` decorator
4. Add tests in `tests/unit/test_calculations.py`

### Processing New Species Data
1. Download data: `uv run bigmap download --species XXXX --output data/`
2. Create configuration in `cfg/species/species_XXXX.yaml`
3. Run analysis: `uv run bigmap calculate data/species_XXXX.zarr --config cfg/species/species_XXXX.yaml`

### Extending the CLI
1. Add new command in `cli/main.py` using `@app.command()` decorator
2. Use type hints for arguments (Typer will handle parsing)
3. Use rich console for output formatting
4. Add corresponding tests in `tests/unit/test_cli.py`