# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development
```bash
# Run tests
pytest
pytest --cov=bigmap --cov-report=html  # with coverage report
pytest tests/test_core.py  # run specific test file

# Code formatting
black bigmap/
isort bigmap/

# Linting and type checking
flake8 bigmap/
mypy bigmap/

# Run all quality checks
pre-commit run --all-files

# Documentation
mkdocs build    # build docs
mkdocs serve    # serve docs locally at http://127.0.0.1:8000
```

### CLI Commands
```bash
# Main CLI entry point
bigmap --help

# Available commands:
bigmap calculate    # Run forest metric calculations
bigmap config       # Manage configuration files
bigmap list-species # List available species from REST API
bigmap download     # Download species data via REST API
```

## Architecture

### Core Components

1. **Calculation Framework** (`bigmap/core/calculations.py`)
   - Plugin-based architecture for different forest metrics
   - Available calculations: species_richness, total_biomass, shannon_diversity, dominant_species, species_proportions, species_percentages, species_group_proportions
   - Processes data in chunks for memory efficiency

2. **CLI System** (`bigmap/cli/main.py`)
   - Built with Typer for type-safe command handling
   - Rich console output for better user experience
   - Commands organized by functionality (calculate, compare, config, download)

3. **Data Processing**
   - Works with BIGMAP 2018 forest data (30m resolution GeoTIFF files)
   - Uses Zarr arrays for efficient storage and processing
   - Handles species data through REST API integration
   - Supports batch processing of multiple species

4. **Configuration System**
   - YAML-based configuration files in `cfg/` directory
   - Templates for different analysis types (diversity, comparison, species proportions)
   - Pydantic models for validation

### Key Design Patterns

1. **Chunked Processing**: Large raster datasets are processed in configurable chunks to manage memory usage
2. **Plugin Architecture**: New calculations can be added by implementing the base calculation interface
3. **Async Downloads**: Species data downloads use asyncio for concurrent operations
4. **Type Safety**: Extensive use of type hints and Pydantic models for validation

### Data Flow

1. Raw BIGMAP GeoTIFF files → Clip to NC boundaries → Create Zarr arrays
2. Zarr arrays + Configuration → Calculate metrics → Generate outputs (maps, statistics)
3. REST API → Download species data → Append to existing Zarr arrays

### Important Paths

- Species data is typically stored in Zarr arrays
- Configuration files should be placed in `cfg/` directory
- Output maps and analysis results go to user-specified directories
- The system expects ESRI:102039 projection for spatial data

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.