# Montana Forest Analysis Pipeline

This project analyzes forest biomass and species distribution in Montana using BIGMAP 2018 data.

## Pipeline Overview

The analysis pipeline consists of the following stages:

### Stage 1: Download Species Data
```bash
uv run python scripts/01_download_montana_species_state_plane.py
```
Downloads forest species biomass data from the FIA BIGMAP ImageServer for Montana in State Plane coordinates.

### Stage 2: Build Zarr Store
```bash
uv run python scripts/02_build_montana_zarr_clipped.py
```
Combines downloaded species data into a zarr store, clips to Montana boundaries, and adds a timber total layer.

### Stage 3: Add Dominant Species Layer
```bash
uv run python scripts/03_add_dominant_species_layer.py
```
Calculates and adds a dominant species classification layer showing which species has the highest biomass at each location.

### Stage 4: Generate Mill Isochrone (if needed)
```bash
# Using SocialMapper to generate isochrones
uv run python scripts/generate_mill_isochrones.py

# Or for custom travel times and modes
uv run python scripts/generate_mill_isochrones.py -t 60 -t 90 -t 120 --mode drive
```
Generates drive-time isochrone data for the mill location using the SocialMapper package. See [Isochrone Generation Guide](docs/isochrone_generation.md) for detailed instructions.

### Stage 5: Create Main Map
```bash
uv run python scripts/04_create_main_map.py
```
Creates the main visualization map showing dominant tree species within the 120-minute drive time supply area.

### Stage 6: Create Boundary Overview Map
```bash
uv run python scripts/05_create_boundary_overview_map.py
```
Creates an overview map showing state boundaries, county boundaries, isochrone, and mill location.

### Stage 7: Analyze Mill Supply Area
```bash
uv run python scripts/06_analyze_mill_isochrone_biomass.py
```
Analyzes biomass within the 120-minute drive time of the mill and generates a detailed report.

## Output Structure

All outputs are saved to the `output/` directory:
- `mill_dominant_species_isochrone.png` - Main map showing dominant species in mill supply area
- `montana_boundary_overview.png` - Geographic context map with boundaries and isochrone
- `mill_site_analysis_report.md` - Detailed biomass analysis report
- `mill_isochrone_120min.geojson` - Extracted isochrone geometry
- `data/` - Zarr store and downloaded data files

## Configuration

Project settings are managed in `config/montana_project.yml`, including:
- Species list and codes
- Coordinate reference systems
- Bounding boxes
- Layer indices
- Output paths

## Requirements

- Python 3.11+
- See `pyproject.toml` for full dependency list
- SocialMapper package (for isochrone generation)

## Installation

```bash
uv venv
uv pip install -e ".[dev]"

# Install SocialMapper for isochrone generation
uv pip install socialmapper
```