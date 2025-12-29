# Installation Guide

This guide covers installation of GridFIA for various use cases.

## Requirements

- **Python**: 3.9 or higher
- **Operating System**: Linux, macOS, or Windows
- **Disk Space**: ~500MB for package and dependencies; additional space for data

## Quick Installation

### Using pip

```bash
pip install gridfia
```

### Using uv (recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer:

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install GridFIA
uv pip install gridfia
```

## Verify Installation

Test that GridFIA is installed correctly:

```python
import gridfia
print(f"GridFIA v{gridfia.__version__} installed successfully!")

# Test API functionality
from gridfia import GridFIA
api = GridFIA()

# List species from BIGMAP (requires internet)
species = api.list_species()
print(f"Connected to BIGMAP: {len(species)} species available")
```

Run this test script:

```bash
python -c "from gridfia import GridFIA; api = GridFIA(); print(f'GridFIA ready: {len(api.list_species())} species available')"
```

## Installation Options

### Standard Installation

Includes core functionality for forest analysis:

```bash
pip install gridfia
```

### Development Installation

For contributing to GridFIA:

```bash
git clone https://github.com/mihiarc/gridfia.git
cd gridfia
uv pip install -e ".[dev,test,docs]"
```

### Documentation Build

To build documentation locally:

```bash
pip install gridfia[docs]
# Or with uv
uv pip install gridfia[docs]
```

## Dependencies

GridFIA installs the following dependencies automatically:

### Core Dependencies

| Package | Purpose |
|---------|---------|
| **numpy** | Numerical computing |
| **pandas** | Data manipulation |
| **xarray** | N-dimensional arrays |
| **zarr** | Cloud-optimized storage |
| **rasterio** | Geospatial raster I/O |
| **geopandas** | Geospatial vector operations |

### Visualization

| Package | Purpose |
|---------|---------|
| **matplotlib** | Plotting and maps |
| **rich** | Terminal output formatting |

### Configuration & Validation

| Package | Purpose |
|---------|---------|
| **pydantic** (>=2.0) | Data validation |
| **pydantic-settings** | Settings management |
| **requests** | HTTP client for BIGMAP API |

## Verify Dependencies

Check that all dependencies are available:

```python
import sys
import importlib

packages = [
    'numpy', 'pandas', 'xarray', 'zarr',
    'rasterio', 'geopandas', 'matplotlib',
    'rich', 'pydantic', 'requests'
]

missing = []
for pkg in packages:
    try:
        importlib.import_module(pkg)
        print(f"[OK] {pkg}")
    except ImportError:
        print(f"[MISSING] {pkg}")
        missing.append(pkg)

if missing:
    print(f"\nMissing: {', '.join(missing)}")
    print("Run: pip install " + " ".join(missing))
else:
    print("\nAll dependencies installed!")
```

## Virtual Environments

We recommend using virtual environments to avoid dependency conflicts.

### Using uv (recommended)

```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install GridFIA
uv pip install gridfia
```

### Using venv

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install GridFIA
pip install gridfia
```

### Using conda

```bash
# Create conda environment
conda create -n gridfia python=3.11
conda activate gridfia

# Install GridFIA
pip install gridfia
```

## Platform-Specific Notes

### macOS

On Apple Silicon (M1/M2/M3), some dependencies compile from source:

```bash
# Ensure Xcode command line tools are installed
xcode-select --install

# Install GridFIA
pip install gridfia
```

### Windows

Rasterio may require additional setup on Windows:

```bash
# Option 1: Use conda (handles binary dependencies)
conda install -c conda-forge rasterio
pip install gridfia

# Option 2: Install from wheel
pip install gridfia
```

### Linux

On Debian/Ubuntu, install GDAL system libraries first:

```bash
sudo apt-get update
sudo apt-get install -y gdal-bin libgdal-dev

pip install gridfia
```

## Troubleshooting

### Import Error: No module named 'gridfia'

Ensure you're using the correct Python environment:

```bash
# Check which Python
which python
python -c "import sys; print(sys.executable)"

# Reinstall in correct environment
pip install --force-reinstall gridfia
```

### GDAL/Rasterio Installation Issues

If rasterio fails to install:

```bash
# Use conda for binary distribution
conda install -c conda-forge rasterio geopandas
pip install gridfia
```

### Network Connection Errors

GridFIA requires internet access to download BIGMAP data. Test connectivity:

```python
from gridfia import GridFIA

api = GridFIA()
try:
    species = api.list_species()
    print(f"Connection successful: {len(species)} species")
except Exception as e:
    print(f"Connection failed: {e}")
```

### Memory Issues with Large Datasets

For large state-wide analyses, increase available memory or use chunked processing:

```python
from gridfia import GridFIA

api = GridFIA()

# Process county by county instead of entire state
zarr_path = api.create_zarr(
    input_dir="data/",
    output_path="data/forest.zarr",
    chunk_size=(1, 1000, 1000)  # Smaller chunks use less memory
)
```

## Upgrading

### Upgrade to Latest Version

```bash
pip install --upgrade gridfia
```

### Check Current Version

```python
import gridfia
print(gridfia.__version__)
```

## Uninstalling

```bash
pip uninstall gridfia
```

## Next Steps

After installation:

1. Read the [Quick Start Guide](../user-guide/getting-started.md)
2. Explore the [API Reference](../api/index.md)
3. Try the [Tutorials](../tutorials/species-diversity-analysis.md)
